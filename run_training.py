import os
import numpy as np
import time
import configparser
from collections import defaultdict
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from utils.misc import str2bool, parseArguments
from utils.train_fns import run_iter, create_loss_weight_interpolator
from utils.model import build_model
from utils.dataloaders import build_h5_dataloader, build_fits_dataloader
from utils.plotting_fns import plot_progress3, plot_batch
from utils.eval_fns import mae_predict

def main(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    n_gpu = torch.cuda.device_count()

    print(f'Using Torch version: {torch.__version__}')
    print(f'Using a {device} device with {n_gpu} GPU(s)')

    # Directories
    cur_dir = os.path.dirname(__file__)
    config_dir = os.path.join(cur_dir, 'configs/')
    model_dir = os.path.join(cur_dir, 'models/')
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(cur_dir, 'data/')
    fig_dir = os.path.join(cur_dir, 'figures/')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    # Load model configuration
    model_name = args.model_name
    config = configparser.ConfigParser()
    config.read(config_dir+model_name+'.ini')

    # Display model configuration
    print('\nCreating model: %s'%model_name)
    print('\nConfiguration:')
    for key_head in config.keys():
        if key_head=='DEFAULT':
            continue
        print('  %s' % key_head)
        for key in config[key_head].keys():
            print('    %s: %s'%(key, config[key_head][key]))

    # Construct the model, optimizer, etc.
    model_filename =  os.path.join(model_dir, model_name+'.pth.tar') 
    model, losses, cur_iter, optimizer, lr_scheduler = build_model(config, model_filename, 
                                                                   device, build_optimizer=True)


    # Data loader stuff
    num_workers = min([os.cpu_count(),12*n_gpu])
    if num_workers>1:
        num_workers -=1
    num_workers = num_workers//2
        
    # Masking stuff
    if 'mim' in config['ARCHITECTURE']['model_type']:
        mask_ratio = None
        max_mask_ratio = float(config['MIM TRAINING']['max_mask_ratio'])
    else:
        mask_ratio = float(config['MIM TRAINING']['mask_ratio'])
        max_mask_ratio = None

    # Build dataloaders
    if 'train_mim_data_file' in config['DATA']:
        # Using .h5 training file
        dataloader_mim_train = build_h5_dataloader(os.path.join(data_dir, config['DATA']['train_mim_data_file']), 
                                                batch_size=int(config['MIM TRAINING']['batch_size']), 
                                                num_workers=num_workers,
                                                patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                                num_channels=int(config['ARCHITECTURE']['num_channels']), 
                                                max_mask_ratio=max_mask_ratio, 
                                                img_size=int(config['ARCHITECTURE']['img_size']),
                                                num_patches=model.module.patch_embed.num_patches,
                                                shuffle=True)
        print('The training set consists of %i cutouts.' % (len(dataloader_train.dataset)))
        train_nested_batches = False
    else:
        # Using fits files in training directory
        # Might need to decrease num_workers and increase cutouts_per_tile
        dataloader_mim_train =  build_fits_dataloader(eval(config['DATA']['train_mim_data_paths']), 
                                                  bands=eval(config['DATA']['bands']), 
                                                  min_bands=int(config['DATA']['min_bands']), 
                                                  batch_size=int(config['MIM TRAINING']['batch_size']),
                                                  num_workers=num_workers,
                                                  patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                                  max_mask_ratio=max_mask_ratio, 
                                                  img_size=int(config['ARCHITECTURE']['img_size']), 
                                                  cutouts_per_tile=int(config['DATA']['cutouts_per_tile']), 
                                                  use_calexp=str2bool(config['DATA']['use_calexp']),
                                                  ra_dec=True,
                                                  augment=False, 
                                                  shuffle=True)
        train_nested_batches = True
    
    dataloader_mim_val = build_h5_dataloader(os.path.join(data_dir, config['DATA']['val_mim_data_file']), 
                                          batch_size=int(config['MIM TRAINING']['batch_size']), 
                                          num_workers=num_workers,
                                          patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                          num_channels=int(config['ARCHITECTURE']['num_channels']), 
                                          max_mask_ratio=max_mask_ratio, 
                                          img_size=int(config['ARCHITECTURE']['img_size']),
                                          num_patches=model.module.patch_embed.num_patches,
                                          shuffle=True)

    # Dataloaders for supervised training
    num_train = int(config['SUPERVISED TRAINING']['num_train'])
    if num_train>-1:
        if 'crossentropy' in config['SUPERVISED TRAINING']['loss_fn'].lower():
            train_indices = select_training_indices(os.path.join(data_dir, config['DATA']['train_data_file']), 
                                                    num_train, balanced=False)
        else:    
            train_indices = range(num_train)
    else:
        train_indices = None
    dataloader_sup_train = build_h5_dataloader(os.path.join(data_dir, config['DATA']['train_labels_data_file']), 
                                           batch_size=int(config['SUPERVISED TRAINING']['batch_size']), 
                                           num_workers=num_workers,
                                           label_keys=eval(config['DATA']['label_keys']),
                                           img_size=int(config['ARCHITECTURE']['img_size']),
                                           patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                           num_channels=int(config['ARCHITECTURE']['num_channels']), 
                                           num_patches=model.module.patch_embed.num_patches,
                                           augment=str2bool(config['SUPERVISED TRAINING']['augment']),
                                           brightness=float(config['SUPERVISED TRAINING']['brightness']), 
                                           noise=float(config['SUPERVISED TRAINING']['noise']), 
                                           nan_channels=int(config['SUPERVISED TRAINING']['nan_channels']),
                                           shuffle=True,
                                           indices=train_indices)
    
    dataloader_sup_val = build_h5_dataloader(os.path.join(data_dir, config['DATA']['val_labels_data_file']), 
                                        batch_size=int(config['SUPERVISED TRAINING']['batch_size']), 
                                        num_workers=num_workers,
                                        label_keys=eval(config['DATA']['label_keys']),
                                        img_size=int(config['ARCHITECTURE']['img_size']),
                                        patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                         num_channels=int(config['ARCHITECTURE']['num_channels']), 
                                        num_patches=model.module.patch_embed.num_patches,
                                        shuffle=True)


    # Supervised training params
    get_sup_loss_weight = create_loss_weight_interpolator(float(config['SUPERVISED TRAINING']['init_loss_weight']), 
                                                      float(config['SUPERVISED TRAINING']['final_loss_weight']), 
                                                      int(float(config['SUPERVISED TRAINING']['loss_weight_plateau_batch_iters'])))

    # Training iterations paramaters are all measured from starting point
    total_batch_iters = int(float(config['MIM TRAINING']['total_batch_iters']))
    stop_mim_batch_iters = int(float(config['MIM TRAINING']['stop_mim_batch_iters']))
    start_sup_batch_iters = int(float(config['SUPERVISED TRAINING']['start_batch_iters']))

        
    train_network(model, dataloader_mim_train, dataloader_mim_val, train_nested_batches,
                  dataloader_sup_train, dataloader_sup_val, 
                  optimizer, lr_scheduler, device,
                  mask_ratio, losses, cur_iter,  get_sup_loss_weight,
                  total_batch_iters, stop_mim_batch_iters, start_sup_batch_iters,
                  args.verbose_iters, args.cp_time, model_filename, fig_dir)

def get_train_samples(dataloader, train_nested_batches):
    '''Accomodates both dataloaders.'''
    if train_nested_batches:
        # Iterate through all of the tiles
        for sample_batches, masks, ra_decs in dataloader:
            # Iterate through each batch of images in this tile of the sky
            for samples, mask, ra_dec in zip(sample_batches[0], masks[0], ra_decs[0]):
                yield samples, mask, ra_dec
    else:
        for samples, mask, ra_dec in dataloader:
            yield samples, mask, ra_dec

def train_network(model, dataloader_mim_train, dataloader_mim_val, train_nested_batches,
                  dataloader_sup_train, dataloader_sup_val, 
                  optimizer, lr_scheduler, device,
                  mask_ratio, losses, cur_iter,  get_sup_loss_weight, 
                  total_batch_iters, stop_mim_batch_iters, start_sup_batch_iters,
                  verbose_iters, cp_time, model_filename, fig_dir):
    print('Training the network with a batch size of %i per GPU ...' % (dataloader_mim_train.batch_size))
    print('Progress will be displayed every %i batch iterations and the model will be saved every %i minutes.'%
          (verbose_iters, cp_time))
    
    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    #time1 = time.time()

    dataloader_mim_train_iterator = iter(get_train_samples(dataloader_mim_train, train_nested_batches))
    dataloader_sup_train_iterator = iter(dataloader_sup_train)
    dataloader_sup_val_iterator = iter(dataloader_sup_val)
    while cur_iter < (total_batch_iters):

        if cur_iter < stop_mim_batch_iters:
            # Iterate through MIM training dataset
            try:
                samples, masks, ra_decs = next(dataloader_mim_train_iterator)
            except StopIteration:
                dataloader_mim_train_iterator = iter(get_train_samples(dataloader_mim_train, train_nested_batches))
                samples, masks, ra_decs = next(dataloader_mim_train_iterator)
            # Switch to GPU if available
            samples = samples.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            ra_decs = ra_decs.to(device, non_blocking=True)
        else:
            samples, masks, ra_decs = None, None, None
        if cur_iter >= start_sup_batch_iters:
            # Iterate through Supervised training dataset
            try:
                samples_sup, sample_masks_sup, ra_decs_sup, sample_labels_sup = next(dataloader_sup_train_iterator)
            except StopIteration:
                dataloader_sup_train_iterator = iter(dataloader_sup_train)
                samples_sup, sample_masks_sup, ra_decs_sup, sample_labels_sup = next(dataloader_sup_train_iterator)
            # Switch to GPU if available
            samples_sup = samples_sup.to(device, non_blocking=True)
            sample_masks_sup = sample_masks_sup.to(device, non_blocking=True)
            ra_decs_sup = ra_decs_sup.to(device, non_blocking=True)
            sample_labels_sup = sample_labels_sup.to(device, non_blocking=True)
        else:
            samples_sup, sample_masks_sup, ra_decs_sup, sample_labels_sup = None, None, None, None

        sup_loss_weight = get_sup_loss_weight(cur_iter)
        
        # Run an iteration of training
        model, optimizer, lr_scheduler, losses_cp = run_iter(model, samples, ra_decs, masks, mask_ratio, 
                                                         samples_sup, ra_decs_sup, sample_masks_sup, sample_labels_sup, 
                                                         optimizer, lr_scheduler, sup_loss_weight,
                                                         losses_cp, mode='train')
        
        # Evaluate validation set and display progress
        if cur_iter % verbose_iters == 0:

            with torch.no_grad():
                # Calculate average loss on validation set
                for i, (samples, masks, ra_decs) in enumerate(dataloader_mim_val):
                
                    try:
                        samples_sup, sample_masks_sup, ra_decs_sup, sample_labels_sup = next(dataloader_sup_val_iterator)
                    except StopIteration:
                        dataloader_sup_val_iterator = iter(dataloader_sup_val)
                        samples_sup, sample_masks_sup, ra_decs_sup, sample_labels_sup = next(dataloader_sup_val_iterator)
                        
                    # Switch to GPU if available
                    samples = samples.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    ra_decs = ra_decs.to(device, non_blocking=True)
                
                    samples_sup = samples_sup.to(device, non_blocking=True)
                    sample_masks_sup = sample_masks_sup.to(device, non_blocking=True)
                    ra_decs_sup = ra_decs_sup.to(device, non_blocking=True)
                    sample_labels_sup = sample_labels_sup.to(device, non_blocking=True)
                
                    # Run an iteration
                    model, optimizer, lr_scheduler, losses_cp = run_iter(model, samples, ra_decs, masks, mask_ratio, 
                                                         samples_sup, ra_decs_sup, sample_masks_sup, sample_labels_sup, 
                                                         optimizer, lr_scheduler, sup_loss_weight,
                                                         losses_cp, mode='val')
                    # Don't bother with the whole dataset
                    if i>=200:
                        break
                
            # Calculate averages
            for k in losses_cp.keys():
                losses[k].append(np.mean(np.array(losses_cp[k]), axis=0))
            losses['batch_iters'].append(cur_iter)
                    
            # Print current status
            print('\nBatch Iterations: %i/%i ' % (cur_iter, total_batch_iters))
            print('Losses:')
            print('\tTraining Dataset')
            print('\t\tMIM Loss: %0.5f'% (losses['mim_train_loss'][-1]))
            print('\t\tSupervised Loss: %0.5f'% (losses['sup_train_loss'][-1]))
            if 'crossentropy' in model.module.sup_loss_fn.lower():
                print('\t\tSupervised Accuracy: %0.5f'% (losses['sup_train_acc'][-1]))
            else:
                print('\t\tSupervised MAE: %0.5f'% (losses['sup_train_mae'][-1]))
            print('\tValidation Dataset')
            print('\t\tMIM Loss: %0.5f'% (losses['mim_val_loss'][-1]))
            print('\t\tSupervised Loss: %0.5f'% (losses['sup_val_loss'][-1]))
            if 'crossentropy' in model.module.sup_loss_fn.lower():
                print('\t\tSupervised Accuracy: %0.5f'% (losses['sup_val_acc'][-1]))
            else:
                print('\t\tSupervised MAE: %0.5f'% (losses['sup_val_mae'][-1]))

            # Reset checkpoint loss dictionary
            losses_cp = defaultdict(list)

            if len(losses['batch_iters'])>1:
                # Plot progress
                plot_progress3(losses, y_lims=[(0.5,0.7), (0.,0.02), (0., 0.1)], 
                              savename=os.path.join(fig_dir, 
                                                    f'{os.path.basename(model_filename).split(".")[0]}_progress.png'))
            # Plot 5 validation samples
            pred_imgs, mask_imgs, orig_imgs = mae_predict(model, dataloader_mim_val, 
                                                              device, 
                                                              mask_ratio, 
                                                              single_batch=True)
            plot_batch(orig_imgs, mask_imgs, pred_imgs, n_samples=5, channel_index=0,
                           savename=os.path.join(fig_dir, 
                                                 f'{os.path.basename(model_filename).split(".")[0]}_{cur_iter}iters.png'))

        # Increase the iteration
        cur_iter += 1
        
        if (time.time() - cp_start_time) >= cp_time*60:
        
            # Save periodically
            print('Saving network...')
            torch.save({'batch_iters': cur_iter,
                        'losses': losses,
                        'optimizer' : optimizer.state_dict(),
                        'lr_scheduler' : lr_scheduler.state_dict(),
                        'model' : model.module.state_dict()},
                        model_filename)
        
            cp_start_time = time.time()
        
        if cur_iter > total_batch_iters:
        
            # Save after training
            print('Saving network...')
            torch.save({'batch_iters': cur_iter,
                        'losses': losses,
                        'optimizer' : optimizer.state_dict(),
                        'lr_scheduler' : lr_scheduler.state_dict(),
                        'model' : model.module.state_dict()},
                        model_filename)
            # Finish training
            break 

# Run the training
if __name__=="__main__":
    args = parseArguments()
    args = args.parse_args()
    main(args)
    
    print('\nTraining complete.')