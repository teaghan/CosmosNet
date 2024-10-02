import h5py
import numpy as np
import torch
import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from dataloaders import build_h5_dataloader
from eval_fns import mae_latent
from misc import select_centre

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def create_loss_weight_interpolator(initial_value, final_value, start_iter, max_iter):
    """
    Creates a function for linear interpolation of the loss weight.
    
    Args:
    - initial_value (float): Initial loss weight.
    - final_value (float): Final loss weight.
    - max_iter (int): Maximum number of iterations for training.
    
    Returns:
    - function: A function that computes the interpolated loss weight based on the current iteration.
    """
    def interpolate_loss_weight(current_iter):
        """
        Linearly interpolates the loss weight based on current iteration.
        
        Args:
        - current_iter (int): Current iteration in training.
        
        Returns:
        - float: Interpolated loss weight.
        """
        if current_iter < start_iter:
            return initial_value
        elif current_iter >= max_iter:
            return final_value
        else:
            return initial_value + (final_value - initial_value) * (current_iter / max_iter)
    
    return interpolate_loss_weight

def supervised_loss(model, tgt_labels, pred_labels):
    # Compute loss
    if 'crossentropy' in model.module.sup_loss_fn.lower():
        loss = torch.nn.CrossEntropyLoss()(pred_labels, tgt_labels.squeeze(1))
        with torch.no_grad():
            metric = (torch.max(pred_labels, 1)[1] == tgt_labels.squeeze(1)).float().mean()
    else:
        tgt_labels = model.module.normalize_labels(tgt_labels)
        if 'mse' in model.module.sup_loss_fn.lower():
            loss = torch.nn.MSELoss()(pred_labels, tgt_labels)
        elif 'huber' in model.module.sup_loss_fn.lower():
            loss = torch.nn.functional.huber_loss(pred_labels, tgt_labels, delta=1.0)
        
        with torch.no_grad():
            metric = torch.nn.L1Loss()(pred_labels, tgt_labels)
    
    if loss.numel() > 1:
        # In case of multiple GPUs
        loss = loss.unsqueeze(0).mean()
        with torch.no_grad():
            metric = metric.unsqueeze(0).mean()
    
    return loss, metric

def run_iter(model, 
             mim_samples, mim_ra_decs, mim_masks, mask_ratio, 
             sup_samples, sup_ra_decs, sup_masks, sup_labels,
             optimizer, lr_scheduler, sup_loss_weight,
             losses_cp, mode='train'):
        
    if mode=='train':
        model.train(True)
    else:
        model.train(False)

    loss = 0.
    if mim_samples is not None:
        # Run MIM predictions and calculate loss
        mim_loss, _, _ = model(mim_samples, ra_dec=mim_ra_decs, 
                       mask_ratio=mask_ratio, mask=mim_masks, 
                       run_mim=True, run_pred=False)
        if mim_loss.numel()>1:
            # In case of multiple GPUs
            mim_loss = mim_loss.unsqueeze(0).mean()
        loss = loss + mim_loss
    else:
        mim_loss = np.nan

    if sup_samples is not None:
        # Run Supervised predictions
        sup_preds = model(sup_samples, ra_dec=sup_ra_decs, 
                       mask_ratio=mask_ratio, mask=sup_masks, 
                       run_mim=False, run_pred=True)
        # Calculate supervised loss
        sup_loss, sup_metric = supervised_loss(model, sup_labels, sup_preds)
        
        # Combine losses
        loss = loss + sup_loss_weight*sup_loss
    else:
        sup_loss, sup_metric = np.nan, np.nan
    
    if 'train' in mode:
        
        # Update the gradients
        loss.backward()
        # Adjust network weights
        optimizer.step()
        # Reset gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Adjust learning rate
        lr_scheduler.step()
        
        # Save loss and metrics
        losses_cp['mim_train_loss'].append(float(mim_loss))
        losses_cp['sup_train_loss'].append(float(sup_loss))
        if 'crossentropy' in model.module.sup_loss_fn.lower():
            losses_cp['sup_train_acc'].append(float(sup_metric))
        else: 
            losses_cp['sup_train_mae'].append(float(sup_metric))

    else:
        # Save loss and metrics
        losses_cp['mim_val_loss'].append(float(mim_loss))
        losses_cp['sup_val_loss'].append(float(sup_loss))
        if 'crossentropy' in model.module.sup_loss_fn.lower():
            losses_cp['sup_val_acc'].append(float(sup_metric))
        else: 
            losses_cp['sup_val_mae'].append(float(sup_metric))
                
    return model, optimizer, lr_scheduler, losses_cp

def linear_probe(model, losses_cp, device, dataloader_template, class_data_path=None,
                 regress_data_path=None, combine='central', remove_cls=True):
    '''Train a quick linear probing model to evaluate the quality of the embeddings.'''

    if combine=='token':
        remove_cls = False
    
    model.train(False)
    if class_data_path:
        # Classifier task
        x,y = get_embeddings(class_data_path, 
                             model, device, dataloader_template,
                             y_label='class', combine=combine, remove_cls=remove_cls)
        
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Creating and training a classifier
        clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000, C=0.01, random_state=42)
        clf.fit(X_train, y_train)
        
        # Predicting the class label
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        
        # Evaluating the classifier
        test_accuracy = accuracy_score(y_test, y_pred_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
    
        losses_cp['train_lp_acc'].append(float(train_accuracy))
        losses_cp['val_lp_acc'].append(float(test_accuracy))
    if regress_data_path:
        # Regression task
        x,y = get_embeddings(regress_data_path, 
                             model, device, dataloader_template,
                             y_label='zspec', combine=combine, remove_cls=remove_cls)
    
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Creating and training a linear model for regression
        #regressor = LinearRegression()
        regressor = ElasticNet(alpha=0.0001, l1_ratio=0.9, max_iter=10000, random_state=42)
        regressor.fit(X_train, y_train)
        
        # Predicting the continuous values 
        y_pred_test = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)
        
        # Evaluating the regressor
        #mse_test = mean_squared_error(y_test, y_pred_test)
        #mse_train = mean_squared_error(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        losses_cp['train_lp_r2'].append(float(r2_train))
        losses_cp['val_lp_r2'].append(float(r2_test))

def get_embeddings(data_path, model, device, 
                   dataloader_template, y_label='class', combine='central', remove_cls=True):

    # Data loader
    dataloader = build_h5_dataloader(data_path, 
                                         batch_size=64, 
                                         num_workers=dataloader_template.num_workers,
                                         img_size=dataloader_template.dataset.img_size,
                                         num_patches=dataloader_template.dataset.num_patches,
                                         patch_size=model.module.patch_embed.patch_size[0], 
                                         num_channels=model.module.in_chans, 
                                         max_mask_ratio=None,
                                         shuffle=False)

    # Map target samples to latent-space
    latent_features = mae_latent(model, dataloader, device, verbose=0, remove_cls=remove_cls)
    latent_features = latent_features.data.cpu().numpy()

    # Collect targets
    with h5py.File(data_path, "r") as f:
        y = f[y_label][:]

    if model.module.attn_pool:
        # There is only one output set of features if there is an attention pooling layer
        combine='flatten'

    scale = True
    if combine=='token':
        x = latent_features[:,:1].reshape(latent_features.shape[0], -1)
    elif combine=='flatten':
        x = latent_features.reshape(latent_features.shape[0], -1)
    elif combine=='pool':
        x = np.max(latent_features, axis=1)
    elif combine=='centralpool':
        x = select_centre(latent_features, n_patches=16)
        x = np.max(x, axis=1)
    elif combine=='central':
        x = select_centre(latent_features, n_patches=4)
        x = x.reshape(x.shape[0], -1)
    elif combine=='mean':
        x = np.mean(latent_features, axis=1)
    else:
        x = latent_features
        x = (x - np.nanmean(x)) / np.nanstd(x)
        scale = False
        
    if scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    return x, y
