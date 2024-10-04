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

from torchcp.regression.loss import QuantileLoss

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
        elif 'quantile' in model.module.sup_loss_fn.lower():
        
            # Loop over each label to compute the loss
            loss = 0
            quantile_loss = QuantileLoss(model.module.quantiles)
            num_labels = model.module.num_classes // 2
            for i in range(num_labels):
                # Compute quantile loss for the i-th label
                loss = loss + quantile_loss(pred_labels[:, i], tgt_labels[:, i].unsqueeze(1))
        
            # Average loss over all labels
            loss /= num_labels

            # Calculate central predictions
            pred_labels = (pred_labels[..., 0] + pred_labels[..., 1]) / 2
        
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
