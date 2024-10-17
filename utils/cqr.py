# From: https://github.com/ml-stat-Sustech/TorchCP/blob/master/torchcp/regression/predictors/

import math
import torch

from torchcp.regression.utils.metrics import Metrics
from torchcp.utils.common import get_device
from torchcp.utils.common import calculate_conformal_value

class CQRMIM(object):
    """
    Conformalized Quantile Regression (Romano et al., 2019)
    paper: https://arxiv.org/abs/1905.03222

    :param model: a pytorch model that can output alpha/2 and 1-alpha/2 quantile regression.
    """

    def __init__(self, model):
        self._model = model
        self._device = get_device(model)
        self._metric = Metrics()
        
    def calculate_score(self, predicts, y_truth):
        if len(predicts.shape) ==2:
            predicts = predicts.unsqueeze(1)
        if len(y_truth.shape) ==1:
            y_truth = y_truth.unsqueeze(1)
        return torch.maximum(predicts[..., 0] - y_truth, y_truth - predicts[..., 1])

    def calibrate(self, cal_dataloader, alpha):
        self._model.eval()
        predicts_list = []
        y_truth_list = []
        with torch.no_grad():

            for i, (samples, masks, ra_decs, labels) in enumerate(cal_dataloader):
        
                # Switch to GPU if available
                samples = samples.to(self._device, non_blocking=True)
                masks = masks.to(self._device, non_blocking=True)
                ra_decs = ra_decs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)
                
                # Run predictions
                model_outputs = self._model(samples, mask=masks, ra_dec=ra_decs, run_mim=False, run_pred=True).detach()
                # Rescale back to original scale
                model_outputs = self._model.module.denormalize_labels(model_outputs)
                
                predicts_list.append(model_outputs)
                y_truth_list.append(labels)
            
            predicts = torch.cat(predicts_list).float().to(self._device)
            y_truth = torch.cat(y_truth_list).to(self._device)
        self.calculate_threshold(predicts, y_truth, alpha)

    def calculate_threshold(self, predicts, y_truth, alpha):
        scores = self.calculate_score(predicts, y_truth)
        self.q_hat = self._calculate_conformal_value(scores, alpha)
        
    def _calculate_conformal_value(self, scores, alpha):
        return calculate_conformal_value(scores, alpha)

    def predict(self, samples, masks, ra_decs):
        self._model.eval()
        # Switch to GPU if available
        samples = samples.to(self._device, non_blocking=True)
        masks = masks.to(self._device, non_blocking=True)
        ra_decs = ra_decs.to(self._device, non_blocking=True)

        predicts_batch = self._model(samples, mask=masks, ra_dec=ra_decs, 
                                     run_mim=False, run_pred=True).float()
        # Rescale back to original scale
        predicts_batch = self._model.module.denormalize_labels(predicts_batch)
        
        if len(predicts_batch.shape)==2:
            predicts_batch = predicts_batch.unsqueeze(1)
        prediction_intervals = samples.new_zeros((predicts_batch.shape[0],self.q_hat.shape[0] , 2))
        prediction_intervals[..., 0] = predicts_batch[..., 0] - self.q_hat.view(1, self.q_hat.shape[0], 1)
        prediction_intervals[..., 1] = predicts_batch[..., 1] + self.q_hat.view(1, self.q_hat.shape[0], 1)
        return prediction_intervals

    def predict_all(self, data_loader, show_metrics=True):
        y_list = []
        predict_list = []
        with torch.no_grad():

            for i, (samples, masks, ra_decs, labels) in enumerate(data_loader):
                tmp_prediction_intervals = self.predict(samples, masks, ra_decs)
                y_list.append(labels)
                predict_list.append(tmp_prediction_intervals)

        predicts = torch.cat(predict_list,dim=0).to(self._device)
        test_y = torch.cat(y_list).to(self._device)

        if show_metrics:
            res_dict = {"Coverage_rate": self._metric('coverage_rate')(predicts, test_y),
                        "Average_size": self._metric('average_size')(predicts),
                        "q-hat": self.q_hat.float()}
            print("Calibration Results:", res_dict)

        return predicts, test_y

    def evaluate(self, data_loader):
        y_list = []
        predict_list = []
        with torch.no_grad():

            for i, (samples, masks, ra_decs, labels) in enumerate(data_loader):
                tmp_prediction_intervals = self.predict(samples, masks, ra_decs)
                y_list.append(labels)
                predict_list.append(tmp_prediction_intervals)

        predicts = torch.cat(predict_list,dim=0).to(self._device)
        test_y = torch.cat(y_list).to(self._device)

        res_dict = {"Coverage_rate": self._metric('coverage_rate')(predicts, test_y),
                    "Average_size": self._metric('average_size')(predicts)}
        return res_dict