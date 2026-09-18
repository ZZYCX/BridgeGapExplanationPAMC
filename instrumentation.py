import numpy as np
import copy
import csv
import math
import os
import torch
import metrics


class AdaptiveBoostDiagnosticsAccumulator:
    CSV_FIELDS = [
        'epoch', 'clean_rate_used', 'FN_total', 'TN_total', 'rejected_FN',
        'rejected_TN', 'rejected_total', 'FN_rejection_rate',
        'TN_rejection_rate', 'reject_precision', 'mean_raw_loss_FN',
        'mean_raw_loss_TN', 'loss_gap', 'mean_delta_g_FN', 'mean_delta_g_TN',
        'mean_delta_g_ObsPos', 'boost_selectivity_ratio', 'mean_alpha_eff_FN',
        'mean_alpha_eff_TN', 'mean_alpha_eff_ObsPos', 'alpha_eff_gap',
        'mean_positive_cam_mass_FN', 'mean_positive_cam_mass_TN',
        'mean_positive_cam_mass_ObsPos', 'mean_dominant_ratio_FN',
        'mean_dominant_ratio_TN', 'mean_dominant_ratio_ObsPos',
    ]

    def __init__(self):
        self.counts = {
            'FN_total': 0, 'TN_total': 0, 'rejected_FN': 0,
            'rejected_TN': 0, 'rejected_total': 0,
        }
        self.stats = {}

    def _accumulate(self, name, values, mask):
        selected = values.detach()[mask]
        if selected.numel() == 0:
            return
        total, count = self.stats.get(name, (0.0, 0))
        self.stats[name] = (total + selected.double().sum().item(),
                            count + selected.numel())

    def update(self, label_vec_obs, label_vec_true, boost_diag, loss_diag):
        with torch.no_grad():
            observed = label_vec_obs.detach()
            clean = label_vec_true.detach()
            hidden_fn = (observed == 0) & (clean == 1)
            true_negative = (observed == 0) & (clean != 1)
            observed_positive = (observed == 1) & (clean == 1)
            rejected = loss_diag['rejection_mask'].detach().bool()

            self.counts['FN_total'] += hidden_fn.sum().item()
            self.counts['TN_total'] += true_negative.sum().item()
            self.counts['rejected_FN'] += (rejected & hidden_fn).sum().item()
            self.counts['rejected_TN'] += (rejected & true_negative).sum().item()
            self.counts['rejected_total'] += rejected.sum().item()

            masks = {
                'FN': hidden_fn,
                'TN': true_negative,
                'ObsPos': observed_positive,
            }
            for suffix, mask in masks.items():
                self._accumulate('delta_g_' + suffix, boost_diag['delta_g'], mask)
                self._accumulate(
                    'alpha_eff_' + suffix,
                    boost_diag['alpha_eff'],
                    mask & boost_diag['alpha_eff_valid'],
                )
                self._accumulate(
                    'positive_cam_mass_' + suffix,
                    boost_diag['positive_cam_mass'],
                    mask,
                )
                self._accumulate(
                    'dominant_ratio_' + suffix,
                    boost_diag['dominant_ratio'],
                    mask & boost_diag['dominant_ratio_valid'],
                )

            self._accumulate('raw_loss_FN', loss_diag['raw_loss_matrix'], hidden_fn)
            self._accumulate('raw_loss_TN', loss_diag['raw_loss_matrix'], true_negative)

    def _mean(self, name):
        total, count = self.stats.get(name, (0.0, 0))
        return total / count if count else float('nan')

    @staticmethod
    def _ratio(numerator, denominator, empty_value=0.0):
        return numerator / denominator if denominator else empty_value

    def summarize(self, epoch, clean_rate_used):
        result = dict(self.counts)
        result['epoch'] = epoch
        result['clean_rate_used'] = clean_rate_used
        result['FN_rejection_rate'] = self._ratio(
            result['rejected_FN'], result['FN_total'])
        result['TN_rejection_rate'] = self._ratio(
            result['rejected_TN'], result['TN_total'])
        rejected_known = result['rejected_FN'] + result['rejected_TN']
        result['reject_precision'] = self._ratio(
            result['rejected_FN'], rejected_known, float('nan'))

        for stat_name in [
            'raw_loss_FN', 'raw_loss_TN', 'delta_g_FN', 'delta_g_TN',
            'delta_g_ObsPos', 'alpha_eff_FN', 'alpha_eff_TN',
            'alpha_eff_ObsPos', 'positive_cam_mass_FN',
            'positive_cam_mass_TN', 'positive_cam_mass_ObsPos',
            'dominant_ratio_FN', 'dominant_ratio_TN',
            'dominant_ratio_ObsPos',
        ]:
            result['mean_' + stat_name] = self._mean(stat_name)

        result['loss_gap'] = (
            result['mean_raw_loss_FN'] - result['mean_raw_loss_TN'])
        result['boost_selectivity_ratio'] = (
            result['mean_delta_g_FN'] / (result['mean_delta_g_TN'] + 1e-6))
        result['alpha_eff_gap'] = (
            result['mean_alpha_eff_FN'] - result['mean_alpha_eff_TN'])
        return result

    @classmethod
    def append_csv(cls, path, summary):
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=cls.CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({key: summary[key] for key in cls.CSV_FIELDS})

    @staticmethod
    def print_summary(summary):
        precision = summary['reject_precision']
        precision_text = 'N/A' if math.isnan(precision) else f'{precision:.4f}'
        print(f"[AdaptiveBoost][Epoch {summary['epoch']}]")
        print(f"FN reject:  {summary['rejected_FN']} / {summary['FN_total']} = "
              f"{summary['FN_rejection_rate']:.4f}")
        print(f"TN reject:  {summary['rejected_TN']} / {summary['TN_total']} = "
              f"{summary['TN_rejection_rate']:.4f}")
        print(f'Reject precision: {precision_text}')
        print(f"Boost delta FN/TN: {summary['mean_delta_g_FN']:.4f} / "
              f"{summary['mean_delta_g_TN']:.4f}")
        print(f"Boost selectivity: {summary['boost_selectivity_ratio']:.4f}")
        print(f"Alpha_eff FN/TN: {summary['mean_alpha_eff_FN']:.4f} / "
              f"{summary['mean_alpha_eff_TN']:.4f}")
        print(f"Raw loss FN/TN: {summary['mean_raw_loss_FN']:.4f} / "
              f"{summary['mean_raw_loss_TN']:.4f}")


class train_logger:
    
    '''
    An instance of this class keeps track of various metrics throughout
    the training process.
    '''
    
    def __init__(self, params):
        
        self.params = params
        
        # epoch-level objects:
        self.best_stop_metric = -np.Inf
        self.best_epoch = -1
        self.running_loss = 0.0
        self.num_examples = 0
        
        # batch-level objects:
        self.temp_preds = []
        self.temp_true = [] # true labels
        self.temp_obs = [] # observed labels
        self.temp_indices = [] # indices for each example
        self.temp_batch_loss = []
        self.temp_batch_reg = []
        
        # output objects: 
        self.logs = {}
        self.logs['metrics'] = {}
        self.logs['best_preds'] = {}
        self.logs['gt'] ={}
        self.logs['obs'] = {}
        self.logs['targ'] = {}
        self.logs['idx'] = {}
        for field in self.logs:
            for phase in ['train', 'val', 'test']:
                self.logs[field][phase] = {}
    
    def compute_phase_metrics(self, phase, epoch, labels_est):
        
        '''
        Compute and store end-of-phase metrics. 
        '''
        
        self.logs['metrics'][phase][epoch] = {} 
        
        # compute metrics w.r.t. clean ground truth labels:
        metrics_clean = compute_metrics(self.temp_preds, self.temp_true)
        for k in metrics_clean:
            self.logs['metrics'][phase][epoch][k + '_clean'] = metrics_clean[k]
        
        # compute metrics w.r.t. observed labels:
        metrics_observed = compute_metrics(self.temp_preds, self.temp_obs)
        for k in metrics_observed:
            self.logs['metrics'][phase][epoch][k + '_observed'] = metrics_observed[k]
        
        if phase == 'train':
            self.logs['metrics'][phase][epoch]['loss'] = self.running_loss / self.num_examples
            self.logs['metrics'][phase][epoch]['est_labels_k_hat'] = float(np.mean(np.sum(labels_est, axis=1)))
            self.logs['metrics'][phase][epoch]['avg_batch_reg'] = np.mean(self.temp_batch_reg)
        else:
            self.logs['metrics'][phase][epoch]['loss'] = -999
            self.logs['metrics'][phase][epoch]['est_labels_k_hat'] = -999
            self.logs['metrics'][phase][epoch]['avg_batch_reg'] = -999
        self.logs['metrics'][phase][epoch]['preds_k_hat'] = np.mean(np.sum(self.temp_preds, axis=1))
   
    def get_stop_metric(self, phase, epoch, variant):
        
        '''
        Query the stop metric.
        '''
        
        assert variant in ['clean', 'observed']
        return self.logs['metrics'][phase][epoch][self.params['stop_metric'] + '_' + variant]

    def update_phase_data(self, batch):
        
        '''
        Store data from a batch for later use in computing metrics. 
        '''
        
        for i in range(len(batch['idx'])):
            self.temp_preds.append(batch['preds_np'][i, :].tolist())
            self.temp_true.append(batch['label_vec_true'][i, :].tolist())
            self.temp_obs.append(batch['label_vec_obs'][i, :].tolist())
            self.temp_indices.append(int(batch['idx'][i]))
            self.num_examples += 1
        self.temp_batch_loss.append(float(batch['loss_np']))
        self.temp_batch_reg.append(float(batch['reg_loss_np']))
        self.running_loss += float(batch['loss_np'] * batch['image'].size(0))
        
    def reset_phase_data(self):
        
        '''
        Reset for a new phase. 
        '''
        
        self.temp_preds = []
        self.temp_true = []
        self.temp_obs = []
        self.temp_indices = []
        self.temp_batch_reg = []
        self.running_loss = 0.0
        self.num_examples = 0.0
        
    def update_best_results(self, phase, epoch, variant):
        
        '''
        Update the current best epoch info if applicable.
        '''
        
        if phase == 'train':
            return False
        elif phase == 'val':
            assert variant in ['clean', 'observed']
            cur_stop_metric = self.get_stop_metric(phase, epoch, variant)
            if cur_stop_metric > self.best_stop_metric:
                self.best_stop_metric = cur_stop_metric
                self.best_epoch = epoch
                self.logs['best_preds'][phase] = self.temp_preds
                self.logs['gt'][phase] = self.temp_true
                self.logs['obs'][phase] = self.temp_obs
                self.logs['idx'][phase] = self.temp_indices
                return True # new best found
            else:
                return False # new best not found
        elif phase == 'test':
            if epoch == self.best_epoch:
                self.logs['best_preds'][phase] = self.temp_preds
                self.logs['gt'][phase] = self.temp_true
                self.logs['obs'][phase] = self.temp_obs
                self.logs['idx'][phase] = self.temp_indices
            return False
        
    def get_logs(self):
        
        '''
        Return a copy of all log data.
        '''
        
        return copy.deepcopy(self.logs)
    
    def report(self, t_i, t_f, phase, epoch):
        report = '[{}] time: {:.2f} min, loss: {:.3f}, {}: {:.2f}, {}: {:.2f}'.format(
            phase,
            (t_f - t_i) / 60.0,
            self.logs['metrics'][phase][epoch]['loss'],
            self.params['stop_metric'] + '_clean',
            self.get_stop_metric(phase, epoch, 'clean'),
            self.params['stop_metric'] + '_observed',
            self.get_stop_metric(phase, epoch, 'observed'),
            )
        print(report)

        
def compute_metrics(y_pred, y_true):
    
    '''
    Given predictions and labels, compute a few metrics.
    '''
    
    num_examples, num_classes = np.shape(y_true)
    
    results = {}
    average_precision_list = []
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_true = np.array(y_true == 1, dtype=np.float32) # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        average_precision_list.append(metrics.compute_avg_precision(y_true[:, j], y_pred[:, j]))
        
    results['map'] = 100.0 * float(np.mean(average_precision_list))
    results['ap'] = 100.0 * np.array(average_precision_list)
    ''''
    for k in [1, 3, 5]:
        rec_at_k = np.array([metrics.compute_recall_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
        prec_at_k = np.array([metrics.compute_precision_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
        results['rec_at_{}'.format(k)] = np.mean(rec_at_k)
        results['prec_at_{}'.format(k)] = np.mean(prec_at_k)
        results['top_{}'.format(k)] = np.mean(prec_at_k > 0)
    '''
    return results

def compute_metrics_openimages(y_pred, y_true):
    
    '''
    Given predictions and labels, compute a few metrics.
    '''
    
    num_examples, num_classes = np.shape(y_true)
    
    results = {}
    average_precision_list = []
    # y_pred = np.array(y_pred)
    # y_true = np.array(y_true)
    # y_true = np.array(y_true == 1, dtype=np.float32) # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        observed_idxs = np.where(y_true[:, j] != -1)[0]
        if len(observed_idxs) == 0:
            continue
        average_precision_list.append(metrics.compute_avg_precision(y_true[observed_idxs, j], y_pred[observed_idxs, j]))
        
    results['map'] = 100.0 * float(np.mean(average_precision_list))
    ''''
    for k in [1, 3, 5]:
        rec_at_k = np.array([metrics.compute_recall_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
        prec_at_k = np.array([metrics.compute_precision_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
        results['rec_at_{}'.format(k)] = np.mean(rec_at_k)
        results['prec_at_{}'.format(k)] = np.mean(prec_at_k)
        results['top_{}'.format(k)] = np.mean(prec_at_k > 0)
    '''
    return results

def compute_aps_openimages(y_pred, y_true):
    
    '''
    Given predictions and labels, compute a few metrics.
    '''
    
    num_examples, num_classes = np.shape(y_true)
    
    results = {}
    average_precision_list = []
    # y_pred = np.array(y_pred)
    # y_true = np.array(y_true)
    # y_true = np.array(y_true == 1, dtype=np.float32) # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        observed_idxs = np.where(y_true[:, j] != -1)[0]
        if len(observed_idxs) == 0:
            continue
        average_precision_list.append(metrics.compute_avg_precision(y_true[observed_idxs, j], y_pred[observed_idxs, j]))

    return np.array(average_precision_list)
