import numpy as np
import copy
import csv
import math
import os
import torch
import metrics


class LLRCandidatePoolOracleAccumulator:
    TOP_RATIOS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    CSV_FIELDS = [
        'epoch', 'top_ratio', 'clean_rate_used', 'llr_active_ratio',
        'unknown_total', 'FN_total', 'TN_total',
        'pool_candidate_count', 'pool_FN', 'pool_TN', 'pool_precision',
        'pool_recall', 'pool_candidates_per_image',
        'top_candidate_count', 'top_FN', 'top_TN', 'top_precision',
        'top_global_FN_recall', 'top_pool_FN_coverage',
        'top_effective_unknown_ratio', 'precision_gain_vs_pool',
    ]

    def __init__(self):
        self.num_images = 0
        self.unknown_total = 0
        self.fn_total = 0
        self.tn_total = 0
        self.candidate_losses = []
        self.candidate_is_fn = []

    @staticmethod
    def _safe_ratio(numerator, denominator, empty_value=0.0):
        return numerator / denominator if denominator else empty_value

    def update(self, label_vec_obs, label_vec_true, raw_loss_matrix,
               rejection_mask):
        """Collect detached entries from the actual LL-R candidate pool."""
        with torch.no_grad():
            observed = label_vec_obs.detach()
            full_labels = label_vec_true.detach()
            losses = raw_loss_matrix.detach()
            rejected = rejection_mask.detach().bool()

            unknown_mask = observed == 0
            hidden_fn_mask = unknown_mask & (full_labels == 1)
            true_negative_mask = unknown_mask & (full_labels != 1)
            candidate_mask = rejected & unknown_mask

            self.num_images += int(observed.size(0))
            self.unknown_total += int(unknown_mask.sum().item())
            self.fn_total += int(hidden_fn_mask.sum().item())
            self.tn_total += int(true_negative_mask.sum().item())

            if candidate_mask.any():
                self.candidate_losses.append(
                    losses[candidate_mask].detach().cpu()
                )
                self.candidate_is_fn.append(
                    hidden_fn_mask[candidate_mask].detach().cpu()
                )

    @torch.no_grad()
    def summarize(self, epoch, clean_rate_used):
        if self.candidate_losses:
            epoch_candidate_losses = torch.cat(self.candidate_losses)
            epoch_candidate_is_fn = torch.cat(self.candidate_is_fn).bool()
            ranking = torch.argsort(epoch_candidate_losses, descending=True)
        else:
            epoch_candidate_losses = torch.empty(0)
            epoch_candidate_is_fn = torch.empty(0, dtype=torch.bool)
            ranking = torch.empty(0, dtype=torch.long)

        pool_candidate_count = int(epoch_candidate_losses.numel())
        pool_fn = int(epoch_candidate_is_fn.sum().item())
        pool_tn = pool_candidate_count - pool_fn
        pool_precision = self._safe_ratio(
            pool_fn, pool_candidate_count, float('nan')
        )
        pool_recall = self._safe_ratio(pool_fn, self.fn_total)
        pool_candidates_per_image = self._safe_ratio(
            pool_candidate_count, self.num_images
        )

        rows = []
        for top_ratio in self.TOP_RATIOS:
            if pool_candidate_count:
                top_candidate_count = max(
                    1, math.ceil(pool_candidate_count * top_ratio)
                )
                top_indices = ranking[:top_candidate_count]
                top_fn = int(epoch_candidate_is_fn[top_indices].sum().item())
                top_tn = top_candidate_count - top_fn
                top_precision = top_fn / top_candidate_count
                top_global_fn_recall = self._safe_ratio(
                    top_fn, self.fn_total
                )
                top_pool_fn_coverage = self._safe_ratio(
                    top_fn, pool_fn, float('nan')
                )
                top_effective_unknown_ratio = self._safe_ratio(
                    top_candidate_count, self.unknown_total
                )
                precision_gain_vs_pool = self._safe_ratio(
                    top_precision, pool_precision, float('nan')
                )
            else:
                top_candidate_count = 0
                top_fn = 0
                top_tn = 0
                top_precision = float('nan')
                top_global_fn_recall = float('nan')
                top_pool_fn_coverage = float('nan')
                top_effective_unknown_ratio = float('nan')
                precision_gain_vs_pool = float('nan')

            rows.append({
                'epoch': epoch,
                'top_ratio': top_ratio,
                'clean_rate_used': clean_rate_used,
                'llr_active_ratio': 1.0 - clean_rate_used,
                'unknown_total': self.unknown_total,
                'FN_total': self.fn_total,
                'TN_total': self.tn_total,
                'pool_candidate_count': pool_candidate_count,
                'pool_FN': pool_fn,
                'pool_TN': pool_tn,
                'pool_precision': pool_precision,
                'pool_recall': pool_recall,
                'pool_candidates_per_image': pool_candidates_per_image,
                'top_candidate_count': top_candidate_count,
                'top_FN': top_fn,
                'top_TN': top_tn,
                'top_precision': top_precision,
                'top_global_FN_recall': top_global_fn_recall,
                'top_pool_FN_coverage': top_pool_fn_coverage,
                'top_effective_unknown_ratio': top_effective_unknown_ratio,
                'precision_gain_vs_pool': precision_gain_vs_pool,
            })
        return rows

    @classmethod
    def append_csv(cls, path, rows):
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        if not write_header:
            with open(path, 'r', newline='', encoding='utf-8') as handle:
                existing_fields = next(csv.reader(handle), [])
            if existing_fields != cls.CSV_FIELDS:
                raise ValueError(
                    f'LL-R candidate oracle CSV schema mismatch: {path}. '
                    'Use a new save_path or move the existing CSV before resuming.'
                )
        with open(path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=cls.CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(
                {key: row[key] for key in cls.CSV_FIELDS} for row in rows
            )

    @staticmethod
    def print_summary(rows):
        if not rows:
            return
        pool = rows[0]

        def format_metric(value):
            return 'N/A' if math.isnan(value) else f'{value:.4f}'

        print(f"[D0 LL-R Candidate Oracle][Epoch {pool['epoch']}]")
        print('LL-R pool:')
        print(f"candidate = {pool['pool_candidate_count']}")
        print(f"FN = {pool['pool_FN']}")
        print(f"TN = {pool['pool_TN']}")
        print(f"precision = {format_metric(pool['pool_precision'])}")
        print(f"recall = {format_metric(pool['pool_recall'])}")
        print('Top-X within LL-R pool:')
        print('ratio   precision   global_FN_recall   pool_FN_coverage')
        for row in rows:
            print(f"{row['top_ratio']:.0%}      "
                  f"{format_metric(row['top_precision'])}      "
                  f"{format_metric(row['top_global_FN_recall'])}"
                  f"             "
                  f"{format_metric(row['top_pool_FN_coverage'])}")


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
