import numpy as np
import copy
import csv
import math
import os
import torch
import metrics


class ReliablePositiveRecoveryState:
    """Epoch-delayed core selection using only actual LL-R candidates."""

    CSV_FIELDS = [
        'epoch', 'clean_rate_used', 'top_q', 'lambda_rec',
        'llr_candidate_count', 'previous_core_count', 'recovered_count',
        'recovered_true_fn', 'recovered_oracle_precision',
        'recovered_mean_prob', 'recovered_mean_positive_grad_proxy',
        'temporal_overlap_rate', 'next_core_count',
    ]

    def __init__(self, num_samples, num_classes, top_q):
        if not 0 < top_q <= 1:
            raise ValueError('top_q must satisfy 0 < top_q <= 1')
        self.top_q = top_q
        self.prev_core_mask = torch.zeros(
            (num_samples, num_classes), dtype=torch.bool, device='cpu'
        )
        self.start_epoch()

    def start_epoch(self):
        # The previous core survives; current candidates and diagnostics reset.
        self.candidate_sample_idx = []
        self.candidate_class_idx = []
        self.candidate_losses = []
        self.recovered_true_fn = 0
        self.recovered_prob_sum = 0.0
        self.recovered_grad_proxy_sum = 0.0

    @torch.no_grad()
    def update_recovery_diagnostics(self, recovery_mask, logits,
                                    label_vec_obs, label_vec_true):
        """Read actual recovered entries for logging only; return batch count."""
        mask = recovery_mask.detach().bool()
        recovered_probs = torch.sigmoid(logits.detach())[mask]
        recovered_count = recovered_probs.numel()
        if recovered_count:
            hidden_fn_mask = ((label_vec_obs.detach() == 0)
                              & (label_vec_true.detach() == 1))
            self.recovered_true_fn += int((mask & hidden_fn_mask).sum().item())
            self.recovered_prob_sum += recovered_probs.double().sum().item()
            self.recovered_grad_proxy_sum += (
                1.0 - recovered_probs
            ).double().sum().item()
        return recovered_count

    def get_previous_core(self, batch_idx):
        indices = torch.as_tensor(batch_idx).detach().to('cpu', dtype=torch.long)
        return self.prev_core_mask[indices]

    @torch.no_grad()
    def collect_current_candidates(self, batch_idx, rejection_mask,
                                   raw_loss_matrix):
        positions = rejection_mask.detach().nonzero(as_tuple=True)
        if positions[0].numel() == 0:
            return
        indices = torch.as_tensor(batch_idx).detach().to('cpu', dtype=torch.long)
        self.candidate_sample_idx.append(indices[positions[0].detach().cpu()])
        self.candidate_class_idx.append(positions[1].detach().cpu())
        self.candidate_losses.append(raw_loss_matrix[positions].detach().cpu())

    @torch.no_grad()
    def finalize_epoch(self, recovered_count=0):
        previous_core_count = int(self.prev_core_mask.sum().item())
        current_core_mask = torch.zeros_like(self.prev_core_mask)
        candidate_count = 0
        if self.candidate_losses:
            candidate_losses = torch.cat(self.candidate_losses)
            sample_idx = torch.cat(self.candidate_sample_idx)
            class_idx = torch.cat(self.candidate_class_idx)
            candidate_count = candidate_losses.numel()
            k = max(1, min(math.ceil(candidate_count * self.top_q), candidate_count))
            top_indices = torch.topk(
                candidate_losses, k, largest=True, sorted=False
            ).indices
            current_core_mask[sample_idx[top_indices], class_idx[top_indices]] = True
        self.prev_core_mask = current_core_mask
        summary = {
            'top_q': self.top_q,
            'llr_candidate_count': candidate_count,
            'previous_core_count': previous_core_count,
            'recovered_count': recovered_count,
            'recovered_true_fn': self.recovered_true_fn,
            'recovered_oracle_precision': (
                self.recovered_true_fn / recovered_count
                if recovered_count else float('nan')
            ),
            'recovered_mean_prob': (
                self.recovered_prob_sum / recovered_count
                if recovered_count else float('nan')
            ),
            'recovered_mean_positive_grad_proxy': (
                self.recovered_grad_proxy_sum / recovered_count
                if recovered_count else float('nan')
            ),
            'temporal_overlap_rate': (
                recovered_count / previous_core_count if previous_core_count else 0.0
            ),
            'next_core_count': int(current_core_mask.sum().item()),
        }
        self.start_epoch()
        return summary

    @classmethod
    def append_csv(cls, path, summary):
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        if not write_header:
            with open(path, newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                existing_fields = reader.fieldnames
                if existing_fields != cls.CSV_FIELDS:
                    # Preserve historical rows; missing diagnostics are unknown.
                    legacy_fields = [field for field in cls.CSV_FIELDS if field not in (
                        'recovered_true_fn', 'recovered_oracle_precision',
                        'recovered_mean_prob', 'recovered_mean_positive_grad_proxy',
                    )]
                    if existing_fields != legacy_fields:
                        raise ValueError(f'RPR CSV schema mismatch: {path}')
                    historical_rows = list(reader)
            if existing_fields != cls.CSV_FIELDS:
                with open(path, 'w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=cls.CSV_FIELDS)
                    writer.writeheader()
                    for row in historical_rows:
                        writer.writerow({key: row.get(key, float('nan'))
                                         for key in cls.CSV_FIELDS})
        with open(path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=cls.CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({key: summary[key] for key in cls.CSV_FIELDS})

    @staticmethod
    def print_summary(summary):
        print(f"[RPR][Epoch {summary['epoch']}]")
        print(f"LL-R candidate count: {summary['llr_candidate_count']}")
        print(f"Top-q selected for next epoch: {summary['next_core_count']}")
        print(f"top_q: {summary['top_q']:.2f}")
        print(f"Previous core count: {summary['previous_core_count']}")
        print(f"Current recovered count: {summary['recovered_count']}")
        for label, key in (
            ('Recovered oracle precision', 'recovered_oracle_precision'),
            ('Recovered mean prob', 'recovered_mean_prob'),
            ('Recovered mean positive-grad proxy', 'recovered_mean_positive_grad_proxy'),
        ):
            value = summary[key]
            formatted = 'N/A' if math.isnan(value) else f'{value:.4f}'
            print(f'{label}: {formatted}')
        print(f"Temporal overlap rate: {summary['temporal_overlap_rate']:.4f}")
        print(f"lambda_rec: {summary['lambda_rec']}")


class AdaptiveBoostDiagnosticsAccumulator:
    CSV_FIELDS = [
        'epoch', 'clean_rate_used', 'FN_total', 'TN_total', 'rejected_FN',
        'rejected_TN', 'normal_TN', 'rejected_total', 'FN_rejection_rate',
        'TN_rejection_rate', 'reject_precision', 'mean_raw_loss_FN',
        'mean_raw_loss_TN', 'loss_gap', 'mean_delta_g_FN', 'mean_delta_g_TN',
        'mean_delta_g_ObsPos', 'boost_selectivity_ratio', 'mean_alpha_eff_FN',
        'mean_alpha_eff_TN', 'mean_alpha_eff_ObsPos', 'alpha_eff_gap',
        'mean_positive_cam_mass_FN', 'mean_positive_cam_mass_TN',
        'mean_positive_cam_mass_ObsPos', 'mean_dominant_ratio_FN',
        'mean_dominant_ratio_TN', 'mean_dominant_ratio_ObsPos',
        'mean_delta_g_rejected_FN', 'mean_delta_g_rejected_TN',
        'mean_delta_g_normal_TN', 'mean_alpha_eff_rejected_FN',
        'mean_alpha_eff_rejected_TN', 'mean_alpha_eff_normal_TN',
        'mean_dominant_ratio_rejected_FN',
        'mean_dominant_ratio_rejected_TN',
        'mean_dominant_ratio_normal_TN',
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
            rejected_fn = rejected & hidden_fn
            rejected_tn = rejected & true_negative
            normal_tn = ~rejected & true_negative

            self.counts['FN_total'] += hidden_fn.sum().item()
            self.counts['TN_total'] += true_negative.sum().item()
            self.counts['rejected_FN'] += rejected_fn.sum().item()
            self.counts['rejected_TN'] += rejected_tn.sum().item()
            self.counts['rejected_total'] += rejected.sum().item()

            masks = {
                'FN': hidden_fn,
                'TN': true_negative,
                'ObsPos': observed_positive,
                'rejected_FN': rejected_fn,
                'rejected_TN': rejected_tn,
                'normal_TN': normal_tn,
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
        result['normal_TN'] = result['TN_total'] - result['rejected_TN']
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
            'delta_g_rejected_FN', 'delta_g_rejected_TN',
            'delta_g_normal_TN', 'alpha_eff_rejected_FN',
            'alpha_eff_rejected_TN', 'alpha_eff_normal_TN',
            'dominant_ratio_rejected_FN',
            'dominant_ratio_rejected_TN', 'dominant_ratio_normal_TN',
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
        if not write_header:
            with open(path, 'r', newline='', encoding='utf-8') as handle:
                existing_fields = next(csv.reader(handle), [])
            if existing_fields != cls.CSV_FIELDS:
                raise ValueError(
                    f'Diagnostics CSV schema mismatch: {path}. '
                    'Use a new save_path or move the existing CSV before resuming.'
                )
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
        print('Rejected/normal CAM diagnostics '
              f"(rejected FN={summary['rejected_FN']}, "
              f"rejected TN={summary['rejected_TN']}, "
              f"normal TN={summary['normal_TN']}):")
        print('  Delta_g rejected-FN/rejected-TN/normal-TN: '
              f"{summary['mean_delta_g_rejected_FN']:.4f} / "
              f"{summary['mean_delta_g_rejected_TN']:.4f} / "
              f"{summary['mean_delta_g_normal_TN']:.4f}")
        print('  Alpha_eff rejected-FN/rejected-TN/normal-TN: '
              f"{summary['mean_alpha_eff_rejected_FN']:.4f} / "
              f"{summary['mean_alpha_eff_rejected_TN']:.4f} / "
              f"{summary['mean_alpha_eff_normal_TN']:.4f}")
        print('  Dominant ratio rejected-FN/rejected-TN/normal-TN: '
              f"{summary['mean_dominant_ratio_rejected_FN']:.4f} / "
              f"{summary['mean_dominant_ratio_rejected_TN']:.4f} / "
              f"{summary['mean_dominant_ratio_normal_TN']:.4f}")
        print(f"Raw loss FN/TN: {summary['mean_raw_loss_FN']:.4f} / "
              f"{summary['mean_raw_loss_TN']:.4f}")


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
