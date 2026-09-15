import csv
import heapq
import os

import numpy as np
import torch
import torch.nn.functional as F

from instrumentation import compute_metrics


GROUPS = ('op', 'fn', 'tn')
STAT_FIELDS = (
    'positive_area',
    'positive_mass',
    'top5_mean',
    'logits_raw',
    'logits_boost',
    'delta_logit',
    'prob_raw',
    'prob_boost',
    'delta_prob',
    'concentration',
)


class BoostDiagnostics:
    def __init__(self, save_path, num_classes):
        self.save_path = os.path.join(save_path, 'diagnostics')
        self.num_classes = num_classes
        os.makedirs(self.save_path, exist_ok=True)
        self.epoch_summary_path = os.path.join(self.save_path, 'epoch_summary.csv')
        self.class_summary_path = os.path.join(self.save_path, 'class_summary.csv')
        self.hard_cases_path = os.path.join(self.save_path, 'hard_cases.csv')
        self.reset_epoch()

    def reset_epoch(self):
        self.group_values = {
            group: {field: [] for field in STAT_FIELDS}
            for group in GROUPS
        }
        self.class_values = {
            'fn_delta_logit': [[] for _ in range(self.num_classes)],
            'tn_delta_logit': [[] for _ in range(self.num_classes)],
            'fn_concentration': [[] for _ in range(self.num_classes)],
            'tn_concentration': [[] for _ in range(self.num_classes)],
        }
        self.class_counts = {
            'fn': np.zeros(self.num_classes, dtype=np.int64),
            'tn': np.zeros(self.num_classes, dtype=np.int64),
        }
        self.reject = {
            'rejected_total': 0,
            'rejected_fn': 0,
            'rejected_tn': 0,
            'all_fn': 0,
            'all_tn': 0,
            'rejected_fn_delta_logit': [],
            'rejected_tn_delta_logit': [],
            'non_rejected_tn_delta_logit': [],
        }
        self.hard_heaps = {
            'tn_delta_logit': [[] for _ in range(self.num_classes)],
            'rejected_tn_delta_logit': [[] for _ in range(self.num_classes)],
            'fn_delta_logit': [[] for _ in range(self.num_classes)],
        }
        self.hard_counter = 0
        self.val_raw = []
        self.val_boost = []
        self.val_true = []

    def update_batch(self, phase, batch, diagnostics, correction_idx=None):
        with torch.no_grad():
            label_obs = batch['label_vec_obs'].detach().cpu()
            label_true = batch['label_vec_true'].detach().cpu()
            image_idx = batch['idx'].detach().cpu().numpy()

            cam_raw = diagnostics['cam_raw'].detach().cpu()
            logits_raw = diagnostics['logits_raw'].detach().cpu()
            logits_boost = diagnostics['logits_boost'].detach().cpu()
            if logits_raw.dim() == 1:
                logits_raw = logits_raw.unsqueeze(0)
                logits_boost = logits_boost.unsqueeze(0)

            prob_raw = torch.sigmoid(logits_raw)
            prob_boost = torch.sigmoid(logits_boost)
            delta_logit = logits_boost - logits_raw
            delta_prob = prob_boost - prob_raw

            flat_cam = cam_raw.flatten(start_dim=2)
            relu_cam = F.relu(flat_cam)
            positive_area = (flat_cam > 0).float().mean(dim=2)
            positive_mass = relu_cam.mean(dim=2)
            topk = max(1, int(np.ceil(flat_cam.size(2) * 0.05)))
            top5_mean = flat_cam.topk(topk, dim=2).values.mean(dim=2)
            top5_relu_sum = relu_cam.topk(topk, dim=2).values.sum(dim=2)
            concentration = top5_relu_sum / (relu_cam.sum(dim=2) + 1e-12)

            op_mask = (label_obs == 1) & (label_true == 1)
            fn_mask = (label_obs == 0) & (label_true == 1)
            tn_mask = (label_obs == 0) & (label_true == 0)

            for group, mask in (('op', op_mask), ('fn', fn_mask), ('tn', tn_mask)):
                self._extend_group(group, mask, positive_area, positive_mass, top5_mean,
                                   logits_raw, logits_boost, delta_logit, prob_raw,
                                   prob_boost, delta_prob, concentration)

            if phase == 'train':
                self._update_train_rejects(fn_mask, tn_mask, delta_logit, correction_idx)
                self._update_class_stats(fn_mask, tn_mask, delta_logit, concentration)
                self._update_hard_cases(image_idx, label_obs, label_true, fn_mask, tn_mask,
                                        delta_logit, logits_raw, logits_boost, prob_raw,
                                        prob_boost, positive_area, positive_mass, top5_mean,
                                        concentration, correction_idx)
            elif phase == 'val':
                self.val_raw.append(prob_raw.numpy())
                self.val_boost.append(prob_boost.numpy())
                self.val_true.append(label_true.numpy())

    def finish_epoch(self, epoch):
        val_metrics_raw, val_metrics_boost = self._val_metrics()
        row = {'epoch': epoch}
        for group in GROUPS:
            for field in STAT_FIELDS:
                values = np.asarray(self.group_values[group][field], dtype=np.float64)
                row[f'{group}_{field}_mean'] = self._mean(values)
                row[f'{group}_{field}_median'] = self._median(values)
                row[f'{group}_{field}_q90'] = self._q(values, 90)

        row.update({
            'mAP_raw': val_metrics_raw['map'],
            'mAP_boost': val_metrics_boost['map'],
            'rejected_total': self.reject['rejected_total'],
            'rejected_fn': self.reject['rejected_fn'],
            'rejected_tn': self.reject['rejected_tn'],
            'all_fn': self.reject['all_fn'],
            'all_tn': self.reject['all_tn'],
            'reject_precision': self._ratio(self.reject['rejected_fn'], self.reject['rejected_total']),
            'fn_reject_recall': self._ratio(self.reject['rejected_fn'], self.reject['all_fn']),
            'tn_reject_rate': self._ratio(self.reject['rejected_tn'], self.reject['all_tn']),
            'rejected_fn_delta_logit': self._mean(self.reject['rejected_fn_delta_logit']),
            'rejected_tn_delta_logit': self._mean(self.reject['rejected_tn_delta_logit']),
            'non_rejected_tn_delta_logit': self._mean(self.reject['non_rejected_tn_delta_logit']),
        })
        self._append_csv(self.epoch_summary_path, row)
        self._write_class_rows(epoch, val_metrics_raw['ap'], val_metrics_boost['ap'])
        self._write_hard_cases(epoch)
        self._print_epoch(epoch, row)
        self.reset_epoch()

    def _extend_group(self, group, mask, *tensors):
        fields = STAT_FIELDS
        for field, tensor in zip(fields, tensors):
            values = tensor[mask].numpy()
            self.group_values[group][field].extend(values.tolist())

    def _update_train_rejects(self, fn_mask, tn_mask, delta_logit, correction_idx):
        self.reject['all_fn'] += int(fn_mask.sum().item())
        self.reject['all_tn'] += int(tn_mask.sum().item())
        rejected = torch.zeros_like(fn_mask, dtype=torch.bool)
        if correction_idx is not None and len(correction_idx) == 2 and correction_idx[0].numel():
            rejected[correction_idx[0].cpu().long(), correction_idx[1].cpu().long()] = True
        rejected_fn = rejected & fn_mask
        rejected_tn = rejected & tn_mask
        non_rejected_tn = (~rejected) & tn_mask
        self.reject['rejected_total'] += int(rejected.sum().item())
        self.reject['rejected_fn'] += int(rejected_fn.sum().item())
        self.reject['rejected_tn'] += int(rejected_tn.sum().item())
        self.reject['rejected_fn_delta_logit'].extend(delta_logit[rejected_fn].numpy().tolist())
        self.reject['rejected_tn_delta_logit'].extend(delta_logit[rejected_tn].numpy().tolist())
        self.reject['non_rejected_tn_delta_logit'].extend(delta_logit[non_rejected_tn].numpy().tolist())

    def _update_class_stats(self, fn_mask, tn_mask, delta_logit, concentration):
        for class_id in range(self.num_classes):
            fn_values = fn_mask[:, class_id]
            tn_values = tn_mask[:, class_id]
            self.class_counts['fn'][class_id] += int(fn_values.sum().item())
            self.class_counts['tn'][class_id] += int(tn_values.sum().item())
            self.class_values['fn_delta_logit'][class_id].extend(delta_logit[:, class_id][fn_values].numpy().tolist())
            self.class_values['tn_delta_logit'][class_id].extend(delta_logit[:, class_id][tn_values].numpy().tolist())
            self.class_values['fn_concentration'][class_id].extend(concentration[:, class_id][fn_values].numpy().tolist())
            self.class_values['tn_concentration'][class_id].extend(concentration[:, class_id][tn_values].numpy().tolist())

    def _update_hard_cases(self, image_idx, label_obs, label_true, fn_mask, tn_mask, delta_logit,
                           logits_raw, logits_boost, prob_raw, prob_boost, positive_area,
                           positive_mass, top5_mean, concentration, correction_idx):
        rejected = torch.zeros_like(fn_mask, dtype=torch.bool)
        if correction_idx is not None and len(correction_idx) == 2 and correction_idx[0].numel():
            rejected[correction_idx[0].cpu().long(), correction_idx[1].cpu().long()] = True
        for case_type, mask in (
            ('tn_delta_logit', tn_mask),
            ('rejected_tn_delta_logit', tn_mask & rejected),
            ('fn_delta_logit', fn_mask),
        ):
            coords = torch.where(mask)
            for batch_i, class_id in zip(coords[0].tolist(), coords[1].tolist()):
                score = float(delta_logit[batch_i, class_id])
                row = {
                    'case_type': case_type,
                    'image_idx': int(image_idx[batch_i]),
                    'class_id': class_id,
                    'observed_label': float(label_obs[batch_i, class_id]),
                    'true_label': float(label_true[batch_i, class_id]),
                    'raw_logit': float(logits_raw[batch_i, class_id]),
                    'boost_logit': float(logits_boost[batch_i, class_id]),
                    'delta_logit': score,
                    'raw_prob': float(prob_raw[batch_i, class_id]),
                    'boost_prob': float(prob_boost[batch_i, class_id]),
                    'positive_area': float(positive_area[batch_i, class_id]),
                    'positive_mass': float(positive_mass[batch_i, class_id]),
                    'top5_mean': float(top5_mean[batch_i, class_id]),
                    'concentration': float(concentration[batch_i, class_id]),
                    'was_rejected': int(rejected[batch_i, class_id]),
                }
                heap = self.hard_heaps[case_type][class_id]
                self.hard_counter += 1
                item = (score, self.hard_counter, row)
                if len(heap) < 50:
                    heapq.heappush(heap, item)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, item)

    def _val_metrics(self):
        y_true = np.concatenate(self.val_true, axis=0) if self.val_true else np.zeros((0, self.num_classes))
        y_raw = np.concatenate(self.val_raw, axis=0) if self.val_raw else np.zeros((0, self.num_classes))
        y_boost = np.concatenate(self.val_boost, axis=0) if self.val_boost else np.zeros((0, self.num_classes))
        return compute_metrics(y_raw, y_true), compute_metrics(y_boost, y_true)

    def _write_class_rows(self, epoch, ap_raw, ap_boost):
        rows = []
        for class_id in range(self.num_classes):
            rows.append({
                'epoch': epoch,
                'class_id': class_id,
                'fn_count': int(self.class_counts['fn'][class_id]),
                'tn_count': int(self.class_counts['tn'][class_id]),
                'fn_delta_logit': self._mean(self.class_values['fn_delta_logit'][class_id]),
                'tn_delta_logit': self._mean(self.class_values['tn_delta_logit'][class_id]),
                'fn_concentration': self._mean(self.class_values['fn_concentration'][class_id]),
                'tn_concentration': self._mean(self.class_values['tn_concentration'][class_id]),
                'ap_raw': float(ap_raw[class_id]),
                'ap_boost': float(ap_boost[class_id]),
                'delta_ap': float(ap_boost[class_id] - ap_raw[class_id]),
            })
        self._append_csv(self.class_summary_path, rows)

    def _write_hard_cases(self, epoch):
        rows = []
        for heaps in self.hard_heaps.values():
            for heap in heaps:
                for _, _, row in heap:
                    row = dict(row)
                    row['epoch'] = epoch
                    rows.append(row)
        rows.sort(key=lambda row: (row['epoch'], row['case_type'], row['class_id'], -row['delta_logit']))
        self._append_csv(self.hard_cases_path, rows)

    def _print_epoch(self, epoch, row):
        print(
            f"[BoostDiag][Epoch {epoch}] "
            f"mAP raw={row['mAP_raw']:.2f} boost={row['mAP_boost']:.2f} | "
            f"delta_logit FN={row['fn_delta_logit_mean']:.3f} TN={row['tn_delta_logit_mean']:.3f} | "
            f"mass FN={row['fn_positive_mass_mean']:.3f} TN={row['tn_positive_mass_mean']:.3f} | "
            f"reject FN={row['rejected_fn']} TN={row['rejected_tn']} "
            f"precision={100.0 * row['reject_precision']:.2f}%"
        )

    def _append_csv(self, path, rows):
        if isinstance(rows, dict):
            rows = [rows]
        if not rows:
            return
        write_header = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _mean(values):
        values = np.asarray(values, dtype=np.float64)
        return float(np.mean(values)) if values.size else float('nan')

    @staticmethod
    def _median(values):
        values = np.asarray(values, dtype=np.float64)
        return float(np.median(values)) if values.size else float('nan')

    @staticmethod
    def _q(values, q):
        values = np.asarray(values, dtype=np.float64)
        return float(np.percentile(values, q)) if values.size else float('nan')

    @staticmethod
    def _ratio(num, den):
        return float(num) / float(den) if den else float('nan')
