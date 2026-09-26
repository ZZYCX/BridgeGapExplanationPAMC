"""Detached, GT-only reporting for temporary LL-R positive correction."""

import csv
import math
import os

import torch
import torch.nn.functional as F


def _empty():
    return dict(count=0, FN=0, logit=0.0, prob=0.0, positive_bce=0.0,
                grad_mag=0.0)


def _add(group, logits, truth):
    n = logits.numel()
    if not n:
        return
    prob = torch.sigmoid(logits)
    group['count'] += n
    group['FN'] += int((truth == 1).sum().item())
    group['logit'] += logits.double().sum().item()
    group['prob'] += prob.double().sum().item()
    group['positive_bce'] += F.softplus(-logits).double().sum().item()
    group['grad_mag'] += (1 - prob).double().sum().item()


def _mean(group, key):
    return group[key] / group['count'] if group['count'] else float('nan')


def _precision(group):
    return group['FN'] / group['count'] if group['count'] else float('nan')


def _write(path, fields, rows):
    header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if header:
            writer.writeheader()
        writer.writerows(rows)


class PseudoPositiveDiagnostics:
    SUMMARY_FIELDS = (
        'epoch', 'active', 'pseudo_positive_ratio', 'start_epoch',
        'candidate_count', 'candidate_FN', 'candidate_TN', 'candidate_precision',
        'pseudo_positive_count', 'pseudo_positive_FN', 'pseudo_positive_TN',
        'pseudo_positive_precision', 'reject_tail_count', 'reject_tail_FN',
        'reject_tail_TN', 'reject_tail_FN_precision',
        'pseudo_positive_logit_mean', 'pseudo_positive_prob_mean',
        'pseudo_positive_positive_bce_mean', 'pseudo_positive_grad_mag_mean',
        'pseudo_FN_count', 'pseudo_TN_count', 'pseudo_FN_prob_mean',
        'pseudo_TN_prob_mean', 'pseudo_FN_positive_bce_mean',
        'pseudo_TN_positive_bce_mean', 'pseudo_FN_grad_mag_mean',
        'pseudo_TN_grad_mag_mean',
    )
    BIN_FIELDS = ('epoch', 'bin', 'count', 'FN', 'TN', 'precision',
                  'prob_mean', 'positive_bce_mean', 'grad_mag_mean')
    CLASS_FIELDS = ('epoch', 'class_id', 'pseudo_count', 'pseudo_FN',
                    'pseudo_TN', 'pseudo_precision', 'pseudo_prob_mean',
                    'pseudo_positive_bce_mean', 'pseudo_grad_mag_mean')
    BINS = (('0-5%', 0.0, 0.05), ('5-10%', 0.05, 0.10),
            ('10-20%', 0.10, 0.20))

    def __init__(self, num_classes):
        self.groups = {name: _empty() for name in
                       ('candidate', 'pseudo', 'reject', 'pseudo_FN', 'pseudo_TN')}
        self.bins = {name: _empty() for name, _, _ in self.BINS}
        self.classes = [_empty() for _ in range(num_classes)]
        self.active = False

    @torch.no_grad()
    def update(self, logits, label_vec_true, diagnostics):
        # The loss and all masks have already been computed without true labels.
        logits = logits.detach()
        truth = label_vec_true.detach()
        candidate = diagnostics['candidate_mask']
        pseudo = diagnostics['pseudo_positive_mask']
        reject = diagnostics['reject_only_mask']
        self.active |= diagnostics['pseudo_positive_active']
        for name, mask in (('candidate', candidate), ('pseudo', pseudo),
                           ('reject', reject)):
            _add(self.groups[name], logits[mask], truth[mask])
        for name, mask in (('pseudo_FN', pseudo & (truth == 1)),
                           ('pseudo_TN', pseudo & (truth != 1))):
            _add(self.groups[name], logits[mask], truth[mask])

        for class_id in range(logits.size(1)):
            mask = pseudo[:, class_id]
            _add(self.classes[class_id], logits[:, class_id][mask],
                 truth[:, class_id][mask])

        # Reconstruct candidate rank using raw BCE only; bins never feed loss.
        indices = torch.nonzero(candidate.flatten()).flatten()
        n = indices.numel()
        if n:
            losses = diagnostics['raw_loss_matrix'].flatten()[indices]
            order = torch.argsort(losses, descending=True, stable=True)
            ranked = indices[order]
            pseudo_flat = pseudo.flatten()
            for name, lo, hi in self.BINS:
                start = math.ceil(lo * n)
                end = math.ceil(hi * n)
                selected = ranked[start:end]
                selected = selected[pseudo_flat[selected]]
                _add(self.bins[name], logits.flatten()[selected],
                     truth.flatten()[selected])

    def write(self, directory, epoch, ratio, start_epoch):
        candidate = self.groups['candidate']
        pseudo = self.groups['pseudo']
        reject = self.groups['reject']
        fn = self.groups['pseudo_FN']
        tn = self.groups['pseudo_TN']
        row = dict(
            epoch=epoch, active=self.active, pseudo_positive_ratio=ratio,
            start_epoch=start_epoch,
            candidate_count=candidate['count'], candidate_FN=candidate['FN'],
            candidate_TN=candidate['count'] - candidate['FN'],
            candidate_precision=_precision(candidate),
            pseudo_positive_count=pseudo['count'], pseudo_positive_FN=pseudo['FN'],
            pseudo_positive_TN=pseudo['count'] - pseudo['FN'],
            pseudo_positive_precision=_precision(pseudo),
            reject_tail_count=reject['count'], reject_tail_FN=reject['FN'],
            reject_tail_TN=reject['count'] - reject['FN'],
            reject_tail_FN_precision=_precision(reject),
            pseudo_positive_logit_mean=_mean(pseudo, 'logit'),
            pseudo_positive_prob_mean=_mean(pseudo, 'prob'),
            pseudo_positive_positive_bce_mean=_mean(pseudo, 'positive_bce'),
            pseudo_positive_grad_mag_mean=_mean(pseudo, 'grad_mag'),
            pseudo_FN_count=fn['count'], pseudo_TN_count=tn['count'],
            pseudo_FN_prob_mean=_mean(fn, 'prob'),
            pseudo_TN_prob_mean=_mean(tn, 'prob'),
            pseudo_FN_positive_bce_mean=_mean(fn, 'positive_bce'),
            pseudo_TN_positive_bce_mean=_mean(tn, 'positive_bce'),
            pseudo_FN_grad_mag_mean=_mean(fn, 'grad_mag'),
            pseudo_TN_grad_mag_mean=_mean(tn, 'grad_mag'),
        )
        _write(os.path.join(directory, 'llr_pseudo_positive_recovery.csv'),
               self.SUMMARY_FIELDS, [row])
        bin_rows = [dict(epoch=epoch, bin=name, count=g['count'], FN=g['FN'],
                         TN=g['count'] - g['FN'], precision=_precision(g),
                         prob_mean=_mean(g, 'prob'),
                         positive_bce_mean=_mean(g, 'positive_bce'),
                         grad_mag_mean=_mean(g, 'grad_mag'))
                    for name, g in self.bins.items()]
        _write(os.path.join(directory, 'llr_pseudo_positive_rank_bins.csv'),
               self.BIN_FIELDS, bin_rows)
        class_rows = [dict(epoch=epoch, class_id=i, pseudo_count=g['count'],
                           pseudo_FN=g['FN'], pseudo_TN=g['count'] - g['FN'],
                           pseudo_precision=_precision(g),
                           pseudo_prob_mean=_mean(g, 'prob'),
                           pseudo_positive_bce_mean=_mean(g, 'positive_bce'),
                           pseudo_grad_mag_mean=_mean(g, 'grad_mag'))
                      for i, g in enumerate(self.classes)]
        _write(os.path.join(directory, 'llr_pseudo_positive_per_class.csv'),
               self.CLASS_FIELDS, class_rows)
