import random
import numpy as np
import torch
import datasets
import models
from metrics import MAP_PROTOCOL
from instrumentation import (
    LLRCandidatePoolOracleAccumulator,
    compute_metrics,
)
import losses
import os
import csv
import json


def load_semantic_cache(P, dataset):
    score_file = P.get('semantic_score_file')
    if not score_file:
        raise ValueError('semantic_score_file is required for Semantic-Adaptive BoostLU')
    metadata_file = os.path.splitext(score_file)[0] + '.json'
    with open(metadata_file, encoding='utf-8') as handle:
        metadata = json.load(handle)
    expected = {'dataset': P['dataset'], 'clip_model': 'ViT-B/16',
                'num_classes': P['num_classes'],
                'split_seed': P['split_seed'], 'ss_seed': P['ss_seed'],
                'val_frac': P['val_frac'], 'ss_frac_train': P['ss_frac_train'],
                'ss_frac_val': P['ss_frac_val']}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Semantic cache metadata mismatch for {key}: {metadata.get(key)!r} != {value!r}')
    A = float(P.get('semantic_lambda_global', 0.5))
    if not 0.0 <= A <= 1.0:
        raise ValueError('semantic_lambda_global must be in [0,1]')
    semantic_q = {}
    with np.load(score_file, allow_pickle=False) as cache:
        component_keys = {f'q_{kind}_{phase}' for kind in ('global', 'local')
                          for phase in ('train', 'val', 'test')}
        available = component_keys.intersection(cache.files)
        if available and available != component_keys:
            raise ValueError('Semantic cache has incomplete global/local components')
        has_components = available == component_keys
        if has_components:
            if metadata.get('cache_schema_version') != 2 or metadata.get('score_source') != 'qz_global_local':
                raise ValueError('Semantic cache schema v2 metadata mismatch')
        elif A != 0.5:
            raise ValueError('Legacy semantic cache contains only pre-fused q.\n'
                             'Regenerate cache with precompute_clip_qz.py\n'
                             'to use semantic_lambda_global != 0.5.')
        elif metadata.get('score_source') != 'qz_combined':
            raise ValueError('Legacy semantic cache score_source mismatch')
        for phase in ('train', 'val', 'test'):
            image_ids = cache[f'{phase}_image_ids']
            if not np.array_equal(image_ids, dataset[phase].image_ids):
                raise ValueError(f'Semantic cache image IDs do not match {phase} split')
            names = (f'q_global_{phase}', f'q_local_{phase}') if has_components else (f'q_{phase}',)
            for name in names:
                part = cache[name]
                if part.shape != (len(dataset[phase]), P['num_classes']):
                    raise ValueError(f'Semantic cache {name} shape mismatch: {part.shape}')
                if not np.isfinite(part).all() or np.any(part < 0) or np.any(part > 1):
                    raise ValueError(f'Semantic cache {name} must be finite and in [0,1]')
            q = ((A * cache[names[0]] + (1.0 - A) * cache[names[1]]).astype(np.float32)
                 if has_components else cache[names[0]])
            semantic_q[phase] = torch.from_numpy(np.array(q, dtype=np.float32, copy=True))
    print('[Semantic-Adaptive BoostLU]')
    print(f'enabled=True\nalpha0={float(P["alpha"])}\ndelta={float(P["semantic_delta"])}')
    print(f'semantic_lambda_global={A}\nsemantic_lambda_local={1.0 - A}')
    print(f'expected alpha range=[{P["alpha"] - P["semantic_delta"]},'
          f'{P["alpha"] + P["semantic_delta"]}]')
    print(f'score source={"qz_global_local" if has_components else "qz_combined"}'
          f'\nCLIP=ViT-B/16\nsemantic score file={score_file}')
    for phase in ('train', 'val', 'test'):
        print(f'q_{phase} shape={tuple(semantic_q[phase].shape)}')
    return semantic_q


class SemanticBoostDiagnostics:
    FIELDS = ('epoch', 'q_hidden_FN_mean', 'q_TN_mean', 'q_observed_positive_mean',
              'alpha_hidden_FN_mean', 'alpha_TN_mean', 'alpha_observed_positive_mean',
              'alpha_mean', 'alpha_std', 'alpha_min', 'alpha_max')

    def __init__(self, alpha0, delta):
        self.alpha0 = alpha0
        self.delta = delta
        self.group_sums = [0.0, 0.0, 0.0]
        self.group_counts = [0, 0, 0]
        self.alpha_sum = 0.0
        self.alpha_sumsq = 0.0
        self.alpha_count = 0
        self.alpha_min = float('inf')
        self.alpha_max = -float('inf')

    @torch.no_grad()
    def update(self, q, observed, true):
        q = q.detach()
        observed = observed.detach()
        true = true.detach()
        masks = ((observed == 0) & (true == 1),
                 (observed == 0) & (true != 1), observed == 1)
        for k, mask in enumerate(masks):
            self.group_sums[k] += q[mask].double().sum().item()
            self.group_counts[k] += mask.sum().item()
        alpha = self.alpha0 + self.delta * (2.0 * q - 1.0)
        self.alpha_sum += alpha.double().sum().item()
        self.alpha_sumsq += alpha.double().square().sum().item()
        self.alpha_count += alpha.numel()
        self.alpha_min = min(self.alpha_min, alpha.min().item())
        self.alpha_max = max(self.alpha_max, alpha.max().item())

    def row(self, epoch):
        means = [s / n if n else float('nan')
                 for s, n in zip(self.group_sums, self.group_counts)]
        alpha_mean = self.alpha_sum / self.alpha_count
        variance = max(0.0, self.alpha_sumsq / self.alpha_count - alpha_mean ** 2)
        return dict(zip(self.FIELDS, (epoch, *means,
                                      *(self.alpha0 + self.delta * (2.0 * q - 1.0)
                                        for q in means),
                                      alpha_mean, variance ** 0.5,
                                      self.alpha_min, self.alpha_max)))

    @classmethod
    def append_csv(cls, path, row):
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=cls.FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_train(P):
    path = os.path.join(P['save_path'], 'bestmodel.pt')
    # There is no resume implementation: never reuse historical selection state
    # or overwrite an existing (possibly sigmoid-scored) checkpoint.
    if os.path.exists(path):
        raise FileExistsError(f'Use a fresh save_path; preserving checkpoint: {path}')
    seed_everything(P['seed'])
    print(
        f"[Reproducibility] seed={P['seed']}, "
        f"cudnn_deterministic={torch.backends.cudnn.deterministic}, "
        f"cudnn_benchmark={torch.backends.cudnn.benchmark}"
    )
    dataset = datasets.get_data(P)
    semantic_q = load_semantic_cache(P, dataset) if P.get('semantic_adaptive_boostlu', False) else None
    if np.min(dataset['train'].label_matrix_obs) < 0:
        raise ValueError('Observed training labels must be non-negative.')

    dataloader = {}
    phase_seed_offset = {'train': 0, 'val': 1, 'test': 2}
    for phase in ['train', 'val', 'test']:
        generator = torch.Generator()
        generator.manual_seed(P['seed'] + phase_seed_offset[phase])
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size = P['bsize'],
            shuffle = phase == 'train',
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False,
            pin_memory = True,
            worker_init_fn=seed_worker,
            generator=generator,
        )
    
    model = models.ImageClassifier(P)
    
    feature_extractor_params = [param for param in list(model.feature_extractor.parameters()) if param.requires_grad]
    onebyone_conv_params = [param for param in list(model.onebyone_conv.parameters()) if param.requires_grad]
    opt_params = [
        {'params': feature_extractor_params, 'lr' : P['lr']},
        {'params': onebyone_conv_params, 'lr' : P['lr_mult'] * P['lr']}
    ]
  
    optimizer = torch.optim.Adam(opt_params, lr=P['lr'])
    # training loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    bestmap_val = -float('inf')
    print(f'[Evaluation] mAP protocol={MAP_PROTOCOL}; alpha={P["alpha"]}')
    rank_oracle_path = os.path.join(
        P['save_path'], 'llr_candidate_pool_oracle_curve.csv'
    )

    for epoch in range(1, P['num_epochs']+1):
        clean_rate_used = P['clean_rate']
        rank_oracle_diag = LLRCandidatePoolOracleAccumulator()
        semantic_diag = (SemanticBoostDiagnostics(P['alpha'], P['semantic_delta'])
                         if semantic_q is not None else None)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                y_true = np.zeros((len(dataset[phase]), P['num_classes']))
                pred_batches = []
                batch_stack = 0

            grad_context = torch.enable_grad() if phase == 'train' else torch.inference_mode()
            with grad_context:
                for batch in dataloader[phase]:
                    image = batch['image'].to(device, non_blocking=True)

                    if phase == 'train':
                        label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)

                    semantic_q_batch = None
                    if semantic_q is not None:
                        batch_idx = batch['idx'].long()
                        semantic_q_batch = semantic_q[phase].index_select(0, batch_idx).to(
                            device, non_blocking=True
                        )
                    logits = model(image, semantic_q=semantic_q_batch)
                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)

                    if phase == 'train':
                        loss, correction_idx, loss_diag = losses.compute_batch_loss(
                            logits, label_vec_obs, P, return_diagnostics=True,
                        )
                        label_vec_true = batch['label_vec_true'].to(
                            device, non_blocking=True
                        )
                        if semantic_diag is not None:
                            semantic_diag.update(semantic_q_batch, label_vec_obs, label_vec_true)
                        rank_oracle_diag.update(
                            label_vec_obs,
                            label_vec_true,
                            loss_diag['raw_loss_matrix'],
                            loss_diag['rejection_mask'],
                        )
                        loss.backward()
                        optimizer.step()

                        if P['largelossmod_scheme'] == 'LL-Cp' and correction_idx[1].numel():
                            idx = batch['idx']
                            dataset[phase].label_matrix_obs[idx[correction_idx[0].cpu()], correction_idx[1].cpu()] = 1.0
                    else:
                        pred_batches.append(logits.to(device='cpu', dtype=torch.float64))
                        label_vec_true = batch['label_vec_true'].numpy()
                        this_batch_size = label_vec_true.shape[0]
                        y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
                        batch_stack += this_batch_size

                if phase == 'val':
                    assert batch_stack == len(dataset[phase])
                    y_pred = torch.cat(pred_batches, dim=0).cpu().numpy()
                    metrics = compute_metrics(y_pred, y_true)
                elif phase == 'train':
                    rank_oracle_rows = rank_oracle_diag.summarize(
                        epoch, clean_rate_used
                    )
                    rank_oracle_diag.append_csv(
                        rank_oracle_path, rank_oracle_rows
                    )
                    rank_oracle_diag.print_summary(rank_oracle_rows)
                    if semantic_diag is not None:
                        semantic_row = semantic_diag.row(epoch)
                        semantic_diag.append_csv(
                            os.path.join(P['save_path'], 'semantic_boost_diagnostics.csv'),
                            semantic_row,
                        )
                        print(f'[Semantic][Epoch {epoch}] mean alpha FN='
                              f'{semantic_row["alpha_hidden_FN_mean"]:.4f}; '
                              f'mean alpha TN={semantic_row["alpha_TN_mean"]:.4f}; '
                              f'LL-R candidate FN precision='
                              f'{rank_oracle_rows[0]["pool_precision"]:.4f}')
        del y_pred
        del y_true
        map_val = metrics['map']
                
        print(f"Epoch {epoch} : val mAP {map_val:.3f}")
        P['clean_rate'] -= P['delta_rel']
                
        if bestmap_val < map_val:
            bestmap_val = map_val
            bestmap_epoch = epoch
            
            print(f'Saving model weight for best val mAP {bestmap_val:.3f}')
            path = os.path.join(P['save_path'], 'bestmodel.pt')
            checkpoint_config = dict(P)
            checkpoint_config.update(
                map_protocol=MAP_PROTOCOL,
                bestmap_val=bestmap_val,
                bestmap_epoch=bestmap_epoch,
                best_alpha=P['alpha'],
            )
            torch.save((model.state_dict(), checkpoint_config), path)

    # Test phase
    path = os.path.join(P['save_path'], 'bestmodel.pt')
    model_state, _ = torch.load(path)
    model.load_state_dict(model_state)

    phase = 'test'
    model.eval()
    y_true = np.zeros((len(dataset[phase]), P['num_classes']))
    pred_batches = []
    batch_stack = 0
    with torch.inference_mode():
        for batch in dataloader[phase]:
            image = batch['image'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].numpy()

            semantic_q_batch = None
            if semantic_q is not None:
                batch_idx = batch['idx'].long()
                semantic_q_batch = semantic_q[phase].index_select(0, batch_idx).to(
                    device, non_blocking=True
                )
            logits = model(image, semantic_q=semantic_q_batch)
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            pred_batches.append(logits.to(device='cpu', dtype=torch.float64))

            this_batch_size = label_vec_true.shape[0]
            y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
            batch_stack += this_batch_size

        assert batch_stack == len(dataset[phase])
        y_pred = torch.cat(pred_batches, dim=0).cpu().numpy()
    metrics = compute_metrics(y_pred, y_true)
    map_test = metrics['map']
    ap_test = metrics['ap']

    print('Training procedure completed!')
    print(f'Test mAP : {map_test:.3f} when trained until epoch {bestmap_epoch}')

    np.save(os.path.join(P['save_path'], 'test_ap.npy'), ap_test)
