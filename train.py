import random
import numpy as np
import torch
import datasets
import models
from instrumentation import (
    AdaptiveBoostDiagnosticsAccumulator,
    LLRCandidatePoolOracleAccumulator,
    ReliablePositiveRecoveryState,
    compute_metrics,
)
import losses
import os

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
    seed_everything(P['seed'])
    print(
        f"[Reproducibility] seed={P['seed']}, "
        f"cudnn_deterministic={torch.backends.cudnn.deterministic}, "
        f"cudnn_benchmark={torch.backends.cudnn.benchmark}"
    )
    dataset = datasets.get_data(P)
    if np.min(dataset['train'].label_matrix_obs) < 0:
        raise ValueError('Observed training labels must be non-negative.')

    recovery_state = None
    if P['largelossmod_scheme'] == 'LL-R' and P.get('use_pseudo_labels', False):
        recovery_state = ReliablePositiveRecoveryState(
            num_samples=len(dataset['train']),
            num_classes=P['num_classes'], top_q=P['top_q'],
        )

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

    bestmap_val = 0
    diagnostics_path = os.path.join(P['save_path'], 'adaptive_boost_diagnostics.csv')
    rank_oracle_path = os.path.join(
        P['save_path'], 'llr_candidate_pool_oracle_curve.csv'
    )

    for epoch in range(1, P['num_epochs']+1):
        clean_rate_used = P['clean_rate']
        recovered_count = 0
        if recovery_state is not None:
            recovery_state.start_epoch()
        diagnostics = AdaptiveBoostDiagnosticsAccumulator()
        rank_oracle_diag = LLRCandidatePoolOracleAccumulator()
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

                    if phase == 'train':
                        logits, boost_diag = model(
                            image, return_boost_diagnostics=True
                        )
                    else:
                        logits = model(image)
                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)

                    if phase == 'train':
                        recovery_seed_mask = None
                        if recovery_state is not None:
                            recovery_seed_mask = recovery_state.get_previous_core(
                                batch['idx']
                            ).to(device, non_blocking=True)
                        loss, correction_idx, loss_diag = losses.compute_batch_loss(
                            logits, label_vec_obs, P, return_diagnostics=True,
                            recovery_seed_mask=recovery_seed_mask,
                        )
                        if recovery_state is not None:
                            recovery_state.collect_current_candidates(
                                batch_idx=batch['idx'],
                                rejection_mask=loss_diag['rejection_mask'],
                                raw_loss_matrix=loss_diag['raw_loss_matrix'],
                            )
                        label_vec_true = batch['label_vec_true'].to(
                            device, non_blocking=True
                        )
                        if recovery_state is not None:
                            recovered_count += recovery_state.update_recovery_diagnostics(
                                loss_diag['recovery_mask'], logits,
                                label_vec_obs, label_vec_true,
                            )
                        diagnostics.update(
                            label_vec_obs, label_vec_true, boost_diag, loss_diag
                        )
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
                        pred_batches.append(torch.sigmoid(logits))
                        label_vec_true = batch['label_vec_true'].numpy()
                        this_batch_size = label_vec_true.shape[0]
                        y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
                        batch_stack += this_batch_size

                if phase == 'val':
                    y_pred = torch.cat(pred_batches, dim=0).cpu().numpy()
                    metrics = compute_metrics(y_pred, y_true)
                elif phase == 'train':
                    if recovery_state is not None:
                        recovery_summary = recovery_state.finalize_epoch(recovered_count)
                        recovery_summary.update(
                            epoch=epoch, clean_rate_used=clean_rate_used,
                            lambda_rec=P['lambda_rec'],
                        )
                        recovery_state.append_csv(
                            os.path.join(P['save_path'], 'reliable_positive_recovery.csv'),
                            recovery_summary,
                        )
                        recovery_state.print_summary(recovery_summary)
                    diagnostics_summary = diagnostics.summarize(
                        epoch, clean_rate_used
                    )
                    diagnostics.append_csv(
                        diagnostics_path, diagnostics_summary
                    )
                    diagnostics.print_summary(diagnostics_summary)
                    rank_oracle_rows = rank_oracle_diag.summarize(
                        epoch, clean_rate_used
                    )
                    rank_oracle_diag.append_csv(
                        rank_oracle_path, rank_oracle_rows
                    )
                    rank_oracle_diag.print_summary(rank_oracle_rows)
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
            torch.save((model.state_dict(), P), path)

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

            logits = model(image)
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            pred_batches.append(torch.sigmoid(logits))

            this_batch_size = label_vec_true.shape[0]
            y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
            batch_stack += this_batch_size

        y_pred = torch.cat(pred_batches, dim=0).cpu().numpy()
    metrics = compute_metrics(y_pred, y_true)
    map_test = metrics['map']
    ap_test = metrics['ap']

    print('Training procedure completed!')
    print(f'Test mAP : {map_test:.3f} when trained until epoch {bestmap_epoch}')

    np.save(os.path.join(P['save_path'], 'test_ap.npy'), ap_test)
