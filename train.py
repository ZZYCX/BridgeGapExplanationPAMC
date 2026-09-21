import random
import numpy as np
import torch
import datasets
import models
from instrumentation import (
    LLRCandidatePoolOracleAccumulator,
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
    rank_oracle_path = os.path.join(
        P['save_path'], 'llr_candidate_pool_oracle_curve.csv'
    )

    for epoch in range(1, P['num_epochs']+1):
        clean_rate_used = P['clean_rate']
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

                    logits = model(image)
                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)

                    if phase == 'train':
                        loss, correction_idx, loss_diag = losses.compute_batch_loss(
                            logits, label_vec_obs, P, return_diagnostics=True,
                        )
                        label_vec_true = batch['label_vec_true'].to(
                            device, non_blocking=True
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
