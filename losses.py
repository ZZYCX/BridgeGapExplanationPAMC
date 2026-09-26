import torch
import torch.nn.functional as F
import math 

   

'''
loss functions
'''

def loss_an(logits, observed_labels, compute_corrected=False):
    loss_matrix = F.binary_cross_entropy_with_logits(logits, observed_labels, reduction='none')
    corrected_loss_matrix = None
    if compute_corrected:
        corrected_labels = torch.logical_not(observed_labels).float()
        corrected_loss_matrix = F.binary_cross_entropy_with_logits(
            logits,
            corrected_labels,
            reduction='none',
        )
    return loss_matrix, corrected_loss_matrix


'''
top-level wrapper
'''

def compute_batch_loss(logits, label_vec, P, return_diagnostics=False, epoch=None):
     
    assert logits.dim() == 2
    
    batch_size = int(logits.size(0))
    num_classes = int(logits.size(1))
    

    if P['dataset'] == 'OPENIMAGES':
        unobserved_mask = (label_vec == -1)
    else:
        unobserved_mask = (label_vec == 0)
    
    # Corrected BCE is also needed for temporary LL-R positive recovery.
    compute_corrected = (
        P['clean_rate'] != 1
        and (P['largelossmod_scheme'] in ['LL-Ct', 'LL-Cp']
             or P.get('llr_pseudo_positive_recovery', False))
    )
    loss_matrix, corrected_loss_matrix = loss_an(
        logits,
        label_vec.clip(0),
        compute_corrected=compute_corrected,
    )

    correction_idx = [torch.Tensor([]), torch.Tensor([])]
    rejection_mask = torch.zeros_like(unobserved_mask, dtype=torch.bool)
    candidate_mask = torch.zeros_like(rejection_mask)
    pseudo_positive_mask = torch.zeros_like(rejection_mask)
    reject_only_mask = torch.zeros_like(rejection_mask)
    pseudo_positive_active = (
        P.get('llr_pseudo_positive_recovery', False)
        and P['largelossmod_scheme'] == 'LL-R'
        and epoch is not None
        and epoch >= P.get('llr_pseudo_positive_start_epoch', 5)
        and P['clean_rate'] != 1
    )
    pseudo_positive_ratio = P.get('llr_pseudo_positive_ratio', 0.20)

    if P['clean_rate'] == 1: # if epoch is 1, do not modify losses
        final_loss_matrix = loss_matrix
    else:
        if P['largelossmod_scheme'] == 'LL-Cp':
            k = math.ceil(batch_size * num_classes * P['delta_rel'])
        else:
            k = math.ceil(batch_size * num_classes * (1-P['clean_rate']))
    
        unobserved_loss = unobserved_mask.bool() * loss_matrix
        topk = torch.topk(unobserved_loss.flatten(), k)
        topk_lossvalue = topk.values[-1]
        rejection_mask = unobserved_loss >= topk_lossvalue
        correction_idx = torch.where(rejection_mask)
        candidate_mask = rejection_mask & unobserved_mask.bool()
        reject_only_mask = candidate_mask.clone()


        if P['largelossmod_scheme'] in ['LL-Ct', 'LL-Cp']:
            final_loss_matrix = torch.where(rejection_mask, corrected_loss_matrix, loss_matrix)
        else:
            zero_loss_matrix = torch.zeros_like(loss_matrix)
            final_loss_matrix = torch.where(rejection_mask, zero_loss_matrix, loss_matrix)
            if pseudo_positive_active:
                candidate_indices = torch.nonzero(candidate_mask.flatten()).flatten()
                candidate_losses = loss_matrix.flatten()[candidate_indices]
                n_candidate = candidate_indices.numel()
                n_corr = math.ceil(pseudo_positive_ratio * n_candidate)
                if pseudo_positive_ratio == 0:
                    n_corr = 0
                elif pseudo_positive_ratio == 1:
                    n_corr = n_candidate
                if n_corr:
                    ranking = torch.argsort(candidate_losses, descending=True, stable=True)
                    pseudo_positive_mask.flatten()[candidate_indices[ranking[:n_corr]]] = True
                    reject_only_mask = candidate_mask & ~pseudo_positive_mask
                    final_loss_matrix = loss_matrix.clone()
                    final_loss_matrix[candidate_mask] = 0
                    final_loss_matrix[pseudo_positive_mask] = corrected_loss_matrix[pseudo_positive_mask]

    assert not (pseudo_positive_mask & ~candidate_mask).any()
    assert not (reject_only_mask & ~candidate_mask).any()
    assert not (pseudo_positive_mask & reject_only_mask).any()
    assert torch.equal(pseudo_positive_mask | reject_only_mask, candidate_mask)
    assert not (pseudo_positive_mask & ~unobserved_mask.bool()).any()
                
    main_loss = final_loss_matrix.mean()

    if return_diagnostics:
        loss_diagnostics = {
            'raw_loss_matrix': loss_matrix.detach(),
            'unobserved_mask': unobserved_mask.detach().bool(),
            'rejection_mask': rejection_mask.detach(),
            'candidate_mask': candidate_mask.detach(),
            'pseudo_positive_mask': pseudo_positive_mask.detach(),
            'reject_only_mask': reject_only_mask.detach(),
            'pseudo_positive_active': pseudo_positive_active,
            'pseudo_positive_ratio': pseudo_positive_ratio,
        }
        return main_loss, correction_idx, loss_diagnostics

    return main_loss, correction_idx
