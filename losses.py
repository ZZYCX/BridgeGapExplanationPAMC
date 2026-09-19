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

def compute_batch_loss(logits, label_vec, P, return_diagnostics=False,
                       recovery_seed_mask=None):
     
    assert logits.dim() == 2
    
    batch_size = int(logits.size(0))
    num_classes = int(logits.size(1))
    

    if P['dataset'] == 'OPENIMAGES':
        unobserved_mask = (label_vec == -1)
    else:
        unobserved_mask = (label_vec == 0)
    
    # LL-Ct/LL-Cp need the corrected-label BCE only after rejection starts.
    compute_corrected = (
        P['clean_rate'] != 1
        and P['largelossmod_scheme'] in ['LL-Ct', 'LL-Cp']
    )
    loss_matrix, corrected_loss_matrix = loss_an(
        logits,
        label_vec.clip(0),
        compute_corrected=compute_corrected,
    )

    correction_idx = [torch.Tensor([]), torch.Tensor([])]
    rejection_mask = torch.zeros_like(unobserved_mask, dtype=torch.bool)

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


        if P['largelossmod_scheme'] in ['LL-Ct', 'LL-Cp']:
            final_loss_matrix = torch.where(rejection_mask, corrected_loss_matrix, loss_matrix)
        else:
            zero_loss_matrix = torch.zeros_like(loss_matrix)
            final_loss_matrix = torch.where(rejection_mask, zero_loss_matrix, loss_matrix)
                
    # Recover only previous-core entries that remain actual LL-R candidates.
    # Keep rejection_mask and raw losses untouched for ranking and diagnostics.
    recovery_mask = torch.zeros_like(rejection_mask)
    if P['largelossmod_scheme'] == 'LL-R' and recovery_seed_mask is not None:
        recovery_mask = (
            rejection_mask & unobserved_mask.bool() & recovery_seed_mask.bool()
        )
        positive_loss_matrix = F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits), reduction='none'
        )
        final_loss_matrix = torch.where(
            recovery_mask, P['lambda_rec'] * positive_loss_matrix,
            final_loss_matrix,
        )

    main_loss = final_loss_matrix.mean()

    if return_diagnostics:
        loss_diagnostics = {
            'raw_loss_matrix': loss_matrix.detach(),
            'unobserved_mask': unobserved_mask.detach().bool(),
            'rejection_mask': rejection_mask.detach(),
            'recovery_mask': recovery_mask.detach(),
        }
        return main_loss, correction_idx, loss_diagnostics

    return main_loss, correction_idx
