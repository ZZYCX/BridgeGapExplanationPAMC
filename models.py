import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)

class ImageClassifier(torch.nn.Module):
    def __init__(self, P):
        super(ImageClassifier, self).__init__()
        
        self.arch = P['arch']
        if P['dataset'] == 'OPENIMAGES':
            feature_extractor = torchvision.models.resnet101(pretrained=P['use_pretrained'])
        else:
            feature_extractor = torchvision.models.resnet50(pretrained=P['use_pretrained'])
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])

        if P['freeze_feature_extractor']:
            for param in feature_extractor.parameters():
                param.requires_grad = False
        else:
            for param in feature_extractor.parameters():
                param.requires_grad = True

        self.feature_extractor = feature_extractor
        self.avgpool = GlobalAvgPool2d()
        self.onebyone_conv = nn.Conv2d(P['feat_dim'], P['num_classes'], 1)
        self.alpha = P['alpha']
        self.semantic_adaptive_boostlu = P.get('semantic_adaptive_boostlu', False)
        self.semantic_delta = P.get('semantic_delta', 3.0)
        self.is_cub = P['dataset'] == 'cub'
        if self.is_cub:
            # Derived from the final training split; not a checkpoint parameter.
            self.register_buffer('class_has_observed_positive', None, persistent=False)

    def set_cub_observed_positive_mask(self, observed_labels):
        if not self.is_cub:
            raise ValueError('Observed-positive BoostLU mask is only defined for CUB')
        observed = torch.as_tensor(observed_labels)
        if observed.ndim != 2 or observed.shape[1] != self.onebyone_conv.out_channels:
            raise ValueError('CUB observed labels must have shape [N, num_classes]')
        mask = (observed == 1).any(dim=0)
        self.class_has_observed_positive = mask.to(self.onebyone_conv.weight.device)
        zero_ids = torch.where(~mask)[0].tolist()
        print('[CUB BoostLU mask]')
        print(f'num_classes={len(mask)}')
        print(f'num_zero_observed_positive={len(zero_ids)}')
        print(f'zero_observed_positive_class_ids={zero_ids} (0-based label columns)')

    def _apply_cub_training_mask(self, cam_raw, cam_boosted):
        if self.is_cub and self.training:
            if self.class_has_observed_positive is None:
                raise RuntimeError('Initialize CUB BoostLU mask from final training observed labels')
            return torch.where(self.class_has_observed_positive[None, :, None, None],
                               cam_boosted, cam_raw)
        return cam_boosted

    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        
    def _fixed_boostlu(self, cam_raw):
        return self._apply_cub_training_mask(
            cam_raw, torch.where(cam_raw > 0, cam_raw * self.alpha, cam_raw))

    def _semantic_adaptive_boostlu(self, cam_raw, semantic_q):
        if cam_raw.ndim != 4 or semantic_q.ndim != 2 or semantic_q.shape != cam_raw.shape[:2]:
            raise ValueError('Expected cam_raw [B,C,H,W] and semantic_q [B,C]')
        if not torch.isfinite(semantic_q).all():
            raise ValueError('semantic_q must be finite')
        semantic_q = semantic_q.clamp(0.0, 1.0)
        alpha_ic = self.alpha + self.semantic_delta * (2.0 * semantic_q - 1.0)
        alpha_map = alpha_ic[:, :, None, None]
        return self._apply_cub_training_mask(
            cam_raw, torch.where(cam_raw > 0, cam_raw * alpha_map, cam_raw))

    def forward(self, x, semantic_q=None):
        if self.semantic_adaptive_boostlu and semantic_q is None:
            raise ValueError('semantic_q is required when semantic adaptive BoostLU is enabled')
        feats = self.feature_extractor(x)
        cam_raw = self.onebyone_conv(feats)
        if self.semantic_adaptive_boostlu:
            cam_boosted = self._semantic_adaptive_boostlu(cam_raw, semantic_q)
        else:
            cam_boosted = self._fixed_boostlu(cam_raw)
        return F.adaptive_avg_pool2d(cam_boosted, 1).squeeze(-1).squeeze(-1)
