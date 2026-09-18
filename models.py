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
        self.boostlu_mode = P.get('boostlu_mode', 'fixed')
        self.adaptive_tau = P.get('adaptive_tau', 0.9)
        self.adaptive_temp = P.get('adaptive_temp', 0.1)
        self.adaptive_topk_ratio = P.get('adaptive_topk_ratio', 0.20)
        self.adaptive_detach_gate = P.get('adaptive_detach_gate', True)

        if self.boostlu_mode not in ['fixed', 'adaptive']:
            raise ValueError('boostlu_mode must be either fixed or adaptive.')
        if not self.adaptive_detach_gate:
            raise ValueError('Adaptive BoostLU requires a detached gate.')
        if not 0 < self.adaptive_topk_ratio <= 1:
            raise ValueError('adaptive_topk_ratio must be in (0, 1].')
        if self.adaptive_temp <= 0:
            raise ValueError('adaptive_temp must be positive.')

    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        
    def _relative_positive_strength(self, cam_raw):
        positive = torch.relu(cam_raw.detach())
        batch_size, num_classes, height, width = positive.shape
        flat = positive.flatten(2)
        k = max(1, int(round(height * width * self.adaptive_topk_ratio)))
        topk_values = torch.topk(
            flat,
            k=k,
            dim=2,
            largest=True,
            sorted=False,
        ).values
        scale = topk_values.mean(dim=2, keepdim=True).clamp_min(1e-6)
        scale = scale.view(batch_size, num_classes, 1, 1)
        return positive, positive / scale

    def _fixed_boostlu(self, cam_raw):
        alpha_map = torch.full_like(cam_raw, self.alpha)
        return torch.where(cam_raw > 0, cam_raw * alpha_map, cam_raw), alpha_map

    def _adaptive_boostlu(self, cam_raw, z):
        gate = torch.sigmoid((z - self.adaptive_tau) / self.adaptive_temp)
        alpha_map = 1.0 + (self.alpha - 1.0) * gate
        return torch.where(cam_raw > 0, cam_raw * alpha_map, cam_raw), alpha_map

    def _boost_diagnostics(self, cam_raw, cam_boosted, alpha_map, positive, z):
        positive_sum = positive.sum(dim=(2, 3))
        positive_valid = positive_sum > 1e-6
        positive_pixel_mask = positive > 0
        positive_pixel_count = positive_pixel_mask.sum(dim=(2, 3))

        diagnostics = {
            'delta_g': (
                self.avgpool(cam_boosted.detach()) - self.avgpool(cam_raw.detach())
            ),
            'alpha_eff': (
                (alpha_map.detach() * positive).sum(dim=(2, 3))
                / positive_sum.clamp_min(1e-6)
            ),
            'alpha_eff_valid': positive_valid,
            'positive_cam_mass': positive.mean(dim=(2, 3)),
            'dominant_ratio': (
                ((z >= 1) & positive_pixel_mask).sum(dim=(2, 3)).float()
                / positive_pixel_count.clamp_min(1)
            ),
            'dominant_ratio_valid': positive_pixel_count > 0,
            'alpha_map_requires_grad': alpha_map.requires_grad,
        }
        return {key: value.detach() if torch.is_tensor(value) else value
                for key, value in diagnostics.items()}

    def forward(self, x, return_boost_diagnostics=False):
        feats = self.feature_extractor(x)
        cam_raw = self.onebyone_conv(feats)

        if self.boostlu_mode == 'adaptive':
            positive, z = self._relative_positive_strength(cam_raw)
            cam_boosted, alpha_map = self._adaptive_boostlu(cam_raw, z)
        else:
            cam_boosted, alpha_map = self._fixed_boostlu(cam_raw)

        logits = F.adaptive_avg_pool2d(cam_boosted, 1).squeeze(-1).squeeze(-1)
        if return_boost_diagnostics:
            if self.boostlu_mode != 'adaptive':
                positive, z = self._relative_positive_strength(cam_raw)
            diagnostics = self._boost_diagnostics(
                cam_raw, cam_boosted, alpha_map, positive, z
            )
            return logits, diagnostics
        return logits

