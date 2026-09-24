"""Precompute frozen CLIP qz_combined evidence for the baseline COCO split.
   用冻结的 CLIP ViT-B/16 分别计算原图的 global 相似度，以及五个固定裁剪中最大的 local 相似度。
   每个类别只用 train split 的分数计算均值和标准差，再把 train、val、test 的分数标准化、经过 sigmoid，以默认的 0.5/0.5 权重合成 \(q_{ic}\in[0,1]\)。结果保存为 NPZ 缓存及同名 JSON 元数据
"""

#CLIP location: /home/ubuntu2/.cache/clip/ViT-B-16.pt    --clip-model "ViT-B/16" --clip-cache /home/ubuntu2/.cache/clip
#CLIP location: /home/sx639/BridgeGapExplanationPAMC_official/data/ViT-B-16.pt    --clip-model "ViT-B/16" --clip-cache /home/sx639/BridgeGapExplanationPAMC_official/data
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import datasets


SPLIT_SEED = 1200
SS_SEED = 999
VAL_FRAC = 0.2
SS_FRAC_TRAIN = 1.0
SS_FRAC_VAL = 1.0
EPS = 1e-6
PROMPT = 'a photo of a {class_name}'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='coco')
    p.add_argument('--clip-model', default='ViT-B/16')
    p.add_argument('--clip-cache', default='/home/ubuntu2/.cache/clip')
    p.add_argument('--output', required=True)
    p.add_argument('--clip-batch-size', type=int, default=32)
    p.add_argument('--local-crop-ratio', type=float, default=0.6)
    p.add_argument('--lambda-global', type=float, default=0.5)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.dataset != 'coco' or args.clip_model != 'ViT-B/16':
        p.error('This cache requires COCO and OpenAI CLIP ViT-B/16')
    if args.clip_batch_size < 1 or not 0 < args.local_crop_ratio <= 1:
        p.error('clip batch size must be positive and local crop ratio must be in (0,1]')
    if not 0 <= args.lambda_global <= 1:
        p.error('lambda-global must be in [0,1]')
    if not args.output.endswith('.npz'):
        p.error('--output must be an .npz path')
    return args


def split_image_ids(dataset_name):
    """Mirror datasets.multilabel split order using only image ID arrays."""
    base = datasets.get_metadata(dataset_name)['path_to_dataset']
    train_ids = np.load(os.path.join(base, 'formatted_train_images.npy'), allow_pickle=False)
    test_ids = np.load(os.path.join(base, 'formatted_val_images.npy'), allow_pickle=False)
    train_idx, val_idx = datasets.generate_split(
        len(train_ids), VAL_FRAC, np.random.RandomState(SPLIT_SEED)
    )
    split_idx = {'train': train_idx, 'val': val_idx}
    ss_rng = np.random.RandomState(SS_SEED)
    for phase, fraction in (('train', SS_FRAC_TRAIN), ('val', SS_FRAC_VAL)):
        idx = split_idx[phase]
        count = int(np.round(fraction * len(idx)))
        split_idx[phase] = idx[np.sort(ss_rng.permutation(len(idx))[:count])]
    return {'train': train_ids[split_idx['train']],
            'val': train_ids[split_idx['val']], 'test': test_ids}


class ClipViews(Dataset):
    def __init__(self, image_ids, image_root, preprocess, crop_ratio):
        self.image_ids = image_ids
        self.image_root = image_root
        self.preprocess = preprocess
        self.crop_ratio = crop_ratio

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        path = os.path.join(self.image_root, str(self.image_ids[index]))
        with Image.open(path) as source:
            image = source.convert('RGB')
        width, height = image.size
        cw = max(1, min(width, int(round(width * self.crop_ratio))))
        ch = max(1, min(height, int(round(height * self.crop_ratio))))
        boxes = [(0, 0), (width - cw, 0), (0, height - ch),
                 (width - cw, height - ch), ((width - cw) // 2, (height - ch) // 2)]
        global_view = self.preprocess(image)
        local_views = torch.stack([
            self.preprocess(image.crop((x, y, x + cw, y + ch))) for x, y in boxes
        ])
        return global_view, local_views


@torch.inference_mode()
def encode_chunks(model, images, text_features, chunk_size, device):
    scores = []
    for chunk in images.split(chunk_size):
        embedding = F.normalize(model.encode_image(chunk.to(device)).float(), dim=-1)
        scores.append((embedding @ text_features.T).float().cpu().numpy())
    return np.concatenate(scores, axis=0)


@torch.inference_mode()
def extract_split(image_ids, image_root, preprocess, crop_ratio, model,
                  text_features, clip_batch_size, device, phase):
    views = ClipViews(image_ids, image_root, preprocess, crop_ratio)
    loader = DataLoader(views, batch_size=max(1, clip_batch_size // 6),
                        shuffle=False, num_workers=0)
    raw_global, raw_local = [], []
    for batch_number, (global_views, local_views) in enumerate(loader, 1):
        global_score = encode_chunks(model, global_views, text_features,
                                     clip_batch_size, device)
        local_score = encode_chunks(model, local_views.flatten(0, 1),
                                    text_features, clip_batch_size, device)
        raw_global.append(global_score.astype(np.float32))
        raw_local.append(local_score.reshape(len(global_views), 5, -1)
                         .max(axis=1).astype(np.float32))
        if batch_number % 100 == 0 or batch_number == len(loader):
            print(f'[{phase}] {min(batch_number * loader.batch_size, len(views))}/{len(views)}',
                  flush=True)
    return np.concatenate(raw_global), np.concatenate(raw_local)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def build_q(raw_global, raw_local, mu_global, std_global, mu_local,
            std_local, lambda_global):
    z_global = (raw_global - mu_global) / (std_global + EPS)
    z_local = (raw_local - mu_local) / (std_local + EPS)
    q = lambda_global * sigmoid(z_global) + (1.0 - lambda_global) * sigmoid(z_local)
    q = q.astype(np.float32)
    if not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 1):
        raise ValueError('qz_combined must be finite and in [0,1]')
    return q


def main():
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    cache_file = Path(args.clip_cache) / 'ViT-B-16.pt'
    if not cache_file.is_file():
        raise FileNotFoundError(f'Frozen CLIP weights missing: {cache_file}')
    import clip

    model, preprocess = clip.load(args.clip_model, device=args.device,
                                  download_root=args.clip_cache)
    model.eval().requires_grad_(False)
    image_ids = split_image_ids(args.dataset)
    names = datasets.get_category_list({'dataset': args.dataset})
    if len(names) != datasets.get_metadata(args.dataset)['num_classes']:
        raise ValueError('COCO category count mismatch')
    image_root = datasets.get_metadata(args.dataset)['path_to_images']
    with torch.inference_mode():
        tokens = clip.tokenize([PROMPT.format(class_name=name) for name in names]).to(args.device)
        text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
        raw = {}
        for phase in ('train', 'val', 'test'):
            raw[phase] = extract_split(image_ids[phase], image_root, preprocess,
                                       args.local_crop_ratio, model, text_features,
                                       args.clip_batch_size, args.device, phase)
    train_global, train_local = raw['train']
    mu_global = train_global.mean(axis=0, dtype=np.float64)
    std_global = np.maximum(train_global.std(axis=0, dtype=np.float64), EPS)
    mu_local = train_local.mean(axis=0, dtype=np.float64)
    std_local = np.maximum(train_local.std(axis=0, dtype=np.float64), EPS)
    q = {phase: build_q(*raw[phase], mu_global, std_global, mu_local,
                        std_local, args.lambda_global)
         for phase in ('train', 'val', 'test')}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, q_train=q['train'], q_val=q['val'], q_test=q['test'],
                        train_image_ids=image_ids['train'], val_image_ids=image_ids['val'],
                        test_image_ids=image_ids['test'], mu_global=mu_global,
                        std_global=std_global, mu_local=mu_local, std_local=std_local)
    metadata = dict(dataset=args.dataset, clip_model=args.clip_model,
                    prompt_template=PROMPT, local_crop_ratio=args.local_crop_ratio,
                    num_local_crops=5, lambda_global=args.lambda_global, eps=EPS,
                    split_seed=SPLIT_SEED, ss_seed=SS_SEED, val_frac=VAL_FRAC,
                    ss_frac_train=SS_FRAC_TRAIN, ss_frac_val=SS_FRAC_VAL,
                    num_classes=len(names), train_count=len(image_ids['train']),
                    val_count=len(image_ids['val']), test_count=len(image_ids['test']),
                    score_source='qz_combined')
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n',
                                            encoding='utf-8')
    print(f'Saved {output} and {output.with_suffix(".json")}')


if __name__ == '__main__':
    main()
