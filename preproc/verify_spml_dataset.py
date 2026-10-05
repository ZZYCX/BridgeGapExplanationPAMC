"""Verify the Cole/Bridging-the-Gap single-positive benchmark, without training."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from datasets import generate_split, get_metadata, get_category_list

EXPECTED = {'pascal': (5717, 4574, 1143, 5823),
            'coco': (82081, 65665, 16416, 40137),
            'nuswide': (150000, 120000, 30000, 60260),
            'cub': (5994, 4795, 1199, 5794)}
SPLIT_SEED = 1200
VAL_FRAC = 0.2


@contextlib.contextmanager
def project_directory(root):
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_image_coverage(dataset, image_ids, image_root, raise_on_missing=True):
    image_root = Path(image_root)
    missing, found = [], 0
    for image_id in image_ids:
        path = image_root / str(image_id)
        if path.is_file():
            found += 1
        elif len(missing) < 10:
            missing.append(str(path))
    total = len(image_ids)
    name = 'NUS-WIDE' if dataset == 'nuswide' else dataset
    print(f'{name} image coverage:\nexpected = {total}\nfound    = {found}\nmissing  = {total - found}', flush=True)
    if total != found and raise_on_missing:
        prefix = 'NUS-WIDE Flickr images are required at data/nuswide/Flickr. ' if dataset == 'nuswide' else ''
        raise FileNotFoundError(f'{prefix}{total - found} images missing under {image_root}; examples: {missing}')
    result = dict(expected=total, found=found, missing=total - found)
    if total != found:
        result['missing_examples'] = missing
    return result


def verify_observation_seed(labels, observed, seed):
    """Mirror the existing generator, including its negative-label RNG shuffle."""
    rng = np.random.RandomState(seed)
    for row, (full, obs) in enumerate(zip(labels, observed)):
        positive = np.flatnonzero(full == 1)
        rng.shuffle(positive)
        negative = np.flatnonzero(full == 0)
        rng.shuffle(negative)  # num_neg=0 still consumes RNG in the original script
        require(np.array_equal(np.flatnonzero(obs == 1), positive[:1]),
                f'Observed labels do not match observed_seed={seed} at row {row}')


def verify_dataset(dataset, root=PROJECT_ROOT, observed_seed=1200, write_manifest=True):
    with project_directory(Path(root).resolve()):
        meta = get_metadata(dataset)
        base = Path(meta['path_to_dataset'])
        image_root = Path(meta['path_to_images'])
        arrays = {}
        for phase in ('train', 'val'):
            paths = {kind: base / f'formatted_{phase}_{kind}.npy'
                     for kind in ('images', 'labels', 'labels_obs')}
            for kind in ('images', 'labels'):
                path = paths[kind]
                if not path.is_file():
                    raise FileNotFoundError(f'Missing SPML metadata: {path}')
            ids, labels = [np.load(paths[k], allow_pickle=False) for k in ('images', 'labels')]
            observed = np.load(paths['labels_obs'], allow_pickle=False) if paths['labels_obs'].is_file() else None
            require(ids.ndim == 1, f'{phase}: image IDs must be one dimensional')
            require(labels.shape == (len(ids), meta['num_classes']),
                    f'{phase}: image/label length or category count mismatch: {labels.shape}')
            require(np.isin(labels, [0, 1]).all(), f'{phase}: full labels must be 0/1')
            arrays[phase] = (ids, labels, observed)

        train_ids, labels, observed = arrays['train']
        train_idx, val_idx = generate_split(len(train_ids), VAL_FRAC,
                                           np.random.RandomState(SPLIT_SEED))
        counts = (len(train_ids), len(train_idx), len(val_idx), len(arrays['val'][0]))
        require(counts == EXPECTED[dataset],
                f'{dataset}: benchmark size mismatch: {counts} != {EXPECTED[dataset]}')
        print(f'{dataset} split sizes: source_train={counts[0]}, train={counts[1]}, '
              f'val={counts[2]}, test={counts[3]}', flush=True)
        zero_count = int((labels.sum(axis=1) == 0).sum())
        if dataset != 'nuswide':
            require(zero_count == 0,
                    f'Each source training image needs at least one true positive; '
                    f'{dataset} has {zero_count} zero-positive source training images.')
        for phase, (_, labels, observed) in arrays.items():
            if observed is None:
                raise FileNotFoundError(f'Missing SPML metadata: {base}/formatted_{phase}_labels_obs.npy')
            require(observed.shape == labels.shape, f'{phase}: observed label shape mismatch')
            require(np.isin(observed, [0, 1]).all(), f'{phase}: observed labels must be 0/1')
            require((observed <= labels).all(), f'{phase}: observed positive is not a true positive')
            if dataset == 'nuswide':
                true_count, obs_count = labels.sum(axis=1), observed.sum(axis=1)
                require((obs_count[true_count > 0] == 1).all(),
                        f'{phase}: positive NUS-WIDE images need exactly one observed positive')
                require((obs_count[true_count == 0] == 0).all(),
                        f'{phase}: all-zero NUS-WIDE images must have zero observed positives')
            else:
                require((labels.sum(axis=1) >= 1).all(),
                        f'{phase}: every {dataset} image needs at least one true positive')
                require((observed.sum(axis=1) == 1).all(),
                        f'{phase}: every {dataset} image needs exactly one observed positive')
            verify_observation_seed(labels, observed, observed_seed)

        train_ids, labels, observed = arrays['train']
        if dataset != 'nuswide':
            require((observed[train_idx].sum(axis=1) == 1).all(),
                    'Final training split must have exactly one observed positive per image')
        coverage = verify_image_coverage(dataset,
                                        np.concatenate([arrays[p][0] for p in ('train', 'val')]),
                                        image_root, raise_on_missing=False)
        category_names = get_category_list({'dataset': dataset})
        true_count = labels[train_idx].sum(axis=0).astype(int)
        obs_count = observed[train_idx].sum(axis=0).astype(int)
        zero = np.flatnonzero(obs_count == 0).tolist()
        manifest = dict(dataset=dataset, num_classes=meta['num_classes'],
                        source_train_count=counts[0], train_count=counts[1],
                        val_count=counts[2], test_count=counts[3],
                        split_seed=SPLIT_SEED, observed_seed=observed_seed, val_frac=VAL_FRAC,
                        zero_observed_positive_classes=zero,
                        class_id_convention='0-based label columns',
                        category_names=category_names,
                        true_positive_count=true_count.tolist(),
                        observed_positive_count=obs_count.tolist(),
                        source_zero_positive_images=zero_count,
                        final_train_zero_positive_images=int((labels[train_idx].sum(axis=1) == 0).sum()),
                        final_val_zero_positive_images=int((labels[val_idx].sum(axis=1) == 0).sum()),
                        test_zero_positive_images=int((arrays['val'][1].sum(axis=1) == 0).sum()),
                        zero_positive_ratio=zero_count / counts[0],
                        zero_positive_policy='keep full/observed all-zero rows' if dataset == 'nuswide' else 'require true positive',
                        observation_seed_verified=True,
                        image_coverage=coverage,
                        validation_status='passed' if coverage['missing'] == 0 else 'failed_image_coverage',
                        ready_for_training_and_cache=coverage['missing'] == 0,
                        validation_test_labels='full; supplied by datasets.get_data')
        print(json.dumps(manifest, indent=2))
        if write_manifest:
            path = base / 'spml_manifest.json'
            path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
            print(f'Saved {path.resolve()}')
        if coverage['missing']:
            prefix = 'NUS-WIDE Flickr images are required at data/nuswide/Flickr. ' if dataset == 'nuswide' else ''
            raise FileNotFoundError(f'{prefix}{coverage["missing"]} images missing under {image_root}; '
                                    f'examples: {coverage["missing_examples"]}')
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(EXPECTED), required=True)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--observed-seed', type=int, default=1200)
    args = parser.parse_args()
    try:
        verify_dataset(args.dataset, args.root, args.observed_seed)
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        parser.exit(1, f'SPML verification failed: {exc}\n')


if __name__ == '__main__':
    main()
