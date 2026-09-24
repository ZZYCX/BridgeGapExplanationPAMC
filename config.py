import os
import sys
import argparse
from munch import Munch as mch
from os.path import join as ospj
from datetime import datetime

#项目运行指令：python main.py --dataset coco --largelossmod_scheme LL-R
_DATASET = ('pascal', 'coco', 'nuswide', 'cub', 'openimages')
_SCHEMES = ('LL-R', 'LL-Ct', 'LL-Cp')
_LOOKUP = {
    'feat_dim': {
        'resnet50': 2048,
        'resnet101' : 2048,
    },
    'num_classes': {
        'pascal': 20,
        'coco': 80,
        'nuswide': 81,
        'cub': 312,
        'openimages': 5000,
    },
    'delta_rel': {
        'LL-R': 0.5,
        'LL-Ct': 0.2,
        'LL-Cp': 0.1,
    }
}

def set_dir(runs_dir, exp_name):
    runs_dir = ospj(runs_dir, exp_name)
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    return runs_dir

def set_default_configs(args):
    args.ss_seed = 999
    args.ss_frac_train = 1.0
    args.ss_frac_val = 1.0
    args.use_feats = False
    args.val_frac = 0.2
    args.split_seed = 1200
    args.train_set_variant = 'observed'
    args.val_set_variant = 'clean'
    args.arch = 'resnet50'
    args.freeze_feature_extractor = False
    args.use_pretrained = True
    args.num_workers = 4
    args.lr_mult = 10
    args.save_path = './results'

    return args

def set_follow_up_configs(args):
    args.feat_dim = _LOOKUP['feat_dim'][args.arch]
    args.num_classes = _LOOKUP['num_classes'][args.dataset]
    args.delta_rel = _LOOKUP['delta_rel'][args.largelossmod_scheme]
    now = datetime.now()
    args.experiment_name = str(now).split(".")[0].replace('-','').replace(" ","_").replace(":","")
    args.save_path = set_dir(args.save_path, args.experiment_name)

    if args.delta_rel != 0:
        args.delta_rel /= 100
    args.clean_rate = 1

    return args


def get_configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True,
                        choices=_DATASET)
    parser.add_argument('--largelossmod_scheme', type=str, required=True, 
                        choices=_SCHEMES)
                        
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Global random seed for reproducible training.'
    )
    parser.add_argument('--gpu_num', type=str, default='0')
    parser.add_argument('--bsize', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=5)
    parser.add_argument('--semantic_adaptive_boostlu', action='store_true')
    parser.add_argument('--semantic_delta', type=float, default=5.0)
    #实际正式训练中--semantic_lambda_global 控制 global 分量的权重 A；local 分量的权重是 1-A。
    # train.py 从新缓存读取两个分量后，按 q = A·q_global + (1-A)·q_local 构造传给模型的 semantic_q
    parser.add_argument('--semantic_lambda_global', type=float, default=0.5)
    parser.add_argument('--semantic_score_file', type=str, default=None)
    args = parser.parse_args()
    if not 0.0 <= args.semantic_lambda_global <= 1.0:
        parser.error('--semantic_lambda_global must be in [0,1]')
    if args.semantic_adaptive_boostlu and args.semantic_score_file is None:
        parser.error('--semantic_score_file is required with --semantic_adaptive_boostlu')
    args = set_default_configs(args)
    args = set_follow_up_configs(args)
    args = mch(**vars(args))

    return args


