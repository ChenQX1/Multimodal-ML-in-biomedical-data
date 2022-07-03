from imageio import save
import cfgs

import argparse
import datetime
import json
from typing import Dict
import yaml
import numpy as np
from pprint import pprint as print
import os
import random
import torch
import util
import torch.backends.cudnn as cudnn


# TODO:
# two models: one for img classification, one for EHR classification
# two mode: training and test
class CfgParser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser('Configuration file parser.')
        self.parser.add_argument('--cfg_file', type=str, default='base_cfg')
        args = self.parser.parse_args()
        with open(f'./cfgs/{args.cfg_file}.yaml', 'r') as fd:
            self.cfg_data: Dict = yaml.safe_load(fd)
        # Set cfgs for image modal
        mode = self.cfg_data['mode']
        if mode == 'train':
            self.img_modal = cfgs.train_cfg.PENetCfg()
        elif mode == 'test':
            self.img_modal = cfgs.test_cfg.PENetCfg()
        else:
            raise ValueError('The experiment mode must be "train" or "test" .')
        
        self.name = self.cfg_data['name']

        self.img_modal = self._parse_img_modal_cfg(self.img_modal)

        # TODO: add cfg for elastic net
        # self.elastic_net_cfg = None

    # TODO: rename args to 'self.img_modal'
    def _parse_img_modal_cfg(self, args):
        [setattr(self.img_modal, k, v) for k, v in self.cfg_data['multimodal']['image'].items()]
        self.img_modal.name = self.cfg_data['name']
        self.img_modal.data_dir = self.cfg_data['dataset']['data_dir']
        self.img_modal.save_dir = self.cfg_data['log']['save_dir']

        date_string = datetime.datetime.now() .strftime("%Y%m%d_%H%M%S")
        save_dir = '/'.join([self.img_modal.save_dir, f'{self.name}_{date_string}'])
        os.makedirs(save_dir, exist_ok=True)
        with open('/'.join([save_dir, 'args.json']), 'w') as fd:
            json.dump(vars(self.img_modal), fd, indent=4, sort_keys=True)
            fd.write('\n')
            # json.dump(vars(self.elastic_net_cfg), fd, indent=4, sort_keys=True)
            fd.write('\n')
        
        args.start_epoch = 1  # Gets updated if we load a checkpoint
        if not args.is_training and not args.ckpt_path and not (hasattr(args, 'test_2d') and args.test_2d):
            raise ValueError('Must specify --ckpt_path in test mode.')
        if args.is_training and args.epochs_per_save % args.epochs_per_eval != 0:
            raise ValueError('epochs_per_save must be divisible by epochs_per_eval.')
        if args.is_training:
            args.maximize_metric = not args.best_ckpt_metric.endswith('loss')
            if args.lr_scheduler == 'multi_step':
                args.lr_milestones = util.args_to_list(args.lr_milestones, allow_empty=False)
        if not args.pkl_path:
            args.pkl_path = os.path.join(args.data_dir, 'series_list.pkl')

        # # Set up resize and crop
        # args.resize_shape = util.args_to_list(args.resize_shape, allow_empty=False, arg_type=int, allow_negative=False)
        # args.crop_shape = util.args_to_list(args.crop_shape, allow_empty=False, arg_type=int, allow_negative=False)

        # Set up available GPUs
        # args.gpu_ids = util.args_to_list(args.gpu_ids, allow_empty=True, arg_type=int, allow_negative=False)
        if args.gpu_ids == -1: args.gpu_ids = []
        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda'
            cudnn.benchmark = args.cudnn_benchmark
        else:
            args.device = 'cpu'
            # args.device = 'mps'

        # Set random seed for a deterministic run
        if args.deterministic:
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            cudnn.deterministic = True

        # Map dataset name to a class
        if args.dataset == 'kinetics':
            args.dataset = 'KineticsDataset'
        elif args.dataset == 'pe':
            args.dataset = 'CTPEDataset3d'

        if args.is_training and args.use_pretrained:
            if args.model != 'PENet' and args.model != 'PENetClassifier' and args.model != 'PEElasticNet':
                raise ValueError('Pre-training only supported for PENet/PENetClassifier loading PENetClassifier.')
            if not args.ckpt_path:
                raise ValueError('Must specify a checkpoint path for pre-trained model.')

        args.data_loader = 'CTDataLoader'
        if args.model == 'PENet':
            if args.model_depth != 50:
                raise ValueError('Invalid model depth for PENet: {}'.format(args.model_depth))
            args.loader = 'window'
        elif args.model == 'PENetClassifier' or args.model == 'PEElasticNet':
            if args.model_depth != 50:
                raise ValueError('Invalid model depth for PENet: {}'.format(args.model_depth))
            args.loader = 'window'
            if args.dataset == 'KineticsDataset':
                args.data_loader = 'KineticsDataLoader'

        # Set up output dir (test mode only)
        if not args.is_training:
            args.results_dir = os.path.join(args.results_dir, '{}_{}'.format(args.name, date_string))
            os.makedirs(args.results_dir, exist_ok=True)
        
        return args
