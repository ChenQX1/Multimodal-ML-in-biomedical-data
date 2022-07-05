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

        self.mode = self.cfg_data['mode']
        if self.mode == 'train':
            self.img_modal = cfgs.train_cfg.PENetCfg()
            self.ehr_modal = cfgs.train_cfg.EHRCfg()
        elif self.mode == 'test':
            self.img_modal = cfgs.test_cfg.PENetCfg()
            self.ehr_modal = cfgs.test_cfg.EHRCfg()
        else:
            raise ValueError('The experiment mode must be "train" or "test" .')

        self.name = self.cfg_data['name']
        self.save_dir = self.cfg_data['log']['save_dir']
        self.train_img = self.cfg_data['train_img']
        self.train_ehr = self.cfg_data['train_ehr']
        self.joint_training = self.cfg_data['joint_training']
        self.num_epochs = self.cfg_data['num_epochs']

        self.img_modal = self._parse_img_modal_cfg(self.img_modal)
        self.ehr_modal = self._parse_ehr_modal_cfg(self.ehr_modal)
        self._save_cfgs()

    def _save_cfgs(self):
        date_string = datetime.datetime.now() .strftime("%Y%m%d_%H%M%S")
        save_dir = '/'.join([self.save_dir, f'{self.name}_{date_string}'])
        os.makedirs(save_dir, exist_ok=True)
        with open('/'.join([save_dir, 'args.json']), 'w') as fd:
            json.dump(vars(self.img_modal), fd, indent=4, sort_keys=True)
            fd.write('\n')
            json.dump(vars(self.ehr_modal), fd, indent=4, sort_keys=True)
    

    def _parse_img_modal_cfg(self, args):
        args.rand_seed = self.cfg_data['rand_seed']
        args.gpu_ids = self.cfg_data['gpu_ids']
        args.cudnn_benchmark = self.cfg_data['cudnn_benchmark']
        [setattr(args, k, v)
         for k, v in self.cfg_data['multimodal']['image'].items()]
        args.name = self.cfg_data['name']
        args.data_dir = self.cfg_data['dataset']['data_dir']
        args.save_dir = self.cfg_data['log']['save_dir']

        args.start_epoch = 1  # Gets updated if we load a checkpoint
        if not args.is_training and not args.ckpt_path and not (hasattr(args, 'test_2d') and args.test_2d):
            raise ValueError('Must specify --ckpt_path in test mode.')
        if args.is_training and args.epochs_per_save % args.epochs_per_eval != 0:
            raise ValueError(
                'epochs_per_save must be divisible by epochs_per_eval.')
        if args.is_training:
            args.maximize_metric = not args.best_ckpt_metric.endswith('loss')
            if args.lr_scheduler == 'multi_step':
                args.lr_milestones = util.args_to_list(
                    args.lr_milestones, allow_empty=False)
        if not args.pkl_path:
            args.pkl_path = os.path.join(args.data_dir, 'series_list.pkl')

        if args.gpu_ids == -1:
            args.gpu_ids = []
        elif isinstance(args.gpu_ids, int):
            args.gpu_ids = [args.gpu_ids]
        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda'
            cudnn.benchmark = args.cudnn_benchmark
        else:
            args.device = 'cpu'
            # args.device = 'mps'

        # Set random seed for a deterministic run
        if args.rand_seed:
            torch.manual_seed(args.rand_seed)
            np.random.seed(args.rand_seed)
            random.seed(args.rand_seed)
            cudnn.deterministic = True

        # Map dataset name to a class
        if args.dataset == 'kinetics':
            args.dataset = 'KineticsDataset'
        elif args.dataset == 'pe':
            args.dataset = 'CTPEDataset3d'

        if args.is_training and args.use_pretrained:
            if args.model != 'PENet' and args.model != 'PENetClassifier' and args.model != 'PEElasticNet':
                raise ValueError(
                    'Pre-training only supported for PENet/PENetClassifier loading PENetClassifier.')
            if not args.ckpt_path:
                raise ValueError(
                    'Must specify a checkpoint path for pre-trained model.')

        args.data_loader = 'CTDataLoader'
        if args.model == 'PENet':
            if args.model_depth != 50:
                raise ValueError(
                    'Invalid model depth for PENet: {}'.format(args.model_depth))
            args.loader = 'window'
        elif args.model == 'PENetClassifier' or args.model == 'PEElasticNet':
            if args.model_depth != 50:
                raise ValueError(
                    'Invalid model depth for PENet: {}'.format(args.model_depth))
            args.loader = 'window'
            if args.dataset == 'KineticsDataset':
                args.data_loader = 'KineticsDataLoader'

        # Set up output dir (test mode only)
        if not args.is_training:
            args.results_dir = os.path.join(
                args.results_dir, '{}_{}'.format(args.name, date_string))
            os.makedirs(args.results_dir, exist_ok=True)

        return args

    def _parse_ehr_modal_cfg(self, args):
        args.rand_seed = self.cfg_data['rand_seed']
        args.gpu_ids = self.cfg_data['gpu_ids']
        args.cudnn_benchmark = self.cfg_data['cudnn_benchmark']
        [setattr(args, k, v)
         for k, v in self.cfg_data['multimodal']['EHR'].items()]
        args.name = self.cfg_data['name']
        args.data_dir = self.cfg_data['dataset']['data_dir']
        args.save_dir = self.cfg_data['log']['save_dir']

        if args.gpu_ids == -1:
            args.gpu_ids = []
        elif isinstance(args.gpu_ids, int):
            args.gpu_ids = [args.gpu_ids]

        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda'
            cudnn.benchmark = args.cudnn_benchmark
        elif args.gpu_ids == 'mps':
            args.device = 'mps'
        else:
            args.device = 'cpu'
        
        if args.rand_seed:
            torch.manual_seed(args.rand_seed)
            np.random.seed(args.rand_seed)
            random.seed(args.rand_seed)
            cudnn.deterministic = True

        return args
