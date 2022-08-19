import cfgs

import argparse
import datetime
import json
from typing import Dict
import yaml
import numpy as np
import os
import random
import torch
import util
import torch.backends.cudnn as cudnn


class CfgParser(object):
    def __init__(self, phase=None) -> None:
        self.parser = argparse.ArgumentParser('Configuration file parser.')
        self.parser.add_argument('--cfg_file', type=str, default='base_cfg')
        args = self.parser.parse_args()
        with open(f'./cfgs/{args.cfg_file}.yaml', 'r') as fd:
            self.cfg_data: Dict = yaml.safe_load(fd)
        self.phase = self.cfg_data['phase']
        self.name = self.cfg_data['name']

        self.cfgs_common = cfgs.CommonCfg(name=self.name)
        self.cfgs_ct = cfgs.CTCfg(name=self.name)
        self.cfgs_ehr = cfgs.EHRCfg(name=self.name)
        self.cfgs_joint = cfgs.CommonCfg(name=self.name)
        

        self.cfgs_common = self._parse_common_cfg(self.cfgs_common)
        self.cfgs_ct = self._parse_ct_cfg(self.cfgs_ct)
        self.cfgs_ehr = self._parse_ehr_cfg(self.cfgs_ehr)
        self.cfgs_joint = self._parse_joint_cfg(self.cfgs_joint)
        self._save_cfgs()

    def _save_cfgs(self):
        os.makedirs(self.cfgs_common.save_dir, exist_ok=True)
        with open('/'.join([self.cfgs_common.save_dir, 'args.json']), 'w') as fd:
            json.dump(vars(self.cfgs_ct), fd, indent=4, sort_keys=True)
            fd.write('\n')
            json.dump(vars(self.cfgs_ehr), fd, indent=4, sort_keys=True) 
            fd.write('\n')
            json.dump(vars(self.cfgs_joint), fd, indent=4, sort_keys=True)

    def _parse_common_cfg(self, cfgs):
        cfgs.name = self.name
        cfgs.phase = self.phase
        cfgs.is_training = True if cfgs.phase == 'train' else False
        if self.phase == 'train':
            cfgs.is_training = True
        elif self.phase == 'test':
            cfgs.is_training = False
        else:
            raise ValueError('The experiment mode must be "train" or "test" .')
        [setattr(cfgs, k, v) for k, v in self.cfg_data['common'].items()]
        date_string = datetime.datetime.now() .strftime("%Y%m%d_%H%M%S")
        cfgs.save_dir = '/'.join([cfgs.save_dir, f'{self.name}_{date_string}'])

        if cfgs.gpu_ids == -1:
            cfgs.gpu_ids = []
        elif isinstance(cfgs.gpu_ids, int):
            cfgs.gpu_ids = [cfgs.gpu_ids]
        if len(cfgs.gpu_ids) > 0 and torch.cuda.is_available():
            cfgs.device =f'cuda:{cfgs.gpu_ids[0]}'
            cudnn.benchmark = cfgs.cudnn_benchmark
        elif cfgs.gpu_ids == 'mps':
            cfgs.device = 'mps'
        else:
            cfgs.device = 'cpu'
        
        if cfgs.rand_seed:
            torch.manual_seed(cfgs.rand_seed)
            np.random.seed(cfgs.rand_seed)
            random.seed(cfgs.rand_seed)
            cudnn.deterministic = True
        
        # Set up output dir (test mode only)
        if not cfgs.is_training:
            date_string = datetime.datetime.now() .strftime("%Y%m%d_%H%M%S")
            cfgs.results_dir = os.path.join(
                cfgs.results_dir, '{}_{}'.format(cfgs.name, date_string))
            os.makedirs(cfgs.results_dir, exist_ok=True)

        return cfgs

    def _parse_ct_cfg(self, cfgs):
        [setattr(cfgs, k, v) for k, v in self.cfgs_common.__dict__.items()]
        [setattr(cfgs, k, v)
         for k, v in self.cfg_data['ct'].items()]

        cfgs.start_epoch = 1  # Gets updated if we load a checkpoint
        if not cfgs.is_training and not cfgs.ckpt_path and not (hasattr(cfgs, 'test_2d') and cfgs.test_2d):
            raise ValueError('Must specify --ckpt_path in test mode.')
        if cfgs.is_training and cfgs.epochs_per_save % cfgs.epochs_per_eval != 0:
            raise ValueError(
                'epochs_per_save must be divisible by epochs_per_eval.')

        if cfgs.is_training:
            cfgs.maximize_metric = not cfgs.best_ckpt_metric.endswith('loss')
            if cfgs.lr_scheduler == 'multi_step':
                cfgs.lr_milestones = util.args_to_list(
                    cfgs.lr_milestones, allow_empty=False)
        if not cfgs.pkl_path:
            cfgs.pkl_path = os.path.join(cfgs.data_dir, 'series_list.pkl')

        # Map dataset name to a class
        if cfgs.dataset == 'kinetics':
            cfgs.dataset = 'KineticsDataset'
        elif cfgs.dataset == 'pe':
            cfgs.dataset = 'CTPEDataset3d'

        if cfgs.is_training and cfgs.use_pretrained:
            if cfgs.model != 'PENet' and cfgs.model != 'PENetClassifier' and cfgs.model != 'PEElasticNet':
                raise ValueError(
                    'Pre-training only supported for PENet/PENetClassifier loading PENetClassifier.')
            if not cfgs.ckpt_path:
                raise ValueError(
                    'Must specify a checkpoint path for pre-trained model.')

        cfgs.data_loader = 'CTDataLoader'
        if cfgs.model == 'PENet':
            if cfgs.model_depth != 50:
                raise ValueError(
                    'Invalid model depth for PENet: {}'.format(cfgs.model_depth))
            cfgs.loader = 'window'
        elif cfgs.model == 'PENetClassifier' or cfgs.model == 'PEElasticNet':
            if cfgs.model_depth != 50:
                raise ValueError(
                    'Invalid model depth for PENet: {}'.format(cfgs.model_depth))
            cfgs.loader = 'window'
            if cfgs.dataset == 'KineticsDataset':
                cfgs.data_loader = 'KineticsDataLoader'

        return cfgs

    def _parse_ehr_cfg(self, cfgs):
        [setattr(cfgs, k, v) for k, v in self.cfgs_common.__dict__.items()]
        [setattr(cfgs, k, v)
         for k, v in self.cfg_data['ehr'].items()]

        return cfgs

    def _parse_joint_cfg(self, cfgs):
        [setattr(cfgs, k, v) for k, v in self.cfgs_common.__dict__.items()]
        [setattr(cfgs, k, v) for k, v in self.cfg_data['joint'].items()]

        return cfgs
