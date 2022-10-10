from omegaconf import DictConfig, open_dict, OmegaConf
import datetime
from typing import Dict
import numpy as np
import os
import random
import torch
import torch.backends.cudnn as cudnn


class ConfigParser(object):
    def __init__(self, cfgs: DictConfig) -> None:
        cfgs.common = self._parse_common_cfg(cfgs.common, cfgs.name, cfgs.fusion)
        cfgs.dataset = self._merge_common(cfgs.common, cfgs.dataset)
        cfgs.model = self._merge_common(cfgs.common, cfgs.model)
        if cfgs.fusion:
            cfgs.dataset.ct = self._merge_common(cfgs.common, cfgs.dataset.ct)
            cfgs.dataset.ehr = self._merge_common(cfgs.common, cfgs.dataset.ehr)
            cfgs.model.ct = self._merge_common(cfgs.common, cfgs.model.ct)
            cfgs.model.ehr = self._merge_common(cfgs.common, cfgs.model.ehr)
        
        self.cfgs: DictConfig = cfgs

    def _merge_common(self, from_dict: DictConfig, to_dict: DictConfig):
        with open_dict(to_dict):
            ans = OmegaConf.merge(to_dict, from_dict)

        return ans
    
    def _parse_common_cfg(self, cfgs: DictConfig, task_name: str, fusion: str = None):
        with open_dict(cfgs):
            cfgs.name = task_name
            cfgs.fusion = fusion
            # Time stamp
            date_string = datetime.datetime.now() .strftime("%Y%m%d_%H%M%S")
            cfgs.date_string = date_string
            cfgs.save_dir = '/'.join([cfgs.save_dir, f'{task_name}_{date_string}'])
            
            # device
            if cfgs.gpu_ids == -1:
                cfgs.gpu_ids = []
            elif isinstance(cfgs.gpu_ids, int):
                cfgs.gpu_ids = [cfgs.gpu_ids] 
            if len(cfgs.gpu_ids) > 0 and torch.cuda.is_available():
                cfgs.device =f'cuda:{cfgs.gpu_ids[0]}'
                cudnn.benchmark = cfgs.cudnn_benchmark
            elif len(cfgs.gpu_ids) > 0 and torch.backends.mps.is_available():
                cfgs.device = 'mps'
            else:
                cfgs.device = 'cpu' 
            # random seeds
            if cfgs.rand_seed:
                torch.manual_seed(cfgs.rand_seed)
                np.random.seed(cfgs.rand_seed)
                random.seed(cfgs.rand_seed)
                cudnn.deterministic = True

            cfgs.results_dir = os.path.join(
                cfgs.results_dir, '{}_{}'.format(task_name, date_string))
            if cfgs.phase == 'test':
                os.makedirs(cfgs.results_dir, exist_ok=True)
            else:
                os.makedirs(cfgs.save_dir, exist_ok=True)

            cfgs.maximize_metric = not cfgs.best_ckpt_metric.endswith('loss')

        return cfgs
