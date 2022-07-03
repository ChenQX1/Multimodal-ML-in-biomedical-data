import argparse
import datetime
import json
from typing import Dict
import yaml
import numpy as np
import pprint
import os
import random
import torch
import util
import torch.backends.cudnn as cudnn


class CfgParser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser('Configuration file parser.')
        self.parser.add_argument('--cfg_file', type=str, default='base_cfg')
        args = self.parser.parse_args()
        with open(f'./cfgs/{args.cfg_file}.yaml', 'r') as fd:
            self.cfg_data: Dict = yaml.safe_load(fd) 
        pprint.pprint(self.cfg_data)

    def parse(self):
        pass 
