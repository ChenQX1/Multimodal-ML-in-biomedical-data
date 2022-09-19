import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler

import datasets
from datasets import RadfusionDataset
from omegaconf import DictConfig


class RadfusionDataLoader(DataLoader):
    def __init__(self, cfgs: DictConfig, phase: str, is_training: bool):
        assert cfgs.ct.loader == 'window' or cfgs.ct.loader == 'slice', \
        "Not implemented loader type: {parser.ct_modal.loader}"

        dataset: RadfusionDataset = RadfusionDataset(cfgs, phase=phase, is_training=is_training)

        params = {
            'dataset': dataset,
            'batch_size': cfgs.batch_size,
            'num_workers': cfgs.num_workers,
            'drop_last': True,
            'prefetch_factor': 4,
            'pin_memory': True,
            'shuffle': is_training
        }
        super(RadfusionDataLoader, self).__init__(**params)

    def get_series_label(self, series_idx):
        return self.dataset.ct_data.get_series_label(series_idx)

    def get_series(self, study_num):
        return self.dataset.ct_data.get_series(study_num)
