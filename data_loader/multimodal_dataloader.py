import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler

import datasets
from args.cfg_parser import CfgParser


class MultimodalLoader(DataLoader):
    def __init__(self, parser: CfgParser, phase: str, is_training: bool):
        assert parser.cfgs_ct.loader == 'window' or parser.cfgs_ct.loader == 'slice', \
        "Not implemented loader type: {parser.ct_modal.loader}"

        dataset_fn = datasets.__dict__['MultimodalDataset']
        self.dataset = dataset_fn(parser, phase=phase, is_training=is_training)
        self.batch_size = min(
            parser.cfgs_ct.batch_size, parser.cfgs_ehr.batch_size
        )
        self.phase = phase

        params = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'num_workers': parser.num_workers,
            'drop_last': True,
            'prefetch_factor': 2,
            'pin_memory': True
        }
        if self.phase == 'train':
            params.update(
                {'sampler': DistributedSampler(
                    dataset=self.dataset, shuffle=True, seed=parser.rand_seed)}
            )
        else:
            params.update(
                {'shuffle': is_training}
            )
        super(MultimodalLoader, self).__init__(**params)

    def get_series_label(self, series_idx):
        return self.dataset.ct_data.get_series_label(series_idx)

    def get_series(self, study_num):
        return self.dataset.ct_data.get_series(study_num)
