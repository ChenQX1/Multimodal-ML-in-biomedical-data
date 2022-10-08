from omegaconf import DictConfig
from torch.utils.data import Dataset

import datasets
from datasets import CTPEDataset3d, EHRDataset


class RadfusionDataset(Dataset):
    def __init__(self, cfgs: DictConfig , phase: str, is_training: bool):
        self.cfgs_ct = cfgs.ct
        self.cfgs_ehr = cfgs.ehr
        self.ct_data: CTPEDataset3d = self._get_data(self.cfgs_ct, phase=phase, is_training_set=is_training)
        self.ehr_data: EHRDataset = self._get_data(self.cfgs_ehr, phase=phase)
        self.phase = phase

    def __getitem__(self, index):
        ct_volume, ct_dict = self.ct_data[index]
        study_num = ct_dict['study_num']
        ehr_feat, ehr_label = self.ehr_data[study_num]

        return ct_volume, ehr_feat, ct_dict

    def __len__(self):
        return len(self.ct_data.window_to_series_idx)

    def _get_data(self, args, **kwargs):
        dataset_fn = datasets.__dict__[args.dataset]
        return dataset_fn(args, **kwargs)
