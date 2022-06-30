from ntpath import join
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class EHRDataset(Dataset):
    def __init__(self, args, phase) -> None:
        super(EHRDataset, self).__init__()
        self.ehr_data = pd.read_csv(
            '/'.join([args.data_dir, 'Demographics.csv'])
        )
        if 'Unnamed: 0' in self.ehr_data.columns:
            self.ehr_data.drop('Unnamed: 0', axis=1, inplace=True)
        self.ehr_data = self.ehr_data[self.ehr_data.split == phase].set_index('idx').drop('split', axis=1)
        self.labels = pd.read_csv('/'.join([args.data_dir, 'Labels.csv']))
        self.labels = self.labels[self.labels.split == phase].set_index('idx')[['label']]

    def __len__(self):
        assert len(self.ehr_data) == len(self.labels), 'Different lengths!'
        return len(self.ehr_data)

    def __getitem__(self, index):
        dt = self.ehr_data.loc[index, :].values
        label = self.labels.loc[index, :].values

        return torch.tensor(dt, dtype=torch.float), torch.tensor(label, dtype=torch.float)

