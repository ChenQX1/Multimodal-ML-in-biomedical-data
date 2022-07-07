import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class EHRDataset(Dataset):
    def __init__(self, args, phase) -> None:
        super(EHRDataset, self).__init__()
        self.ehr_data, self.labels = self._load_ehr_data(args, phase)

    def __len__(self):
        return len(self.ehr_data)

    def __getitem__(self, index):
        dt = self.ehr_data.loc[index, :].values
        labels = self.labels.loc[index, :].values

        return torch.tensor(dt, dtype=torch.float), torch.tensor(labels, dtype=torch.float)

    def _load_ehr_data(self, args, phase):
        tb_ls = []

        if args.dataset == 'all_ehr':
            dataset = []
            for tb_ in os.listdir(args.data_dir):
                if tb_.endswith('.csv'):
                    dataset.append(tb_)
            args.dataset = dataset
        
        for tb_ in args.dataset:
            tmp_tb = pd.read_csv('/'.join([args.data_dir, tb_])).groupby('idx').last()
            if 'Unnamed: 0' in tmp_tb.columns:
                tmp_tb.drop('Unnamed: 0', axis=1, inplace=True)
            tmp_tb = tmp_tb[tmp_tb.split == phase].drop(['split'], axis=1)
            tb_ls.append(tmp_tb)

        ans = pd.concat(tb_ls, axis=1)
        ans = ans.loc[:, ~ans.columns.duplicated()].drop(['pe_type'], axis=1)

        labels = ans[['label']]
        dt = ans.drop('label', axis=1)

        return dt, labels
