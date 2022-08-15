import imp
from locale import normalize
import mimetypes
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler
import pickle

class EHRDataset(Dataset):
    def __init__(self, args, phase) -> None:
        super(EHRDataset, self).__init__()
        self.ehr_data, self.labels = self._load_ehr_data(args, phase)
        self.ehr_data = self.transform(self.ehr_data)

    def __len__(self):
        return len(self.ehr_data)

    def __getitem__(self, index):
        dt = self.ehr_data.loc[index, :].values
        labels = self.labels.loc[index, :].values

        return torch.from_numpy(dt), torch.from_numpy(labels)

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

        labels = ans[['label']].astype('float32')
        dt = ans.drop('label', axis=1).astype('float32')

        return dt, labels

    def transform(self, X: pd.DataFrame):
        if(self.phase == 'train'):
            drop_col = X.columns[X.nunique() == 1]
            # remove zero variance featurs
            X = X.drop(drop_col, axis=1)

            std_scaler = StandardScaler()
            # normalize
            X_imputed = std_scaler.fit_transform(X)
            X_imputed_df = pd.DataFrame(data = X_imputed, columns = X.columns, index = X.index)
            
            with open('transform_para.pkl', 'wb') as f:
                pickle.dump([drop_col, std_scaler], f)
        else:
            with open('transform_para.pkl', 'rb') as f:
                drop_col, std_scaler = pickle.load(f)
            # remove zero variance featurs
            X = X.drop(drop_col, axis=1)
            # normalize
            X_imputed = std_scaler.transform(X)
            X_imputed_df = pd.DataFrame(data = X_imputed, columns = X.columns, index = X.index)

        return X_imputed_df
