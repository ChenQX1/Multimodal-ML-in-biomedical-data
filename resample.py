import os
import numpy as np
import pandas as pd
import shutil
import pickle


def sample_data(data_root, sample_data_dir):
    if not os.path.exists(sample_data_dir):
        os.makedirs(sample_data_dir)

    x = pd.read_csv(os.path.join(data_root, 'Demographics.csv'))
    train_idx = x[x['split'] == 'train'].sample(n=10, random_state=42)['idx'].unique().tolist()
    test_idx = x[x['split'] == 'test'].sample(n=5, random_state=42)['idx'].unique().tolist()
    val_idx = x[x['split'] == 'val'].sample(n=5, random_state=42)['idx'].unique().tolist()

    with open('./series_list.pkl', 'rb') as fd:
        series_list = pickle.load(fd)
    sereis_list_smaple = []
    for s in series_list:
        if s.study_num in (train_idx + test_idx + val_idx):
            sereis_list_smaple.append(s)
    with open(os.path.join(sample_data_dir, 'series_list.pkl'), 'wb') as fd:
        pickle.dump(sereis_list_smaple, fd)

    tables = ['Demographics.csv', 'ICD.csv', 'INP_MED.csv', 'Labels.csv', 'LABS.csv', 'OUT_MED.csv']
    idx2map = dict(zip(
        x['idx'].values,
        x['idx'].apply(lambda t: True if t in train_idx + val_idx + test_idx else False).values
    ))

    for tb_name in tables:
        tb = pd.read_csv(os.path.join(data_root, tb_name))
        if 'Unnamed: 0' in tb.columns:
            tb = tb.drop('Unnamed: 0', axis=1)
        tb_sample = tb[tb['idx'].map(idx2map)]
        tb_sample.to_csv('/'.join([sample_data_dir, tb_name]), index=False)

    for img_name in (train_idx + test_idx + val_idx):
        shutil.copy(
            os.path.join(data_root, f'{img_name}.npy'),
            sample_data_dir
        )
    np.save(os.path.join(sample_data_dir, 'train_idx.npy'), np.array(train_idx))
    np.save(os.path.join(sample_data_dir, 'test_idx.npy'), np.array(test_idx))


if __name__ == '__main__':
    data_root = './data'
    sample_data_dir = './data/sample'
    sample_data(data_root, sample_data_dir)
