import os
import numpy as np
import pandas as pd
import shutil
import pickle


def sample_data(data_root, sample_data_dir):
    if not os.path.exists(sample_data_dir):
        os.makedirs(sample_data_dir)

    x = pd.read_csv(os.path.join(data_root, 'Demographics.csv'))
    train_idx = x[x['split'] == 'train'].sample(frac=0.05, random_state=42)['idx'].unique().tolist()
    test_idx = x[x['split'] == 'test'].sample(frac=0.1, random_state=42)['idx'].unique().tolist()

    with open('./series_list.pkl', 'rb') as fd:
        series_list = pickle.load(fd)

    sereis_list_smaple = []

    extra_val_n = 0
    for s in series_list:
        if s.study_num in (train_idx + test_idx):
            sereis_list_smaple.append(s)
        if s.phase == 'val' and (s.study_num not in (train_idx + test_idx)):
            if extra_val_n > 5:
                continue
            sereis_list_smaple.append(s)
            train_idx.append(s.study_num)
            extra_val_n += 1
            

    with open(os.path.join(sample_data_dir, 'series_list.pkl'), 'wb') as fd:
        pickle.dump(sereis_list_smaple, fd)

    np.save(os.path.join(sample_data_dir, 'train_idx.npy'), np.array(train_idx))
    np.save(os.path.join(sample_data_dir, 'test_idx.npy'), np.array(test_idx))
    
    train_idx = np.load(os.path.join(sample_data_dir, 'train_idx.npy'))
    test_idx = np.load(os.path.join(sample_data_dir, 'test_idx.npy'))

    tables = ['Demographics.csv', 'ICD.csv', 'INP_MED.csv', 'Labels.csv', 'LABS.csv', 'OUT_MED.csv']

    x = pd.read_csv(os.path.join(data_root, tables[0]))

    all_idx = x['idx'].unique()

    not_select_train = list(set(all_idx.tolist()) - set(train_idx.tolist()))
    not_select_test = list(set(all_idx.tolist()) - set(test_idx.tolist()))

    select_map_train = dict(zip(train_idx.tolist(), [True for _ in range(len(train_idx))]))
    not_select_map_train = dict(zip(not_select_train, [False for _ in range(len(not_select_train))]))

    select_map_test = dict(zip(test_idx.tolist(), [True for _ in range(len(test_idx))]))
    not_select_map_test = dict(zip(not_select_test, [False for _ in range(len(not_select_test))]))

    for tb_name in tables:
        tb = pd.read_csv(os.path.join(data_root, tb_name))
        if 'Unnamed: 0' in tb.columns:
            tb = tb.drop('Unnamed: 0', axis=1)
        tb_sample_train = tb[tb['idx'].map(select_map_train | not_select_map_train)]
        tb_sample_test = tb[tb['idx'].map(select_map_test | not_select_map_test)]
        tb_merged = pd.concat([tb_sample_train, tb_sample_test], axis=0)
        tb_merged.to_csv(os.path.join(sample_data_dir, tb_name), index=False)

    for img_name in (train_idx.tolist() + test_idx.tolist()):
        shutil.copy(
            os.path.join(data_root, f'{img_name}.npy'),
            sample_data_dir
        )





if __name__ == '__main__':
    data_root = './data'
    sample_data_dir = './data/sample'
    sample_data(data_root, sample_data_dir)
