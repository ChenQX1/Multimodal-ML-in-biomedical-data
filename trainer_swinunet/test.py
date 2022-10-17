import os
import sys
sys.path.insert(0, os.getcwd())
import util
from data_loader import CTDataLoader
from saver.model_saver import ModelSaver
from args import ConfigParser
from collections import defaultdict
import pickle
from time import time
from typing import Dict
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sklearn.metrics as sk_metrics
from PIL import Image

from omegaconf import DictConfig, open_dict
import hydra


@hydra.main(config_path='config', config_name='ct_swinunet', version_base='1.2')
def test(cfgs: DictConfig):
    with open_dict(cfgs):
        cfgs.common.is_training = False
        cfgs.common.phase = 'test'
    cfgs = ConfigParser(cfgs).cfgs

    device = torch.device(cfgs.common.device)
    model: nn.Module
    model, ckpt_info = ModelSaver.load_model(cfgs.model.ckpt_path)
    cfgs.model.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(device)
    model.eval()

    dataloader_test = CTDataLoader(cfgs.dataset, phase='test', is_training=False)
    study2slices = defaultdict(list)
    study2probs = defaultdict(list)
    study2labels = {}
    # logger = TestLogger(cfgs_joint, len(dataloader_test.dataset),
    #                     dataloader_test.dataset.ct_data.pixel_dict)
    util.print_err(f'Writing model outputs to {cfgs.common.results_dir}')
    with tqdm(total=len(dataloader_test.dataset), unit='windows') as progress_bar:
        ct_input: torch.Tensor
        targets_dict: Dict
        for j, (ct_input,  targets_dict) in enumerate(dataloader_test):
            target: torch.Tensor = targets_dict['is_abnormal']
            with torch.no_grad():
                cls_logits = model(ct_input.squeeze().to(device))
                cls_probs = torch.sigmoid(cls_logits)
                max_probs = cls_probs.to('cpu').numpy()
                for study_num, slice_idx, prob in zip(
                    targets_dict['study_num'], targets_dict['slice_idx'], list(max_probs)):
                    study_num = int(study_num) 
                    slice_idx = int(slice_idx)
                    study2slices[study_num].append(slice_idx)
                    study2probs[study_num].append(prob.item())

                    series = dataloader_test.get_series(study_num)
                    if study_num not in study2labels:
                        study2labels[study_num] = int(series.is_positive)
                progress_bar.update(ct_input.size(0))

    # Combine masks
    util.print_err('Combining masks...')
    max_probs = []
    labels = []
    predictions = {}
    print("Get max probability")
    for study_num in tqdm(study2slices):

        # Sort by slice index and get max probability
        slice_list, prob_list = (list(t) for t in zip(*sorted(zip(study2slices[study_num], study2probs[study_num]),
                                                              key=lambda slice_and_prob: slice_and_prob[0])))
        study2slices[study_num] = slice_list
        study2probs[study_num] = prob_list
        max_prob = max(prob_list)
        max_probs.append(max_prob)
        label = study2labels[study_num]
        labels.append(label)
        predictions[study_num] = {'label': label, 'pred': max_prob}

    # Save predictions to file, indexed by study number
    print("Save to pickle")
    with open('{}/preds.pickle'.format(cfgs.common.results_dir), "wb") as fp:
        pickle.dump(predictions, fp)

    # Write the slice indices used for the features
    print("Write slice indices")
    # with open(os.path.join(cfgs_joint.results_dir, 'xgb', 'series2slices.json'), 'w') as json_fh:
    #     json.dump(study2slices, json_fh, sort_keys=True, indent=4)

    # Compute AUROC and AUPRC using max aggregation, write to files
    max_probs, labels = np.array(max_probs), np.array(labels)
    metrics = {
        cfgs.common.phase + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, max_probs),
        cfgs.common.phase + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, max_probs),
    }
    print("Write metrics")
    with open(os.path.join(cfgs.common.results_dir, 'metrics.txt'), 'w') as metrics_fh:
        for k, v in metrics.items():
            metrics_fh.write('{}: {:.5f}\n'.format(k, v))

    curves = {
        cfgs.common.phase + '_' + 'PRC': sk_metrics.precision_recall_curve(labels, max_probs),
        cfgs.common.phase + '_' + 'ROC': sk_metrics.roc_curve(labels, max_probs)
    }
    for name, curve in curves.items():
        curve_np = util.get_plot(name, curve)
        curve_img = Image.fromarray(curve_np)
        curve_img.save(os.path.join(
            cfgs.common.results_dir, '{}.png'.format(name)))


if __name__ == "__main__":
    test()
