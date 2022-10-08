from collections import defaultdict
import sys
import os
import hydra
from omegaconf import DictConfig, open_dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from args.cfg_parser import ConfigParser
import datasets
import sklearn.metrics as sk_metrics
from saver.model_saver import ModelSaver
import util
from PIL import Image


@hydra.main(config_path='config', config_name='ehr_tabnet', version_base='1.2')
def test(cfgs: DictConfig):
    with open_dict(cfgs):
        cfgs.common.is_trianing = False
        cfgs.common.phase = 'test'
    cfgs = ConfigParser(cfgs).cfgs

    device = torch.device(cfgs.common.device)
    data_test = datasets.__dict__[cfgs.dataset.dataset](cfgs.dataset, phase='test')
    dataloader_test= DataLoader(
        data_test, 1, sampler=data_test.ehr_data.index.values, drop_last=False)
    model, ckpt_info = ModelSaver.load_model(cfgs.model.ckpt_path)
    cfgs.model.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(device)
    model.eval()

    preds = []
    labels = []
    with tqdm(total=len(dataloader_test.dataset), unit='sample') as bar:
        with torch.no_grad():
            for j, (ehr_input, target) in enumerate(dataloader_test):
                cls_logits, _ = model(ehr_input.to(device))
                cls_prob = torch.sigmoid(cls_logits).cpu().numpy()
                preds.append(cls_prob.item())
                labels.append(target.item())
    labels = np.array(labels)
    preds = np.array(preds)
    metrics = {
        cfgs.common.phase + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, preds),
        cfgs.common.phase + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, preds),
    }
    curves = {
        cfgs.common.phase + '_' + 'PRC': sk_metrics.precision_recall_curve(labels, preds),
        cfgs.common.phase + '_' + 'ROC': sk_metrics.roc_curve(labels, preds)
    }

    with open(os.path.join(cfgs.common.results_dir, 'metrics.txt'), 'w') as fd:
        for k,v in metrics.items():
            fd.write('{}: {:.5f}\n'.format(k, v))

    for name, curve in curves.items():
        curve_np = util.get_plot(name, curve)
        curve_img = Image.fromarray(curve_np)
        curve_img.save(os.path.join(
            cfgs.common.results_dir, '{}.png'.format(name)))


if __name__ == '__main__':
    test()
