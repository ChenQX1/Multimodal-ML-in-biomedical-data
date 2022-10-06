import os
import sys
sys.path.insert(0, os.getcwd())
import util
from data_loader import RadfusionDataLoader
from saver.model_saver import ModelSaver
from models import FusionModel
from logger.train_logger import TrainLogger
from args.cfg_parser import ConfigParser
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

from evaluator.model_evaluator import ModelEvaluator
from omegaconf import DictConfig, open_dict
import hydra


@hydra.main(config_path='config', config_name='radfusion', version_base='1.2')
def train(cfgs: DictConfig):
    with open_dict(cfgs):
        cfgs.common.is_training = True
        cfgs.common.phase = 'train'
    cfgs: DictConfig = ConfigParser(cfgs).cfgs

    local_rank = 0
    device = torch.device(cfgs.common.device)

    dataloader_train = RadfusionDataLoader(
        cfgs.dataset, phase='train', is_training=True)
    dataloader_val = RadfusionDataLoader(
        cfgs.dataset, phase='val', is_training=False)

    if cfgs.model.ckpt_path and cfgs.model.use_pretrained:
        model, ckpt_info = ModelSaver.load_model(
            cfgs.model.ckpt_path)
        with open_dict(cfgs):
            cfgs.common.start_epoch = ckpt_info['epoch'] + 1
    else:
        subnet_ct, _ = ModelSaver.load_model(cfgs.model.ct.ckpt_path)
        subnet_ehr, _ = ModelSaver.load_model(cfgs.model.ehr.ckpt_path)
        shim_ct = nn.AdaptiveAvgPool3d(1)
        shim_ehr = nn.Linear(
            dataloader_train.dataset.ehr_data.ehr_data.shape[1], cfgs.model.ehr.feat_size)
        classifier_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(cfgs.model.ct.feat_size +
                      cfgs.model.ehr.feat_size, cfgs.common.num_classes)
        )
        model = FusionModel(
            subnet_ct,
            subnet_ehr,
            shim_ct,
            shim_ehr,
            classifier_head
        )
    model = model.to(device)
    print(dataloader_train.dataset.ehr_data.ehr_data.shape)

    optimizer = util.get_optimizer(model.parameters(), args=cfgs.optimizer)
    # lr_schduler = util.get_scheduler(optimizer, parser)
    criterion = util.get_loss_fn(
        is_classification=True, dataset=cfgs.dataset.dataset, size_average=False)
    logger = TrainLogger(
        cfgs.common, len(
            dataloader_train.dataset.ehr_data), dataloader_train.dataset.ct_data.pixel_dict
    )
    evaluator = ModelEvaluator(
        cfgs.model.ct.do_classify, cfgs.dataset.dataset, dataloader_val, logger,
        cfgs.model.ct.agg_method, cfgs.common.num_visuals, cfgs.model.ct.max_eval, cfgs.common.epochs_per_eval, do_fusion=True
    )
    saver = ModelSaver(cfgs.common.save_dir, cfgs.common.epochs_per_save,
                       cfgs.common.max_ckpts, cfgs.common.best_ckpt_metric, cfgs.common.maximize_metric)
    # ----------- Traing Loop ----------
    print(f'Lenght of dataset: {len(dataloader_train.dataset)}')
    print(f'Number of iterations: {len(dataloader_train)}')
    loss_train = []
    for i in range(cfgs.common.num_epochs):
        logger.start_epoch()
        model.train()
        loss_epoch = []
        t = tqdm(total=len(dataloader_train))
        for iter_n, (ct_input, ehr_input, target_dict) in enumerate(dataloader_train):
            logger.start_iter()
            target = target_dict['is_abnormal']
            cls_logits = model(ct_input.to(device), ehr_input.to(device))
            loss = criterion(cls_logits, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.detach().cpu().numpy())
            if iter_n != 0 and iter_n % 100 == 0:
                print(
                    f'    ---------------- iter: {iter_n} training loss: {loss}')
            logger.log_iter(
                ct_input, cls_logits, target_dict, loss, optimizer
            )
            t.update(1)
            logger.end_iter()
        t.close()

        metrics, curves = evaluator.evaluate(model, device, logger.epoch)
        loss_train.append(np.mean(loss_epoch))
        print(f'    epoch {i} training loss:    {loss_train[-1]}')
        if local_rank == 0:
            saver.save(
                i, model, optimizer, None, device, metric_val=metrics.get(cfgs.common.best_ckpt_metric, None)
            )
        logger.end_epoch(metrics, curves)


@hydra.main(config_path='config', config_name='base', version_base='1.2')
def test(cfgs: DictConfig):
    with open_dict():
        cfgs.common.is_training = False
        cfgs.common.phase = 'test'
    cfgs = ConfigParser(cfgs).cfgs

    device = torch.device(cfgs.common.device)
    model: nn.Module
    model, ckpt_info = ModelSaver.load_model(cfgs.model.ckpt_path)
    cfgs.model.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(device)
    model.eval()
    
    dataloader_test = RadfusionDataLoader(cfgs.dataset, phase='test', is_training=False)
    study2slices = defaultdict(list)
    study2probs = defaultdict(list)
    study2labels = {}
    # logger = TestLogger(cfgs_joint, len(dataloader_test.dataset),
    #                     dataloader_test.dataset.ct_data.pixel_dict)
    util.print_err(f'Writing model outputs to {cfgs.common.resutls_dir}')
    with tqdm(total=len(dataloader_test.dataset), unit='windows') as progress_bar:
        ct_input: torch.Tensor
        ehr_input: torch.Tensor
        targets_dict: Dict
        for j, (ct_input, ehr_input, targets_dict) in enumerate(dataloader_test):
            target: torch.Tensor = targets_dict['is_abnormal']
            with torch.no_grad():
                cls_logits = model(ct_input.to(device), ehr_input.to(device))
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
    train()
    # test()