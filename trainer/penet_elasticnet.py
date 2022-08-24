import os
import sys
sys.path.insert(0, os.getcwd())
import util
from data_loader import MultimodalLoader
from saver.model_saver import ModelSaver
from models import FusionModel
from logger.train_logger import TrainLogger
from args.cfg_parser import CfgParser
from collections import defaultdict
import json
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


def train(parser: CfgParser):
    cfgs_ct = parser.cfgs_ct
    cfgs_ehr = parser.cfgs_ehr
    cfgs_joint = parser.cfgs_joint
    local_rank = 0
    device = torch.device(cfgs_joint.device)

    dataloader_train = MultimodalLoader(
        parser, phase='train', is_training=cfgs_joint.is_training)
    dataloader_val = MultimodalLoader(
        parser, phase='val', is_training=cfgs_joint.is_training)

    if cfgs_joint.ckpt_path and not cfgs_joint.use_pretrained:
        model, ckpt_info = ModelSaver.load_model(
            cfgs_joint.ckpt_path)
        cfgs_joint.start_epoch = ckpt_info['epoch'] + 1
    else:
        subnet_ct, _ = ModelSaver.load_model(cfgs_ct.ckpt_path)
        subnet_ehr, _ = ModelSaver.load_model(cfgs_ehr.ckpt_path)
        shim_ct = nn.AdaptiveAvgPool3d(1)
        shim_ehr = nn.Linear(
            dataloader_train.dataset.ehr_data.ehr_data.shape[1], cfgs_ehr.feat_size)
        classifier_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(cfgs_ct.feat_size +
                      cfgs_ehr.feat_size, cfgs_joint.num_classes)
        )
        model = FusionModel(
            subnet_ct,
            subnet_ehr,
            shim_ct,
            shim_ehr,
            classifier_head
        )
    model = model.to(device)

    optimizer = util.get_optimizer(model.parameters(), args=cfgs_joint)
    # lr_schduler = util.get_scheduler(optimizer, parser)
    criterion = util.get_loss_fn(
        is_classification=True, dataset=cfgs_joint.dataset, size_average=False)
    logger = TrainLogger(
        cfgs_ct, len(
            dataloader_train.dataset.ehr_data), dataloader_train.dataset.ct_data.pixel_dict
    )
    evaluator = ModelEvaluator(
        cfgs_ct.do_classify, cfgs_ct.dataset, dataloader_val, logger,
        cfgs_ct.agg_method, cfgs_ct.num_visuals, cfgs_ct.max_eval, cfgs_ct.epochs_per_eval, joint_training=True
    )
    saver = ModelSaver(cfgs_joint.save_dir, cfgs_joint.epochs_per_save,
                       cfgs_joint.max_ckpts, cfgs_joint.best_ckpt_metric, cfgs_ct.maximize_metric)
    # ----------- Traing Loop ----------
    print(f'Lenght of dataset: {len(dataloader_train.dataset)}')
    print(f'Number of iterations: {len(dataloader_train)}')
    loss_train = []
    for i in range(cfgs_joint.num_epochs):
        logger.start_epoch()
        model.train()
        loss_epoch = []
        t = tqdm(total=len(dataloader_train))
        ts = time()
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
        print(f'Time: {time() - ts} s')
        t.close()

        metrics, curves = evaluator.evaluate(model, device, logger.epoch)
        loss_train.append(np.mean(loss_epoch))
        print(f'    epoch {i} training loss:    {loss_train[-1]}')
        if local_rank == 0:
            saver.save(
                i, model, optimizer, None, device, metric_val=metrics.get(cfgs_joint.best_ckpt_metric, None)
            )
        logger.end_epoch(metrics, curves)


def test(parser: CfgParser):
    cfgs_joint = parser.cfgs_joint
    cfgs_ct = parser.cfgs_ct
    cfgs_ehr = parser.cfgs_ehr

    device = torch.device(cfgs_joint.device)
    if cfgs_joint.joint_training:
        model: nn.Module
        ckpt_info: Dict
        model, ckpt_info = ModelSaver.load_model(cfgs_joint.ckpt_path)
        cfgs_joint.start_epoch = ckpt_info['epoch'] + 1
        model = model.to(device)
        model.eval()
    else:
        model_ct: nn.Module
        ckpt_info_ct: Dict
        model_ehr: nn.Module
        ckpt_info_ehr: Dict
        model_ct, ckpt_info_ct = ModelSaver.load_model(cfgs_ct.ckpt_path)
        cfgs_ct.start_epoch = ckpt_info_ct['epoch'] + 1
        model_ehr, ckpt_info_ehr = ModelSaver.load_model(cfgs_ehr.ckpt_path)
        cfgs_ehr.start_epoch = ckpt_info_ehr['epoch'] + 1
        model_ct = model_ct.to(device)
        model_ehr = model_ehr.to(device)
    dataloader_test = MultimodalLoader(parser, phase='test', is_training=False)
    study2slices = defaultdict(list)
    study2probs = defaultdict(list)
    study2labels = {}
    # logger = TestLogger(cfgs_joint, len(dataloader_test.dataset),
    #                     dataloader_test.dataset.ct_data.pixel_dict)
    util.print_err(f'Writing model outputs to {cfgs_joint.resutls_dir}')
    with tqdm(total=len(dataloader_test.dataset), unit='windows') as progress_bar:
        ct_input: torch.Tensor
        ehr_input: torch.Tensor
        target: torch.Tensor
        for j, (ct_input, ehr_input, targets_dict) in enumerate(dataloader_test):
            target: torch.Tensor = targets_dict['is_abnormal']
            with torch.no_grad():
                if cfgs_joint.joint_training:
                    cls_logits = model(ct_input.to(device),
                                       ehr_input.to(device))
                else:
                    ct_logits = model_ct(ct_input.to(device))
                    ehr_logits = model_ehr(ehr_input.to(device))
                    cls_logits = (ct_logits + ehr_logits) / 2
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
    with open('{}/preds.pickle'.format(cfgs_joint.results_dir), "wb") as fp:
        pickle.dump(predictions, fp)

    # Write the slice indices used for the features
    print("Write slice indices")
    # with open(os.path.join(cfgs_joint.results_dir, 'xgb', 'series2slices.json'), 'w') as json_fh:
    #     json.dump(study2slices, json_fh, sort_keys=True, indent=4)

    # Compute AUROC and AUPRC using max aggregation, write to files
    max_probs, labels = np.array(max_probs), np.array(labels)
    metrics = {
        cfgs_joint.phase + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, max_probs),
        cfgs_joint.phase + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, max_probs),
    }
    print("Write metrics")
    with open(os.path.join(cfgs_joint.results_dir, 'metrics.txt'), 'w') as metrics_fh:
        for k, v in metrics.items():
            metrics_fh.write('{}: {:.5f}\n'.format(k, v))

    curves = {
        cfgs_joint.phase + '_' + 'PRC': sk_metrics.precision_recall_curve(labels, max_probs),
        cfgs_joint.phase + '_' + 'ROC': sk_metrics.roc_curve(labels, max_probs)
    }
    for name, curve in curves.items():
        curve_np = util.get_plot(name, curve)
        curve_img = Image.fromarray(curve_np)
        curve_img.save(os.path.join(
            cfgs_joint.results_dir, '{}.png'.format(name)))


if __name__ == "__main__":
    parser = CfgParser()
    if parser.phase == 'train':
        train(parser)
    elif parser.phase == 'test':
        test(parser)
    else:
        raise ValueError('`phase` must be `train` or `test`')
