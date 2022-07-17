import json
import pickle
import numpy as np
import os
import sklearn.metrics as sk_metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from args.cfg_parser import CfgParser
from datasets.ehr_dataset import EHRDataset
from models.pe_elastic_net import ElasticNet
import util

from args import TestArgParser
from data_loader import CTDataLoader
from collections import defaultdict
from logger import TestLogger
from PIL import Image
from saver import ModelSaver
from tqdm import tqdm


def test(parser):
    img_modal = parser.img_modal
    ehr_modal = parser.ehr_modal
    device = img_modal.device
    print("Stage 1")
    model_penet, ckpt_info = ModelSaver.load_model(
        img_modal.ckpt_path, img_modal.gpu_ids)
    print("Stage 2")
    img_modal.start_epoch = ckpt_info['epoch'] + 1
    model_penet = model_penet.to(device)
    print("Stage 3")
    model_penet.eval()
    print("Stage 4")
    data_loader_penet = CTDataLoader(
        img_modal, phase=img_modal.phase, is_training=False)
    study2slices = defaultdict(list)
    study2probs = defaultdict(list)
    study2labels = {}
    logger = TestLogger(img_modal, len(data_loader_penet.dataset),
                        data_loader_penet.dataset.pixel_dict)

    # EHR
    dt_ehr = EHRDataset(ehr_modal, phase='test')
    model_elastic_net = ElasticNet(
        in_feats=dt_ehr.ehr_data.shape[1], out_feats=ehr_modal.num_classes)
    model_elastic_net.load_state_dict(torch.load(ehr_modal.ckpt_path, map_location=device))
    model_elastic_net = model_elastic_net.to(device)
    if parser.joint_training:
        connettor_linear = nn.Linear(
            2048*2*6*6, dt_ehr.ehr_data.shape[1]).to(device)
        connettor_linear.load_state_dict(
            torch.load('./ckpts/connector_linear.pth'))

    # Get model outputs, log to TensorBoard, write masks to disk window-by-window
    util.print_err('Writing model outputs to {}...'.format(
        img_modal.results_dir))
    with tqdm(total=len(data_loader_penet.dataset), unit=' windows') as progress_bar:
        for i, (img_input, targets_dict) in enumerate(data_loader_penet):
            with torch.no_grad():
                ehr_input, ehr_target = dt_ehr[targets_dict['study_num'].numpy(
                )]
                ehr_input = ehr_input.to(device)
                if parser.joint_training:
                    img_feat = model_penet.forward_feature(img_input)
                    img_feat = connettor_linear(img_feat)
                    cls_logits = model_elastic_net(ehr_input + img_feat)
                else:
                    img_logits = model_penet(img_input.to(device))
                    ehr_logits = model_elastic_net(ehr_input)
                    cls_logits = (img_logits + ehr_logits) / 2

                cls_probs = torch.sigmoid(cls_logits)

            if img_modal.visualize_all:
                logger.visualize(
                    img_input, cls_logits, targets_dict=None, phase=img_modal.phase, unique_id=i)

            max_probs = cls_probs.to('cpu').numpy()
            for study_num, slice_idx, prob in \
                    zip(targets_dict['study_num'], targets_dict['slice_idx'], list(max_probs)):
                # Convert to standard python data types
                study_num = int(study_num)
                slice_idx = int(slice_idx)

                # Save series num for aggregation
                study2slices[study_num].append(slice_idx)
                study2probs[study_num].append(prob.item())

                series = data_loader_penet.get_series(study_num)
                if study_num not in study2labels:
                    study2labels[study_num] = int(series.is_positive)

            progress_bar.update(img_input.size(0))

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
    with open('{}/preds.pickle'.format(img_modal.results_dir), "wb") as fp:
        pickle.dump(predictions, fp)

    # Write features for XGBoost
    save_for_xgb(img_modal.results_dir, study2probs, study2labels)
    # Write the slice indices used for the features
    print("Write slice indices")
    with open(os.path.join(img_modal.results_dir, 'xgb', 'series2slices.json'), 'w') as json_fh:
        json.dump(study2slices, json_fh, sort_keys=True, indent=4)

    # Compute AUROC and AUPRC using max aggregation, write to files
    max_probs, labels = np.array(max_probs), np.array(labels)
    metrics = {
        img_modal.phase + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, max_probs),
        img_modal.phase + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, max_probs),
    }
    print("Write metrics")
    with open(os.path.join(img_modal.results_dir, 'metrics.txt'), 'w') as metrics_fh:
        for k, v in metrics.items():
            metrics_fh.write('{}: {:.5f}\n'.format(k, v))

    curves = {
        img_modal.phase + '_' + 'PRC': sk_metrics.precision_recall_curve(labels, max_probs),
        img_modal.phase + '_' + 'ROC': sk_metrics.roc_curve(labels, max_probs)
    }
    for name, curve in curves.items():
        curve_np = util.get_plot(name, curve)
        curve_img = Image.fromarray(curve_np)
        curve_img.save(os.path.join(
            img_modal.results_dir, '{}.png'.format(name)))


def save_for_xgb(results_dir, series2probs, series2labels):
    """Write window-level and series-level features to train an XGBoost classifier.
    Args:
        results_dir: Path to results directory for writing outputs.
        series2probs: Dict mapping series numbers to probabilities.
        series2labels: Dict mapping series numbers to labels.
    """

    # Convert to numpy
    xgb_inputs = np.zeros([len(series2probs), max(len(p)
                          for p in series2probs.values())])
    xgb_labels = np.zeros(len(series2labels))
    for i, (series_num, probs) in enumerate(series2probs.items()):
        xgb_inputs[i, :len(probs)] = np.array(probs).ravel()
        xgb_labels[i] = series2labels[series_num]

    # Write to disk
    os.makedirs(os.path.join(results_dir, 'xgb'), exist_ok=True)
    xgb_inputs_path = os.path.join(results_dir, 'xgb', 'inputs.npy')
    xgb_labels_path = os.path.join(results_dir, 'xgb', 'labels.npy')
    np.save(xgb_inputs_path, xgb_inputs)
    np.save(xgb_labels_path, xgb_labels)


if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = CfgParser(phase='test')
    test(parser)
