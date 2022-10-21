import hydra
from omegaconf import DictConfig, open_dict
from evaluator.model_evaluator import ModelEvaluator
from PIL import Image
import sklearn.metrics as sk_metrics
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from typing import Dict
from time import time
from collections import defaultdict
from args import ConfigParser
from logger.train_logger import TrainLogger
from models import MyViT
import models
from saver.model_saver import ModelSaver
from data_loader import CTDataLoader
import util
from logger.my_logger import plot_loss


@hydra.main(config_path='config', config_name='ct_swinunet', version_base='1.2')
def train(cfgs: DictConfig):
    with open_dict(cfgs):
        cfgs.common.is_training = True
        cfgs.common.phase = 'train'
    cfgs: DictConfig = ConfigParser(cfgs).cfgs

    local_rank = 0
    device = torch.device(cfgs.common.device)

    # DataLoader
    dataloader_train = CTDataLoader(
        cfgs.dataset, phase='train', is_training=True)
    dataloader_val = CTDataLoader(
        cfgs.dataset, phase='val', is_training=False)

    # Model
    if cfgs.model.ckpt_path and cfgs.model.use_pretrained:
        model, ckpt_info = ModelSaver.load_model(
            cfgs.model.ckpt_path)
        with open_dict(cfgs):
            cfgs.common.start_epoch = ckpt_info['epoch'] + 1
    else:
        model = models.__dict__[cfgs.model.model](
            img_size = cfgs.dataset.resize_shape[0],
            in_chans=cfgs.dataset.num_slices,
            num_classes=cfgs.dataset.num_slices, drop_rate=cfgs.model.drop_p,
            attn_drop_rate=cfgs.model.attn_drop_p, embed_dim=cfgs.model.embed_dim)
        model.load(cfgs.model.pretrained_ckpt_path)
    model = model.to(device)

    # Optimizer
    optimizer = util.get_optimizer(model.parameters(), args=cfgs.optimizer)
    lr_schduler = util.get_scheduler(optimizer, cfgs.optimizer)
    criterion = util.get_loss_fn(
        is_classification=True, dataset=cfgs.dataset.dataset,
        size_average=False)
    saver = ModelSaver(cfgs.common.save_dir, cfgs.common.epochs_per_save,
                       cfgs.common.max_ckpts, cfgs.common.best_ckpt_metric, cfgs.common.maximize_metric)
    # ----------- Traing Loop ----------
    loss_train = []
    loss_val = []
    for i in range(cfgs.common.num_epochs):
        ts = time()
        model.train()
        loss_epoch = []
        t = tqdm(total=len(dataloader_train) + len(dataloader_val), unit='batch')
        for iter_n, (ct_input, target_dict) in enumerate(dataloader_train):
            target = target_dict['is_abnormal']
            cls_logits = model(ct_input.squeeze().to(device))
            loss = criterion(cls_logits, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.detach().cpu().numpy())
            t.update(1)
        loss_train.append(np.mean(loss_epoch))

        model.eval()
        loss_epoch = []
        with torch.no_grad():
            for iter_n, (ct_input, target_dict) in enumerate(dataloader_val):
                target = target_dict['is_abnormal']
                cls_logits = model(ct_input.squeeze().to(device))
                loss = criterion(cls_logits, target.to(device))
                loss_epoch.append(loss.detach().cpu().numpy())
                t.update(1)
        loss_val.append(np.mean(loss_epoch))
        t.close()

        print(
            f'    epoch {i} training loss:    {loss_train[-1]}, validation loss:  {loss_val[-1]}, time: {time() - ts}')
        if local_rank == 0:
            saver.save(i, model, optimizer, lr_schduler,
                       device, metric_val=loss_val[-1])
        lr_schduler.step()
        plot_loss(loss_train, loss_val, cfgs.common.save_dir)


if __name__ == "__main__":
    train()
