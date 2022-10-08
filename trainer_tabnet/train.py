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
import models
from saver.model_saver import ModelSaver
import util


@hydra.main(config_path='config', config_name='ehr_tabnet', version_base='1.2')
def train(cfgs: DictConfig):
    with open_dict(cfgs):
        cfgs.common.is_training = True
        cfgs.common.phase = 'train'
    cfgs: DictConfig = ConfigParser(cfgs).cfgs
    
    local_rank = 0
    device = torch.device(cfgs.common.device)

    # ------------- Dataloader -----------
    data_train = datasets.__dict__[cfgs.dataset.dataset](cfgs.dataset, phase='train')
    data_val= datasets.__dict__[cfgs.dataset.dataset](cfgs.dataset, phase='val')
    dataloader_train = DataLoader(
        data_train, cfgs.common.batch_size, sampler=data_train.ehr_data.index.values, drop_last=True)
    dataloader_val = DataLoader(
        data_val, cfgs.common.batch_size, sampler=data_val.ehr_data.index.values, drop_last=True)
    # ------------- Model ---------------
    n_in = data_train.ehr_data.shape[1]
    model = models.__dict__[cfgs.model.model](n_in, cfgs.common.num_classes, **cfgs.model)
    model = model.to(device)
    # ------------- Training Loop ------------
    criterion = util.get_loss_fn(is_classification=True, dataset=cfgs.dataset.dataset, size_average=False)
    optimizer = util.get_optimizer(model.parameters(), cfgs.optimizer)
    lr_scheduler = util.get_scheduler(optimizer, cfgs.optimizer)
    saver = ModelSaver(cfgs.common.save_dir, cfgs.common.epochs_per_save,
                       cfgs.common.max_ckpts, cfgs.common.best_ckpt_metric, cfgs.common.maximize_metric)
    loss_train = []
    loss_val = []
    # print(dataloader_train.dataset.ehr_data.shape)  # (1454, 1451)
    for i in tqdm(range(cfgs.common.num_epochs)):
        model.train()
        loss_batch = []
        for iter_n, (feat, target) in enumerate(dataloader_train):
            preds, _ = model(feat.to(device))
            loss = criterion(preds, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_batch.append(loss.detach().cpu().numpy())
            if iter_n != 0 and iter_n % 100 == 0:
                print(f'    --------- iter: {iter_n} loss: {loss}')
        loss_train.append(np.mean(loss_batch))
        model.eval()
        with torch.no_grad():    
            loss_batch = [
                criterion(model(feat.to(device))[0], target.to(
                    device)).detach().cpu().numpy() for feat, target in dataloader_val
            ]
            loss_val.append(np.mean(loss_batch))
        # -------- Saving ckpts ----------------
        print(
            f'    epoch {i} training loss:    {loss_train[-1]}, val loss:     {loss_val[-1]}')
        if local_rank == 0:
            saver.save(
                i, model, optimizer, lr_scheduler, device, metric_val=loss_val[-1]
            )


if __name__ == "__main__":
    train()
