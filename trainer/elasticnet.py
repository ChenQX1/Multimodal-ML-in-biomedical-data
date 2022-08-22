import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
sys.path.insert(0, os.path.abspath('./'))   # !!

from args.cfg_parser import CfgParser
from datasets.ehr_dataset import EHRDataset
from models.elasticnet import ElasticNet
from saver.model_saver import ModelSaver
import util


def train(parser: CfgParser):
    cfgs_ehr = parser.cfgs_ehr
    local_rank = 0
    device = torch.device(cfgs_ehr.device)
    # ------------- Dataloader -----------
    data_train = EHRDataset(cfgs_ehr, phase='train')
    data_val = EHRDataset(cfgs_ehr, phase='val')
    dataloader_train = DataLoader(
        data_train, cfgs_ehr.batch_size, sampler=data_train.ehr_data.index.values, drop_last=True)
    dataloader_val = DataLoader(
        data_val, cfgs_ehr.batch_size, sampler=data_val.ehr_data.index.values, drop_last=True)
    # ------------- Model ---------------
    model = ElasticNet(
        in_feats=data_train.ehr_data.shape[1], out_feats=cfgs_ehr.num_classes)
    model = model.to(device)
    # ------------- Training Loop ------------
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = util.get_optimizer(model.parameters(), cfgs_ehr)
    # lr_scheduler = util.get_scheduler(optimizer, cfgs_ehr)
    saver = ModelSaver(cfgs_ehr.save_dir, cfgs_ehr.epochs_per_save,
                       cfgs_ehr.max_ckpts, cfgs_ehr.best_ckpt_metric, False)
    loss_train = []
    loss_val = []
    for i in tqdm(range(cfgs_ehr.num_epochs)):
        model.train()
        loss_batch = []
        for iter_n, (feat, target) in enumerate(dataloader_train):
            preds = model(feat.to(device))
            loss = criterion(preds, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_batch.append(loss.detach().cpu().numpy())
            if iter_n != 0 and iter_n % 100 == 0:
                print(f'    --------- iter: {iter_n} loss: {loss}')
        loss_train.append(np.mean(loss_batch))
        with torch.no_grad():
            model.eval()
            loss_batch = [
                criterion(model(feat.to(device)), target.to(
                device)).detach().cpu().numpy() for feat, target in dataloader_val
            ]
            loss_val.append(np.mean(loss_batch))
        # -------- Saving ckpts ----------------
        print(f'    epoch {i} training loss:    {loss_train[-1]}, val loss:     {loss_val[-1]}')
        if local_rank == 0:
            saver.save(
                i, model, optimizer, None, device, metric_val=loss_val
            )

def test(cfgs):
    pass


if __name__ == "__main__":
    parser = CfgParser()
    train(parser)
