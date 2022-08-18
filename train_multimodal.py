import os
from args.cfg_parser import CfgParser
import data_loader
import models
import torch
import torch.nn as nn
import util
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver
from datasets.ehr_dataset import EHRDataset
from models.pe_elastic_net import ElasticNet


def fit_multimodal(parser: CfgParser):
    cfgs_ct = parser.cfgs_ct
    cfgs_ehr = parser.cfgs_ehr
    local_rank = dist.get_rank()
    device = torch.device('cuda', local_rank)
    # PENet
    # Get the model
    if cfgs_ct.ckpt_path and not cfgs_ct.use_pretrained:
        model_penet, ckpt_info_penet = ModelSaver.load_model(cfgs_ct.ckpt_path, joint_training=True)
        cfgs_ct.start_epoch = ckpt_info_penet['epoch'] + 1
    else:
        model_fn = models.__dict__[cfgs_ct.model]
        model_penet = model_fn(**vars(cfgs_ct))
        if cfgs_ct.use_pretrained:
            model_penet.load_pretrained(cfgs_ct.ckpt_path, cfgs_ct.gpu_ids)
    model_penet = DDP(model_penet.to(device))
    model_penet.train()
    # Get optimizer and scheduler
    if cfgs_ct.use_pretrained or cfgs_ct.fine_tune:
        parameters = model_penet.module.fine_tuning_parameters(
            cfgs_ct.fine_tuning_boundary, cfgs_ct.fine_tuning_lr)
    else:
        parameters = model_penet.parameters()
    optimizer_penet = util.get_optimizer(parameters, cfgs_ct)
    lr_scheduler_penet = util.get_scheduler(optimizer_penet, cfgs_ct)
    if cfgs_ct.ckpt_path and not cfgs_ct.use_pretrained and not cfgs_ct.fine_tune:
        ModelSaver.load_optimizer(
            cfgs_ct.ckpt_path, optimizer_penet, lr_scheduler_penet)
    # Get logger, evaluator, saver
    cls_loss_fn_penet = util.get_loss_fn(
        is_classification=True, dataset=cfgs_ct.dataset, size_average=False)
    data_loader_fn = data_loader.__dict__[cfgs_ct.data_loader]
    loader_train_ct = data_loader_fn(
        cfgs_ct, phase='train', is_training=True)
    logger = TrainLogger(cfgs_ct, len(loader_train_ct.dataset),
                         loader_train_ct.dataset.pixel_dict)
    eval_loaders = [data_loader_fn(cfgs_ct, phase='val', is_training=False)]
    evaluator = ModelEvaluator(cfgs_ct.do_classify, cfgs_ct.dataset, eval_loaders, logger,
                               cfgs_ct.agg_method, cfgs_ct.num_visuals, cfgs_ct.max_eval, cfgs_ct.epochs_per_eval)
    saver_penet = ModelSaver(cfgs_ct.save_dir, cfgs_ct.epochs_per_save,
                       cfgs_ct.max_ckpts, cfgs_ct.best_ckpt_metric, cfgs_ct.maximize_metric)
    if parser.train_img:
        train_penet(cfgs_ct, logger, loader_train_ct, model_penet,
                    cls_loss_fn_penet, optimizer_penet, lr_scheduler_penet, evaluator, saver_penet)
    ct_feat_size = cfgs_ct.ct_feat_size

    # EHR
    dt_train_ehr = EHRDataset(cfgs_ehr, phase='train')
    dt_val_ehr = EHRDataset(cfgs_ehr, phase='val')
    loader_train_ehr = DataLoader(
        dt_train_ehr, cfgs_ehr.batch_size, sampler=dt_train_ehr.ehr_data.index.values)
    loader_val_ehr = DataLoader(
        dt_val_ehr, cfgs_ehr.batch_size * 2, sampler=dt_val_ehr.ehr_data.index.values)
    if parser.joint_training:
        model_elastic_net = ElasticNet(
            in_feats=dt_train_ehr.ehr_data.shape[1], out_feats=ct_feat_size)
    else:
        model_elastic_net = ElasticNet(
            in_feats=dt_train_ehr.ehr_data.shape[1], out_feats=cfgs_ehr.num_classes)
    model_elastic_net = DDP(model_elastic_net.to(device))
    cls_loss_fn_ehr = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer_ehr = util.get_optimizer(
        model_elastic_net.parameters(), cfgs_ehr)
    if parser.train_ehr and not parser.joint_training:
        train_elastic_net(cfgs_ehr, loader_train_ehr, loader_val_ehr,
                          model_elastic_net, optimizer_ehr, cls_loss_fn_ehr)

    # Multimodal
    if parser.joint_training:
        print('============== Joint Training ==============')
        joint_loss = nn.BCEWithLogitsLoss(reduction='mean')
        classifier_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(ct_feat_size * 2, cfgs_ct.num_classes),
        )
        classifier_head = DDP(classifier_head.to(device))
        optimizer_classifier = util.get_optimizer(classifier_head.parameters())
        loss_log_train = []
        loss_log_val = []
        for i in range(parser.num_epochs):
            loader_train_ct.sampler.set_epoch(i)
            loss_ls = []
            for iter_n, (ct_input, target_dict) in enumerate(loader_train_ct):
                with torch.set_grad_enabled(True):
                    ct_input = ct_input.to(device)
                    ct_target = target_dict['is_abnormal'].to(device)
                    ehr_input, ehr_target = dt_train_ehr[target_dict['study_num'].numpy(
                    )]
                    ehr_input, ehr_target = ehr_input.to(
                        device), ehr_target.to(device)
                    ct_feat = model_penet.module.forward_feature(
                        ct_input)   # !!
                    ehr_feat = model_elastic_net(ehr_input)

                    joint_input = torch.concat([ct_feat, ehr_feat], dim=1)
                    joint_logits = classifier_head(joint_input)
                    loss = joint_loss(joint_logits, ct_target)

                    optimizer_penet.zero_grad()
                    optimizer_ehr.zero_grad()
                    optimizer_classifier.zero_grad()
                    loss.backward()
                    optimizer_penet.step()
                    optimizer_ehr.step()
                    optimizer_classifier.step()

                    loss_ls.append(loss.detach().cpu().numpy())
                    if iter_n % 100 == 0 and iter_n != 0:
                        print(f'    +++++ iter: {iter_n} loss: {loss}')

            loss_mean = np.mean(loss_ls)
            print(
                f'=========== Epoch {i} ============\n    training loss: {loss_mean}')
            loss_log_train.append(loss_mean)
            if local_rank == 0:
                saver_penet.save(i, model_penet.module, optimizer_penet,
                           lr_scheduler_penet, cfgs_ct.device, metric_val=None)
                torch.save(
                    model_elastic_net.module.state_dict(), os.path.join(parser.save_dir, 'joint_elastic_net.pth')
                )
                torch.save(
                    classifier_head.module.state_dict(), os.path.join(parser.save_dir, 'classifier_head.pth')
                )


def train_elastic_net(args, loader_train, loader_val, model, optimizer, cls_loss_fn):
    loss_log_train = []
    loss_log_val = []

    last_loss = -torch.inf
    trigger_times = 0

    for i in range(args.num_epochs):
        model.train()
        loss_ls = []
        for j, batch in enumerate(loader_train):
            dt, target = batch[0].to(args.device), batch[1].to(args.device)
            logits = model(dt)
            loss = cls_loss_fn(logits, target) + args.alpha * args.l1_ratio * \
                model.l1_reg() + 0.5 * args.alpha * (1-args.l1_ratio) * model.l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.detach().cpu().numpy())
        loss_train = np.mean(loss_ls)
        loss_log_train.append(loss_train)

        model.eval()
        loss_ls = []
        with torch.no_grad():
            for j, batch in enumerate(loader_val):
                dt, target = batch[0].to(args.device), batch[1].to(args.device)
                logits = model(dt)
                loss = cls_loss_fn(logits, target) + args.alpha * args.l1_ratio * \
                    model.l1_reg() + 0.5 * args.alpha * (1-args.l1_ratio) * model.l2_reg()
                loss_ls.append(loss.cpu().numpy())
            loss_val = np.mean(loss_ls)
            loss_log_val.append(loss_val)
        print(
            f'=========== Epoch {i} ============\n    training loss: {loss_train}\n   validation loss: {loss_val}')
        if loss_val > last_loss:
            trigger_times += 1
        last_loss = loss_val
        if args.n_early_stopping > 0 and trigger_times >= args.n_early_stopping:
            break

    print(f'Saving Model to: {args.save_dir}')
    torch.save(model.state_dict(),
               '/'.join([args.save_dir, f'elastic_net.pth']))


def train_penet(args, logger, loader_train, model, loss_fn, optimizer, lr_scheduler, evaluator, saver):
    while not logger.is_finished_training():
        logger.start_epoch()
        print('begin training')
        for inputs, target_dict in loader_train:
            logger.start_iter()
            with torch.set_grad_enabled(True):
                inputs = inputs.to(args.device)
                cls_logits = model(inputs)
                cls_target = target_dict['is_abnormal'].to(args.device)
                cls_loss = loss_fn(cls_logits, cls_target)
                loss = cls_loss.mean()

                logger.log_iter(inputs, cls_logits,
                                target_dict, loss, optimizer)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.end_iter()
            util.step_scheduler(lr_scheduler, global_step=logger.global_step)

        metrics, curves = evaluator.evaluate(
            model, args.device, logger.epoch)
        saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,
                   metric_val=metrics.get(args.best_ckpt_metric, None))
        logger.end_epoch(metrics, curves)
        util.step_scheduler(lr_scheduler, metrics, epoch=logger.epoch,
                            best_ckpt_metric=args.best_ckpt_metric)


if __name__ == '__main__':
    util.set_spawn_enabled()
    dist.init_process_group(backend='nccl')

    parser = CfgParser(phase='train')
    fit_multimodal(parser)
