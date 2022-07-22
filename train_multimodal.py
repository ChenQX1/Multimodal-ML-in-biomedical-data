from pyexpat import model
from args.cfg_parser import CfgParser
import data_loader
import models
import torch
import torch.nn as nn
import util
from torch.utils.data import DataLoader
import numpy as np

from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver
from datasets.ehr_dataset import EHRDataset
from models.pe_elastic_net import ElasticNet, PEElasticNet


def fit_multimodal(parser):
    img_modal = parser.img_modal
    ehr_modal = parser.ehr_modal
    device = img_modal.device
    # PENet
    # Get the model
    if img_modal.ckpt_path and not img_modal.use_pretrained:
        ckpt_dict = torch.load(img_modal.ckpt_path, map_location=torch.device('cpu'))
        model_fn = models.__dict__[img_modal.model]
        model_penet = model_fn(**ckpt_dict['model_args'])
        model_penet.load_state_dict(ckpt_dict['model_state'])
        ckpt_info_penet = ckpt_dict['ckpt_info']
        img_modal.start_epoch = ckpt_info_penet['epoch'] + 1
    else:
        model_fn = models.__dict__[img_modal.model]
        model_penet = model_fn(**vars(img_modal))
        if img_modal.use_pretrained:
            model_penet.load_pretrained(img_modal.ckpt_path, img_modal.gpu_ids)
    model_penet = nn.DataParallel(model_penet, img_modal.gpu_ids)
    model_penet = model_penet.to(img_modal.device)
    model_penet.train()
    # Get optimizer and scheduler
    if img_modal.use_pretrained or img_modal.fine_tune:
        parameters = model_penet.module.fine_tuning_parameters(
            img_modal.fine_tuning_boundary, img_modal.fine_tuning_lr)
    else:
        parameters = model_penet.parameters()
    optimizer_penet = util.get_optimizer(parameters, img_modal)
    lr_scheduler_penet = util.get_scheduler(optimizer_penet, img_modal)
    if img_modal.ckpt_path and not img_modal.use_pretrained and not img_modal.fine_tune:
        ModelSaver.load_optimizer(
            img_modal.ckpt_path, optimizer_penet, lr_scheduler_penet)
    # Get logger, evaluator, saver
    cls_loss_fn_penet = util.get_loss_fn(
        is_classification=True, dataset=img_modal.dataset, size_average=False)
    data_loader_fn = data_loader.__dict__[img_modal.data_loader]
    loader_train_penet = data_loader_fn(
        img_modal, phase='train', is_training=True)
    logger = TrainLogger(img_modal, len(loader_train_penet.dataset),
                         loader_train_penet.dataset.pixel_dict)
    eval_loaders = [data_loader_fn(img_modal, phase='val', is_training=False)]
    evaluator = ModelEvaluator(img_modal.do_classify, img_modal.dataset, eval_loaders, logger,
                               img_modal.agg_method, img_modal.num_visuals, img_modal.max_eval, img_modal.epochs_per_eval)
    saver = ModelSaver(img_modal.save_dir, img_modal.epochs_per_save,
                       img_modal.max_ckpts, img_modal.best_ckpt_metric, img_modal.maximize_metric)
    if parser.train_img:
        train_penet(img_modal, logger, loader_train_penet, model_penet,
                    cls_loss_fn_penet, optimizer_penet, lr_scheduler_penet, evaluator, saver)
    img_feat_size = 2048

    # EHR
    dt_train_ehr = EHRDataset(ehr_modal, phase='train')
    dt_val_ehr = EHRDataset(ehr_modal, phase='val')
    loader_train_ehr = DataLoader(
        dt_train_ehr, ehr_modal.batch_size, sampler=dt_train_ehr.ehr_data.index.values)
    loader_val_ehr = DataLoader(
        dt_val_ehr, ehr_modal.batch_size * 2, sampler=dt_val_ehr.ehr_data.index.values)
    if parser.joint_training:
        model_elastic_net = ElasticNet(
        in_feats=dt_train_ehr.ehr_data.shape[1], out_feats=img_feat_size)
    else: 
        model_elastic_net = ElasticNet(
        in_feats=dt_train_ehr.ehr_data.shape[1], out_feats=ehr_modal.num_classes)
    model_elastic_net = model_elastic_net.to(device)
    cls_loss_fn_ehr = nn.BCEWithLogitsLoss(reduction='mean')

    optimizer_ehr = util.get_optimizer(
        model_elastic_net.parameters(), ehr_modal)
    model_elastic_net = model_elastic_net.to(ehr_modal.device)
    if parser.train_ehr:
        train_elastic_net(ehr_modal, loader_train_ehr, loader_val_ehr,
                          model_elastic_net, optimizer_ehr, cls_loss_fn_ehr)

    # Multimodal
    if parser.joint_training:
        print('============== Joint Training ==============')
        joint_loss = nn.BCEWithLogitsLoss(reduction='mean')
        classifier_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(img_feat_size * 2, img_modal.num_classes),
        ).to(device)
        loss_log_train = []
        loss_log_val = []
        for i in range(parser.num_epochs):
            loss_ls = []
            for iter_n, (img_input, target_dict) in enumerate(loader_train_penet):
                with torch.set_grad_enabled(True):
                    img_input = img_input.to(device)
                    img_target = target_dict['is_abnormal'].to(device)
                    ehr_input, ehr_target = dt_train_ehr[target_dict['study_num'].numpy(
                    )]
                    ehr_input, ehr_target = ehr_input.to(device), ehr_target.to(device)
                    img_feat = model_penet.module.forward_feature(img_input)   # !!
                    ehr_feat = model_elastic_net(ehr_input)

                    joint_input = torch.concat([img_feat, ehr_feat], dim=1)
                    joint_logits = classifier_head(joint_input)
                    loss = joint_loss(joint_logits, img_target)

                    optimizer_penet.zero_grad()
                    optimizer_ehr.zero_grad()
                    loss.backward()
                    optimizer_penet.step()
                    optimizer_ehr.step()

                    loss_ls.append(loss.detach().cpu().numpy())
                    if iter_n % 100 == 0:
                        print(f'    +++++ iter: {iter_n} loss: {loss}')
            loss_mean = np.mean(loss_ls)
            print(
                f'=========== Epoch {i} ============\n    training loss: {loss_mean}')
            loss_log_train.append(loss_mean)
        torch.save(classifier_head.state_dict(), '/'.join([parser.save_dir, 'classifier_head.pth']))
        torch.save(model_elastic_net.state_dict(), '/'.join([parser.save_dir, 'joint_elastic_net.pth']))
        saver.save(i, model_penet, optimizer_penet, lr_scheduler_penet, img_modal.device, metric_val=None)


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
            loss = cls_loss_fn(logits, target) + args.alpha * args.l1_ratio * model.l1_reg() + 0.5 * args.alpha * (1-args.l1_ratio)* model.l2_reg()
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
                loss = cls_loss_fn(logits, target) + args.alpha * args.l1_ratio * model.l1_reg() + 0.5 * args.alpha * (1-args.l1_ratio)* model.l2_reg()
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

    parser = CfgParser(phase='train')
    fit_multimodal(parser)
