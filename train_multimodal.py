from args.cfg_parser import CfgParser
import data_loader
import models
import torch
import torch.nn as nn
import util
from torch.utils.data import DataLoader
import numpy as np
import pprint

from args import TrainArgParser
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver
from datasets.ehr_dataset import EHRDataset
from models.pe_elastic_net import ElasticNet, PEElasticNet


def fit_multimodal(img_input, img_label, ehr_input, ehr_label, model_list, loss_fns, optims, logger, target_dict):
    penet, elasticnet = model_list[0], model_list[1]
    logits_img = penet(img_input)
    # Joint Training
    if isinstance(penet.module, PEElasticNet):
        assert len(loss_fns) == 1, 'Joint training needs one loss function.'
        loss_fn = loss_fns[0]
        logits_joint = elasticnet(logits_img + ehr_input)
        cls_loss = loss_fn(logits_joint, ehr_label)
        loss = cls_loss.mean()

        logger.log_iter(img_input, logits_joint, target_dict, loss, optims[0])

        optims[0].zero_grad()
        optims[1].zero_grad()
        loss.backward()
        optims[0].step()
        optims[1].step()
    # Post fusion
    else:
        logits_ehr = elasticnet(ehr_input)
        loss_fn_img = loss_fns[0]
        loss_fn_ehr = loss_fns[1]

        cls_loss_img = loss_fn_img(logits_img, img_label)
        loss_img = cls_loss_img.mean()
        cls_loss_ehr = loss_fn_ehr(logits_ehr, ehr_label)
        loss_ehr = cls_loss_ehr.mean()

        logger.log_iter(img_input, logits_img, target_dict,
                        cls_loss_img.mean(), optims[0])

        for loss, optim in zip([loss_img, loss_ehr], optims):
            optim.zero_grad()
            loss.backward()
            optim.step()


def train_elastic_net(args):
    args.ckpt_path = './ckpts/elastic.pth'
    dt_train = EHRDataset(args, phase='train')
    dt_val = EHRDataset(args, phase='val')
    loader_train = DataLoader(
        dt_train, args.batch_size, sampler=dt_train.ehr_data.index.values)
    loader_val = DataLoader(
        dt_val, args.batch_size * 2, sampler=dt_val.ehr_data.index.values)
    net = ElasticNet(
        in_feats=dt_train.ehr_data.shape[1], out_feats=args.num_classes)
    cls_loss_fn = nn.BCELoss(reduction='mean')
    optimizer = util.get_optimizer(net.parameters(), args)
    net = net.to(args.device)

    loss_log_train = []
    loss_log_val = []

    for i in range(args.num_epochs):
        net.train()
        loss_ls = []
        for j, batch in enumerate(loader_train):
            dt, target = batch[0].to(args.device), batch[1].to(args.device)
            logits = net(dt)
            loss = cls_loss_fn(logits, target) + args.l1_lambda * net.l1_reg()\
                + (1 - args.l1_lambda) * net.l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.detach().cpu().numpy())
        loss_mean = np.mean(loss_ls)
        loss_log_train.append(loss_mean)

        net.eval()
        with torch.no_grad():
            loss_val = np.mean(
                [cls_loss_fn(net(dt.to(args.device)), target.to(
                    args.device)).detach().cpu().numpy() for dt, target in loader_val]
            )
            loss_log_val.append(loss_val)
        print(
            f'=========== Epoch {i} ============\n    training loss: {loss_mean}\n   validation loss: {loss_val}')

    torch.save(net.state_dict(), args.ckpt_path)

    # Prepare for joint training
    # TODO: Refactor this code block
    net = ElasticNet(
        in_feats=dt_train.ehr_data.shape[1], out_feats=args.num_classes)
    net.load_state_dict(torch.load(args.ckpt_path))
    net = nn.DataParallel(net, args.gpu_ids)
    optimizer = torch.optim.Adam(net.parameters())
    cls_loss_fn = nn.BCELoss(reduction='mean')
    dt_train = EHRDataset(args, phase='train')


def train_penet(args):
    # Get the model
    if args.ckpt_path and not args.use_pretrained:
        model, ckpt_info = ModelSaver.load_model(
            args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        model = model_fn(**vars(args))
        if args.use_pretrained:
            model.load_pretrained(args.ckpt_path, args.gpu_ids)
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Get optimizer and scheduler
    if args.use_pretrained or args.fine_tune:
        parameters = model.module.fine_tuning_parameters(
            args.fine_tuning_boundary, args.fine_tuning_lr)
    else:
        parameters = model.parameters()
    optimizer = util.get_optimizer(parameters, args)
    lr_scheduler = util.get_scheduler(optimizer, args)
    if args.ckpt_path and not args.use_pretrained and not args.fine_tune:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)

    # Get logger, evaluator, saver
    cls_loss_fn = util.get_loss_fn(
        is_classification=True, dataset=args.dataset, size_average=False)
    data_loader_fn = data_loader.__dict__[args.data_loader]
    train_loader = data_loader_fn(args, phase='train', is_training=True)
    logger = TrainLogger(args, len(train_loader.dataset),
                         train_loader.dataset.pixel_dict)
    eval_loaders = [data_loader_fn(args, phase='val', is_training=False)]
    evaluator = ModelEvaluator(args.do_classify, args.dataset, eval_loaders, logger,
                               args.agg_method, args.num_visuals, args.max_eval, args.epochs_per_eval)
    saver = ModelSaver(args.save_dir, args.epochs_per_save,
                       args.max_ckpts, args.best_ckpt_metric, args.maximize_metric)
    
    # Train PENet model
    while not logger.is_finished_training():
        logger.start_epoch()

        for inputs, target_dict in train_loader:
            logger.start_iter()
            with torch.set_grad_enabled(True):
                inputs = inputs.to(args.device)
                # img_label = target_dict['is_abnormal'].to(args.device)
                # ehr_input, ehr_label = ehr_train[target_dict['study_num']]
                # ehr_input, ehr_label = ehr_input.to(
                #     args.device), ehr_label.to(args.device)
                cls_logits = model(inputs)
                cls_target = target_dict['is_abnormal'].to(args.device)
                cls_loss = cls_loss_fn(cls_logits, cls_target)
                loss = cls_loss.mean()
                logger.log_iter(inputs, cls_logits, target_dict, loss, optimizer)

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
    # parser = TrainArgParser()
    # args_ = parser.parse_args()
    # train(args_)

    parser = CfgParser()

    train_penet(parser.img_modal)
