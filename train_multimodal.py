import data_loader
import models
import torch
import torch.nn as nn
import util
from torch.utils.data import DataLoader
import numpy as np


from args import TrainArgParser
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver
from datasets.ehr_dataset import EHRDataset
from models.pe_elastic_net import ElasticNet, PEElasticNet


def fit(img_input, img_label, ehr_input, ehr_label, model_list, loss_fns, optims, logger, target_dict):
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


def train_elasticnet(net, train_loader, val_loader, optimizer, loss_fn, n_epoch, device):
    l1_labmda = 0.1
    l2_lambda = 0.1
    net.to(device)
    print(f'============= Trainng EHR data =============')
    for i in range(n_epoch):
        net.train()
        loss_epoch = []
        for j, batch in enumerate(train_loader):
            dt, target = batch[0].to(device), batch[1].to(device)
            logits = net(dt)
            loss = loss_fn(logits, target).mean() + net.l1_reg() * \
                l1_labmda + net.l2_reg()*l2_lambda
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.detach().cpu().numpy())
        print(
            f'    ============== Epoch {i}, mean training loss: {np.mean(loss_epoch)}')
        net.eval()
        with torch.no_grad():
            loss_val = np.mean([loss_fn(net(dt.to(device)), target.to(
                device)).detach().cpu().numpy() for dt, target in val_loader])
            print(
                f'    =========== Epoch {i}, mean validation loss: {loss_val}')

    return net


def train(args):
    if args.ckpt_path and not args.use_pretrained:
        model_img, ckpt_info = ModelSaver.load_model(
            args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        model_img = model_fn(**vars(args))
        if args.use_pretrained:
            model_img.load_pretrained(args.ckpt_path, args.gpu_ids)
        model_img = nn.DataParallel(model_img, args.gpu_ids)
    model_img = model_img.to(args.device)
    model_img.train()

    # Get optimizer and scheduler
    if args.use_pretrained or args.fine_tune:
        parameters = model_img.module.fine_tuning_parameters(
            args.fine_tuning_boundary, args.fine_tuning_lr)
    else:
        parameters = model_img.parameters()
    optimizer_img = util.get_optimizer(parameters, args)
    lr_scheduler = util.get_scheduler(optimizer_img, args)
    if args.ckpt_path and not args.use_pretrained and not args.fine_tune:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer_img, lr_scheduler)

    # Get logger, evaluator, saver
    cls_loss_fn_img = util.get_loss_fn(
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

    # # Train Elastic Net
    # # TODO: Refactor this code block
    elasticnet_ckpt_dir = './ckpts/elastic.pth'
    ehr_train = EHRDataset(args, phase='train')
    ehr_val = EHRDataset(args, phase='val')
    ehr_loader_train = DataLoader(
        ehr_train, 16, sampler=ehr_train.ehr_data.index.values)
    ehr_loader_val = DataLoader(
        ehr_val, 32, sampler=ehr_val.ehr_data.index.values)
    model_ehr = ElasticNet(in_feats=ehr_train.ehr_data.shape[1], out_feats=1)
    cls_loss_fn_ehr = nn.BCELoss(reduction='mean')
    optimizer_ehr = torch.optim.Adam(model_ehr.parameters())
    model_ehr = train_elasticnet(
        model_ehr, ehr_loader_train, ehr_loader_val, optimizer_ehr, cls_loss_fn_ehr, 200, args.device)
    torch.save(model_ehr.state_dict(), elasticnet_ckpt_dir)

    # Prepare for joint training
    # TODO: Refactor this code block
    model_ehr = ElasticNet()
    model_ehr.load_state_dict(torch.load(elasticnet_ckpt_dir))
    model_ehr = nn.DataParallel(model_ehr, args.gpu_ids)
    optimizer_ehr = torch.optim.Adam(model_ehr.parameters())
    cls_loss_fn_ehr = nn.BCELoss(reduction='mean')
    ehr_train = EHRDataset(args, phase='train')

    model_list = [model_img, model_ehr]
    if isinstance(model_img.module, PEElasticNet):  # joint training
        loss_fns = [cls_loss_fn_ehr]
        optims = [optimizer_img, optimizer_ehr]
    else:   # post fusion
        loss_fns = [cls_loss_fn_img, cls_loss_fn_ehr]
        optims = [optimizer_img, optimizer_ehr]

    # Train PENet model
    while not logger.is_finished_training():
        logger.start_epoch()

        for img_input, target_dict in train_loader:
            logger.start_iter()
            with torch.set_grad_enabled(True):
                img_input = img_input.to(args.device)
                img_label = target_dict['is_abnormal'].to(args.device)
                ehr_input, ehr_label = ehr_train[target_dict['study_num']]
                ehr_input, ehr_label = ehr_input.to(
                    args.device), ehr_label.to(args.device)

                fit(img_input, img_label, ehr_input, ehr_label,
                    model_list, loss_fns, optims, logger, target_dict)

            logger.end_iter()
            util.step_scheduler(lr_scheduler, global_step=logger.global_step)

        metrics, curves = evaluator.evaluate(
            model_img, args.device, logger.epoch)
        saver.save(logger.epoch, model_img, optimizer_img, lr_scheduler, args.device,
                   metric_val=metrics.get(args.best_ckpt_metric, None))
        logger.end_epoch(metrics, curves)
        util.step_scheduler(lr_scheduler, metrics, epoch=logger.epoch,
                            best_ckpt_metric=args.best_ckpt_metric)


if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TrainArgParser()
    args_ = parser.parse_args()
    train(args_)
