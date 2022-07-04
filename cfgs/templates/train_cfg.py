from .base_cfg import BasicPENetCfg


class PENetCfg(BasicPENetCfg):
    def __init__(self, **kwargs) -> None:
        super(PENetCfg, self).__init__(**kwargs)

        self.is_training = True
        self.epochs_per_save = 5
        self.iters_per_print = 4
        self.epochs_per_eval = 1
        self.iters_per_visual = 80
        self.learning_rate = 0.1
        self.lr_scheduler = 'cosine_warmup'
        self.lr_decay_gamma = 0.1
        self.lr_decay_step = 300000
        self.lr_warmup_steps = 10000
        self.lr_milestones = '50, 125, 250'
        self.patience = 10
        self.num_epoches =300
        self.max_ckpts = 2
        self.best_ckpt_metric = 'val_loss'
        self.max_eval = -1
        self.optimizer = 'sgd'
        self.sgd_momentum = 0.9
        self.sdg_dampening = 0.9
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        self.dropout_prob = 0
        
        self.hidden_dim = 32
        self.elastic_transform = False
        self.do_hflip = True
        self.do_vflip = False
        self.do_rotate = True
        self.do_jitter = True
        self.do_center_pe = True
        self.abnormal_prob = 0.5
        self.use_pretrained = False
        self.include_normals = False
        self.use_hem = False
        self.fine_tune = True
        self.fine_tuning_lr = 0
        self.fine_tuning_boundary = 'encoder.3'
