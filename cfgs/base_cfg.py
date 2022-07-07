class CommonCfg(object):
    def __init__(self) -> None:
        self.name = None  # required
        self.is_training = None
        self.toy = False
        self.toy_size = 5
        self.cudnn_benchmark = False
        self.data_dir = None
        self.gpu_ids = -1
        self.rand_seed = 42
        self.save_dir = None
        self.resutls_dir = None


# Write fundamental configurations in the class.
# Usually these cfgs do not change.
class IMGCfg(CommonCfg):
    def __init__(self) -> None:
        super().__init__()
        self.series = 'sagittal'
        self.pe_types = ['central', 'segmental']
        self.hide_probability = 0
        self.hide_level = 'window'
        self.only_topmost_window = False
        self.pkl_path = None    # required
        self.img_format = 'raw'
        self.dataset = None     # required
        self.patience = 10
        self.best_ckpt_metric = 'val_loss'
        self.max_eval = -1
        self.max_ckpts = 2

        self.use_pretrained = False
        self.include_normals = False
        self.use_hem = False
        self.fine_tune = True
        self.fine_tuning_lr = 0
        self.fine_tuning_boundary = 'encoder.3'
        self.hidden_dim = 32
        self.elastic_transform = False
        self.do_hflip = True
        self.do_vflip = False
        self.do_rotate = True
        self.do_jitter = True
        self.do_center_pe = True
        self.abnormal_prob = 0.5
        self.resize_shape = [224, 224]  # '224, 224'
        self.crop_shape = [208, 208]    # '208, 208'
        self.min_abnormal_slices = 4

        self.num_workers = 4
        self.vstep_size = 1

        self.model = 'PENet'
        self.ckpt_path = None
        self.num_channels = 3
        self.num_classes = 1
        self.num_slices = 32
        self.batch_size = 8
        self.init_method = 'kaiming'
        self.model_depth = 50

        self.agg_method = 'max'
        self.threshold_size = 0

        self.eval_mode = 'series'
        self.do_classify = False
        self.num_visuals = 4

        self.visualize_all = False


class EHRCfg(CommonCfg):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = None
        self.ckpt_path = None

        self.num_classes = 1
        self.batch_size = 16
        self.num_epochs = 1

        self.optimizer = 'sgd'
        self.learning_rate = 0.01
        # SGD
        self.sgd_momentum = 0.9
        self.sdg_dampening = 0.9
        # Adam
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        # Regularization
        self.dropout_prob = 0
        self.l1_lambda = 0.1
        self.weight_decay = 0.001