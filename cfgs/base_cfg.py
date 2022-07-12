from dataclasses import dataclass


@dataclass
class CommonCfg(object):
        name = None  # required
        is_training = None
        toy = False
        toy_size = 5
        cudnn_benchmark = False
        data_dir = None
        gpu_ids = -1
        rand_seed = 42
        save_dir = None
        resutls_dir = None


# Write fundamental configurations in the class.
# Usually these cfgs do not change.
@dataclass
class IMGCfg(CommonCfg):
        series = 'sagittal'
        pe_types = ['central', 'segmental']
        hide_probability = 0
        hide_level = 'window'
        only_topmost_window = False
        pkl_path = None    # required
        img_format = 'raw'
        dataset = None     # required
        patience = 10
        best_ckpt_metric = 'val_loss'
        max_eval = -1
        max_ckpts = 2

        use_pretrained = False
        include_normals = False
        use_hem = False
        fine_tune = True
        fine_tuning_lr = 0
        fine_tuning_boundary = 'encoder.3'
        hidden_dim = 32
        elastic_transform = False
        do_hflip = True
        do_vflip = False
        do_rotate = True
        do_jitter = True
        do_center_pe = True
        abnormal_prob = 0.5
        resize_shape = [224, 224]  # '224, 224'
        crop_shape = [208, 208]    # '208, 208'
        min_abnormal_slices = 4

        num_workers = 4
        vstep_size = 1

        model = 'PENet'
        ckpt_path = None
        num_channels = 3
        num_classes = 1
        num_slices = 32
        batch_size = 8
        init_method = 'kaiming'
        model_depth = 50

        agg_method = 'max'
        threshold_size = 0

        eval_mode = 'series'
        do_classify = False
        num_visuals = 4

        visualize_all = False

@dataclass
class EHRCfg(CommonCfg):
        dataset = None
        ckpt_path = None

        num_classes = 1
        batch_size = 16
        num_epochs = 1

        optimizer = 'sgd'
        learning_rate = 0.01
        # SGD
        sgd_momentum = 0.9
        sdg_dampening = 0.9
        # Adam
        adam_beta_1 = 0.9
        adam_beta_2 = 0.999
        # Regularization
        dropout_prob = 0
        l1_lambda = 0.1
        weight_decay = 0.001
