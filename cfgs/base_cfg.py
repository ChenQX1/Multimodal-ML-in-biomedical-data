from dataclasses import dataclass
from typing import List, Union


@dataclass
class CommonCfg(object):
    name: str # required
    is_training: bool = False
    cudnn_benchmark: bool = False
    gpu_ids: Union[int, List[int], str] = -1    # `-1` for cpu; `mps` for MPS backend
    rand_seed = 42
    data_dir: str = 'data'
    save_dir: str = 'logs'
    resutls_dir: str = 'results'
    ckpt_path: str = 'ckpts'
    use_pretrained: bool = False

    num_epochs: int = 1
    num_workers: int = 4
    batch_size: int = 8
    num_classes: int = 1

    optimizer: str = 'sgd'
    learning_rate: int = 0.001

# Write fundamental configurations in the class.
# Usually these cfgs do not change.
@dataclass
class CTCfg(CommonCfg):
    toy: bool = False
    toy_size: int = 5
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
    # optimizer
    optimizer = 'sgd'
    learning_rate = 0.01
    sgd_momentum = 0.9
    sdg_dampening = 0.9
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    # Regularization
    dropout_prob = 0
    l1_lambda = 0.1
    weight_decay = 0.001
