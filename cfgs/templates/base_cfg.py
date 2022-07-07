
# class XCfg is a template
# all necessary cfg parsing should be processed in `CfgParser`
class BasicPENetCfg(object):
    def __init__(self, **kwargs) -> None:
        self.name = None  # required
        self.is_training = None
        self.toy = False
        self.toy_size = 5
        self.deterministic = True
        self.cudnn_benchmark = False

        self.series = 'sagittal'
        self.pe_types = ['central', 'segmental']
        self.hide_probability = 0
        self.hide_level = 'window'
        self.only_topmost_window = False

        self.data_dir = None  # required
        self.pkl_path = None    # required
        self.img_format = 'raw'
        self.dataset = None     # required

        self.num_workers = 4
        self.vstep_size = 1

        self.resize_shape = [224, 224]  # '224, 224'
        self.crop_shape = [208, 208]    # '208, 208'
        self.min_abnormal_slices = 4

        self.model = 'PENet'
        self.ckpt_path = None
        self.num_channels = 3
        self.num_classes = 1
        self.num_slices = 32

        self.batch_size = 8
        self.gpu_ids = -1   # set '-1' for cpu, or array like [0, 1, 2] for gpu ids
        self.init_method = 'kaiming'
        self.model_depth = 50

        self.agg_method = 'max'
        self.threshold_size = 0

        self.eval_mode = 'series'
        self.do_classify = False
        self.num_visuals = 4
        self.save_dir = None


class BasicEHRCfg(object):
    def __init__(self) -> None:
        self.is_training = None
        self.data_dir = None
        self.dataset = None
        self.ckpt_path = None

        self.gpu_ids = -1
        self.num_classes = 1
        self.batch_size = 16
