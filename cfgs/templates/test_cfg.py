from .base_cfg import BasicEHRCfg, BasicPENetCfg


class PENetCfg(BasicPENetCfg):
    def __init__(self, **kwargs) -> None:
        super(PENetCfg, self).__init__(**kwargs)
        self.is_training = False
        self.phase = 'val'
        self.results_dir = 'results/'
        self.visualize_all = False


class EHRCfg(BasicEHRCfg):
    def __init__(self, **kwargs) -> None:
        super(EHRCfg, self).__init__(**kwargs)

        self.is_training = False
        