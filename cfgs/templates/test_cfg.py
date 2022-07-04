from .base_cfg import BasicPENetCfg


class PENetCfg(BasicPENetCfg):
    def __init__(self) -> None:
        self.is_training = False
        self.phase = 'val'
        self.results_dir = 'results/'
        self.visualize_all = False
