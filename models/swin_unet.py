from .layers.swin_unet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
import torch.nn as nn
import torch
from copy import deepcopy


class MySwinUnet(SwinTransformerSys):
    def __init__(
            self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
            embed_dim=96, **kwargs):
        super(
            MySwinUnet, self).__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            num_classes=num_classes, embed_dim=embed_dim, **kwargs)
        self.output1 = nn.Sequential(
            nn.Conv2d(num_classes, num_classes // 2, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(num_classes // 2, 1, kernel_size=3),
            nn.ReLU()
        )
        self.output2 = nn.Linear((img_size - 4) ** 2, 1)

        self.model_args = {
            'in_chans': in_chans,
            'num_classes': num_classes,
            'embed_dim': embed_dim
        }
        self.model_args.update(kwargs)

    def forward(self, x: torch.Tensor):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        x = self.output1(x)
        x = x.view(x.shape[0], -1)
        x = self.output2(x)

        return x

    def args_dict(self):
        return self.model_args

    def load(self, ckpt_path):
        pretrained_dict = torch.load(ckpt_path, map_location='cpu')
        pretrained_dict = pretrained_dict['model']

        model_dict = self.state_dict()
        full_dict = deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3-int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k:v})

        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]

        msg = self.load_state_dict(full_dict, strict=False)