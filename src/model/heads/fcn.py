import torch.nn as nn

from . import utils_heads
from .base import BaseHead


class FCNHead(BaseHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head_endpoints = ['final']
        out_channels = self.in_channels // 4

        self.bottleneck = nn.ModuleDict({t: utils_heads.ConvBNReLU(self.in_channels,
                                                                   out_channels,
                                                                   kernel_size=3,
                                                                   norm_layer=nn.BatchNorm2d,
                                                                   activation_layer=nn.ReLU)
                                         for t in self.tasks})
        self.final_logits = nn.ModuleDict({t: nn.Conv2d(out_channels,
                                                        self.task_channel_mapping[t]['final'],
                                                        kernel_size=1,
                                                        bias=True)
                                           for t in self.tasks})
        self.init_weights()

    def forward(self, inp, inp_shape, **kwargs):
        inp = self._transform_inputs(inp)
        task_specific_feats = {t: self.bottleneck[t](inp) for t in self.tasks}

        final_pred = {t: self.final_logits[t](task_specific_feats[t]) for t in self.tasks}

        final_pred = {t: nn.functional.interpolate(
            final_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.tasks}
        return {'final': final_pred}
