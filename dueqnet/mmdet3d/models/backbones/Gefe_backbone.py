import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..model_utils import build_Gefe_conv_layer, build_Gefe_norm_layer, build_Gefe_lift_layer

from mmdet.models import BACKBONES


@BACKBONES.register_module()
class GefeBackbone(BaseModule):
    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 conv_bias=False,
                 norm_eps=1e-3,
                 norm_momentum=0.01,
                 init_cfg=None,
                 Gefe_trconv=True):
        super(GefeBackbone, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]

        # input: (B, in_c, H, W)
        # lifting
        self.lift = nn.Sequential(
            build_Gefe_lift_layer(in_filters[0], in_filters[0], 3,
                                 stride=1, padding=1, bias=conv_bias),
            build_Gefe_norm_layer(in_filters[0]),
            nn.ReLU(inplace=True),
        )
        # after lifting input: (B, out_c, 4, H, W)

        group_blocks = []
        for i, layer_num in enumerate(layer_nums):
            group_block = [
                build_Gefe_conv_layer(in_filters[i], out_channels[i], 3,
                                     stride=layer_strides[i], padding=1, bias=conv_bias),
                build_Gefe_norm_layer(out_channels[i], eps=norm_eps, momentum=norm_momentum),
                nn.ReLU(inplace=True)
            ]
            for j in range(layer_num):
                group_block.append(
                    build_Gefe_conv_layer(out_channels[i], out_channels[i], 3,
                                         stride=1, padding=1, bias=conv_bias))
                group_block.append(build_Gefe_norm_layer(out_channels[i], eps=norm_eps, momentum=norm_momentum))
                group_block.append(nn.ReLU(inplace=True))
            group_block = nn.Sequential(*group_block)
            group_blocks.append(group_block)
        # three c4cnn
        # after first: (B, out_c[0], 4, H/2, W/2)
        # after second: (B, out_c[1], 4, H/4, W/4)
        # after third: (B, out_c[2], 4, H/8, W/8)

        self.blocks = nn.ModuleList(group_blocks)

        self.Gefe_trconv = Gefe_trconv

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W)

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        # 1. lifting
        x = self.lift(x)

        # 2. group conv
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)

        if self.Gefe_trconv:
            return tuple(outs)
        else:
            for i in range(len(outs)):
                s_shape = outs[i].shape
                outs[i] = outs[i].view(s_shape[0], -1, s_shape[-2], s_shape[-1])

            return tuple(outs)
