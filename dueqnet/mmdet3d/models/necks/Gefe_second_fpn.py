import numpy as np
import torch
from ..model_utils import build_Gefe_conv_layer, build_Gefe_norm_layer, build_Gefe_trconv_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS


@NECKS.register_module()
class GefeSECONDFPN(BaseModule):
    """Implement global equivariance feature extraction 02.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 use_conv_for_no_stride=False,
                 conv_bias=False,
                 upsample_bias=False,
                 norm_eps=1e-3,
                 norm_momentum=0.01,
                 init_cfg=None):
        super(GefeSECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_Gefe_trconv_layer(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=upsample_bias)
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_Gefe_conv_layer(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride,
                    bias=conv_bias)

            deblock = nn.Sequential(upsample_layer,
                                    build_Gefe_norm_layer(out_channel, eps=norm_eps, momentum=norm_momentum),
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 5D Tensor in (N, C, 4, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        o_shape = out.shape
        out = out.view(o_shape[0], -1, o_shape[-2], o_shape[-1])
        return [out]
