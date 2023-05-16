from .Gefe_layers_01 import (GefeLiftingConv2d, GefeConv2d,
                             GefeBatchNorm2d, GefeTrConv2d)


def build_Gefe_lift_layer(*args, **kwargs):
    Gefe_lift_layer = GefeLiftingConv2d(*args, **kwargs)

    return Gefe_lift_layer


def build_Gefe_conv_layer(*args, **kwargs):
    Gefe_conv_layer = GefeConv2d(*args, **kwargs)

    return Gefe_conv_layer


def build_Gefe_norm_layer(*args, **kwargs):
    Gefe_norm_layer = GefeBatchNorm2d(*args, **kwargs)

    return Gefe_norm_layer


def build_Gefe_trconv_layer(*args, **kwargs):
    Gefe_trconv_layer = GefeTrConv2d(*args, **kwargs)

    return Gefe_trconv_layer
