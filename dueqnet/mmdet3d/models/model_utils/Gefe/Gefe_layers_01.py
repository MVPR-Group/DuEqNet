import torch
import torch.nn.functional as F

from .group_00 import rotate, rotate_p4


class IsotropicConv2d(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, padding: int = 1, bias: bool = True):

        super(IsotropicConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        # In this block you need to create a tensor which stores the learnable weights
        # Recall that each 3x3 filter has only `2` learnable parameters, one for the center and one for the ring around it.
        # In total, there are `in_channels * out_channels` different filters.
        # Remember to wrap the weights tensor into a `torch.nn.Parameter` and set `requires_grad = True`

        # initialize the weights with some random values from a normal distribution with std = 1 / sqrt(out_channels * in_channels)

        # BEGIN SOLUTION
        self.weight = torch.normal(mean=0, std=1 / (out_channels * in_channels) ** 0.5,
                                   size=(out_channels, in_channels, 2))
        # 卷积核的每个通道都是独立分布的，因此一个核就具有 in_channels*2 个参数
        # 共 out_channels 个卷积核
        self.weight = torch.nn.Parameter(self.weight, requires_grad=True)

        # END SOLUTION
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None

    def build_filter(self) -> torch.Tensor:
        # using the tensor of learnable parameters, build the `out_channels x in_channels x 3 x 3` filter

        # Make sure that the tensor `filter3x3` is on the same device of `self.weight`

        # BEGIN SOLUTION

        out_channels, in_channels, _ = self.weight.shape
        mask = torch.ones(self.kernel_size, self.kernel_size, dtype=torch.bool)
        center = int(self.kernel_size / 2)
        mask[center, center] = 0
        filter3x3 = torch.empty((out_channels, in_channels, self.kernel_size, self.kernel_size),
                                device=self.weight.device)
        # out_channels个核，每个核in_channels个通道
        filter3x3[:, :, center, center] = self.weight[:, :, 0]
        filter3x3[:, :, mask] = self.weight[:, :, 1].unsqueeze(-1)

        # END SOLUTION

        return filter3x3

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _filter = self.build_filter()

        return torch.conv2d(x, _filter,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            bias=self.bias)


class GefeLiftingConv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0,
                 stride: int = 1, dilation: int = 1, bias: bool = True):

        super(GefeLiftingConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels

        # In this block you need to create a tensor which stores the learnable filters
        # Recall that this layer should have `out_channels x in_channels` different learnable filters, each of shape `kernel_size x kernel_size`
        # During the forward pass, you will build the bigger filter of shape `out_channels x 4 x in_channels x kernel_size x kernel_size` by rotating 4 times
        # the learnable filters in `self.weight`

        # initialize the weights with some random values from a normal distribution with std = 1 / sqrt(out_channels * in_channels)

        # BEGIN SOLUTION
        self.weight = torch.normal(mean=0, std=1 / (out_channels * in_channels) ** 0.5,
                                   size=(out_channels, in_channels, kernel_size, kernel_size))
        self.weight = torch.nn.Parameter(self.weight, requires_grad=True)
        # END SOLUTION

        # This time, you also need to build the bias
        # The bias is shared over the 4 rotations
        # In total, the bias has `out_channels` learnable parameters, one for each independent output
        # In the forward pass, you need to convert this bias into an "expanded" bias by repeating each entry `4` times

        # BEGIN SOLUTION
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
        # END SOLUTION

    def build_filter(self) -> torch.Tensor:
        # using the tensors of learnable parameters, build
        # - the `out_channels x 4 x in_channels x kernel_size x kernel_size` filter
        # - the `out_channels x 4` bias

        # Make sure that the filter and the bias tensors are on the same device of `self.weight` and `self.bias`

        # First build the filter
        # Recall that `_filter[:, i, :, :, :]` should contain the learnable filter rotated `i` times

        # BEGIN SOLUTION
        _filter = torch.empty((self.out_channels, 4, self.in_channels, self.kernel_size, self.kernel_size),
                              device=self.weight.device)
        for i in range(4):
            _filter[:, i] = rotate(self.weight, i)
        # END SOLUTION

        # Now build the bias
        # Recall that `_bias[:, i]` should contain a copy of the learnable bias for each `i=0,1,2,3`

        # BEGIN SOLUTION
        if self.bias is not None:
            _bias = self.bias.unsqueeze(1).repeat(1, 4).to(self.bias.device)
        else:
            _bias = None
        # END SOLUTION

        return _filter, _bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _filter, _bias = self.build_filter()

        if _bias is not None:
            assert _bias.shape == (self.out_channels, 4)
        assert _filter.shape == (self.out_channels, 4, self.in_channels, self.kernel_size, self.kernel_size)

        # to be able to use torch.conv2d, we need to reshape the filter and bias to stack together all filters
        _filter = _filter.reshape(self.out_channels * 4, self.in_channels, self.kernel_size, self.kernel_size)
        if _bias is not None:
            _bias = _bias.reshape(self.out_channels * 4)

        out = torch.conv2d(x, _filter, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, bias=_bias)

        # `out` has now shape `batch_size x out_channels*4 x W x H`
        # we need to reshape it to `batch_size x out_channels x 4 x W x H` to have the shape we expect

        return out.view(-1, self.out_channels, 4, out.shape[-2], out.shape[-1])


class GefeConv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0,
                 stride: int = 1, dilation: int = 1, bias: bool = True):

        super(GefeConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels

        # In this block you need to create a tensor which stores the learnable filters
        # Recall that this layer should have `out_channels x in_channels` different learnable filters, each of shape `4 x kernel_size x kernel_size`
        # During the forward pass, you will build the bigger filter of shape `out_channels x 4 x in_channels x 4 x kernel_size x kernel_size` by rotating 4 times
        # the learnable filters in `self.weight`

        # initialize the weights with some random values from a normal distribution with std = 1 / np.sqrt(out_channels * in_channels)

        # BEGIN SOLUTION
        self.weight = torch.normal(mean=0, std=1 / (out_channels * in_channels) ** 0.5,
                                   size=(out_channels, in_channels, 4, kernel_size, kernel_size))
        self.weight = torch.nn.Parameter(self.weight, requires_grad=True)
        # END SOLUTION

        # The bias is shared over the 4 rotations
        # In total, the bias has `out_channels` learnable parameters, one for each independent output
        # In the forward pass, you need to convert this bias into an "expanded" bias by repeating each entry `4` times

        # BEGIN SOLUTION
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
        # END SOLUTION

    def build_filter(self):
        # using the tensors of learnable parameters, build
        # - the `out_channels x 4 x in_channels x 4 x kernel_size x kernel_size` filter
        # - the `out_channels x 4` bias

        # Make sure that the filter and the bias tensors are on the same device of `self.weight` and `self.bias`

        # First build the filter
        # Recall that `_filter[:, r, :, :, :, :]` should contain the learnable filter rotated `r` times
        # Also, recall that a rotation includes both a rotation of the pixels and a cyclic permutation of the 4 rotational input channels

        # BEGIN SOLUTION
        _filter = torch.empty((self.out_channels, 4, self.in_channels, 4, self.kernel_size, self.kernel_size),
                              device=self.weight.device)
        for i in range(4):
            _filter[:, i] = rotate_p4(self.weight, i)

        # END SOLUTION

        # Now build the bias
        # Recall that `_bias[:, i]` should contain a copy of the learnable bias for each `i`

        # BEGIN SOLUTION
        if self.bias is not None:
            _bias = self.bias.unsqueeze(1).repeat(1, 4).to(self.bias.device)
        else:
            _bias = None
        # END SOLUTION

        return _filter, _bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _filter, _bias = self.build_filter()

        if _bias is not None:
            assert _bias.shape == (self.out_channels, 4)
        assert _filter.shape == (self.out_channels, 4, self.in_channels, 4, self.kernel_size, self.kernel_size)

        # to be able to use torch.conv2d, we need to reshape the filter and bias to stack together all filters
        _filter = _filter.reshape(self.out_channels * 4, self.in_channels * 4, self.kernel_size, self.kernel_size)
        if _bias is not None:
            _bias = _bias.reshape(self.out_channels * 4)

        # this time, also the input has shape `batch_size x in_channels x 4 x W x H`
        # so we need to reshape it to `batch_size x in_channels*4 x W x H` to be able to use torch.conv2d
        x = x.view(x.shape[0], self.in_channels * 4, x.shape[-2], x.shape[-1])

        out = torch.conv2d(x, _filter, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, bias=_bias)

        # `out` has now shape `batch_size x out_channels*4 x W x H`
        # we need to reshape it to `batch_size x out_channels x 4 x W x H` to have the shape we expect

        return out.view(-1, self.out_channels, 4, out.shape[-2], out.shape[-1])


class GefeBatchNorm2d(torch.nn.Module):
    def __init__(self, in_channels: int, eps=1e-05, momentum=0.1, affine=True):
        super(GefeBatchNorm2d, self).__init__()

        self.in_channels = in_channels
        # self.eps = eps
        # self.momentum = momentum
        # self.affine = affine

        self.batchnorm2d = torch.nn.BatchNorm2d(num_features=in_channels, eps=eps, momentum=momentum, affine=affine)

    def forward(self, input):
        assert len(input.shape) == 5

        output = []
        for i in range(input.shape[-3]):
            output.append(self.batchnorm2d(input[:, :, i, :, :]))
        output = torch.stack(output, dim=-3)

        return output


class GefeTrConv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0,
                 bias: bool = True):
        super(GefeTrConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = 1

        self.weight = torch.normal(mean=0, std=1 / (out_channels * in_channels) ** 0.5,
                                   size=(out_channels, in_channels, 4, kernel_size, kernel_size))
        self.weight = torch.nn.Parameter(self.weight, requires_grad=True)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None

    def build_filter(self):
        _filter = None
        _bias = None

        _filter = torch.empty((self.out_channels, 4, self.in_channels, 4, self.kernel_size, self.kernel_size),
                              device=self.weight.device)
        for i in range(4):
            _filter[:, i] = rotate_p4(self.weight, i)

        if self.bias is not None:
            _bias = self.bias.unsqueeze(1).repeat(1, 4).to(self.bias.device)
        else:
            _bias = None

        return _filter, _bias

    def forward(self, x):
        _filter, _bias = self.build_filter()

        if _bias is not None:
            assert _bias.shape == (self.out_channels, 4)
        assert _filter.shape == (self.out_channels, 4, self.in_channels, 4, self.kernel_size, self.kernel_size)

        # to be able to use torch.conv2d, we need to reshape the filter and bias to stack together all filters
        _filter = _filter.reshape(self.out_channels*4, self.in_channels*4, self.kernel_size, self.kernel_size)
        _filter = _filter.permute(1, 0, 2, 3)
        if _bias is not None:
            _bias = _bias.reshape(self.out_channels * 4)

        # this time, also the input has shape `batch_size x in_channels x 4 x W x H`
        # so we need to reshape it to `batch_size x in_channels*4 x W x H` to be able to use torch.conv2d
        x = x.view(x.shape[0], self.in_channels*4, x.shape[-2], x.shape[-1])

        out = F.conv_transpose2d(x, _filter, _bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

        # `out` has now shape `batch_size x out_channels*4 x W x H`
        # we need to reshape it to `batch_size x out_channels x 4 x W x H` to have the shape we expect

        return out.view(-1, self.out_channels, 4, out.shape[-2], out.shape[-1])

