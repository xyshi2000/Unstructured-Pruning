import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer
from spikingjelly.clock_driven import surrogate, neuron, functional


def create_msif():
    return neuron.MultiStepIFNode(v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(),
                                  detach_reset=True, backend='cupy')


def create_mslif(v_threshold=1., v_reset=0., tau=2., surrogate_function=surrogate.ATan(),
                 detach_reset=True):
    return neuron.MultiStepLIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau,
                                   surrogate_function=surrogate_function, detach_reset=detach_reset,
                                   backend='cupy')


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                     dilation=dilation, bias=bias, groups=groups)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)
