import torch
from torch import nn
from spikingjelly.clock_driven import neuron, functional
from typing import List


class Mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_value = None
        self.pruning = False
        self.lmbda = 0
        self.temp = 1

    def _init_mask(self, shape, mean: float, std: float = 0):
        self.mask_value = torch.nn.parameter.Parameter(
            torch.normal(mean, std, size=shape, device='cuda'))
        return self.mask_value

    def _pruning(self, flag: bool):
        self.pruning = flag

    def _set_temp(self, temp: float):
        self.temp = temp

    def mask(self):
        if self.mask_value is None:
            return None
        elif self.pruning:
            return torch.sigmoid(self.temp * self.mask_value.detach())
        else:
            return torch.where(self.mask_value > 0, 1, 0).float()

    def left(self):
        if self.mask_value is None:
            return 0., 0.
        else:
            return self.mask().sum().item(), self.mask_value.numel()

    def forward(self, x):
        if self.mask_value is None:
            return x
        elif self.pruning:
            return torch.sigmoid(self.temp * self.mask_value) * x
        else:
            return torch.where(self.mask_value > 0, x, 0)

    def single_step_forward(self, x):
        if self.mask_value is None:
            return x
        mask_value = self.mask_value
        if len(mask_value.shape) == 5:
            mask_value = self.mask_value.squeeze(0)
        if self.pruning:
            return torch.sigmoid(self.temp * mask_value) * x
        else:
            return torch.where(mask_value > 0, x, 0)


class ConvBlock(nn.Module):
    def __init__(self, conv: nn.Conv2d, norm: nn.BatchNorm2d, node: neuron.BaseNode,
                 static: bool = False, T: int = None, sparse_weights: bool = False,
                 sparse_neurons: bool = False) -> None:
        super(ConvBlock, self).__init__()
        assert isinstance(conv, nn.Conv2d)
        assert norm is None or isinstance(norm, nn.BatchNorm2d)
        assert node is None or isinstance(node, neuron.BaseNode)
        self.conv = conv
        self.norm = norm
        self.node = node
        self.static = static
        self.T = T
        self.sparse_weights = sparse_weights
        self.sparse_neurons = sparse_neurons
        self.weight_mask = Mask()
        self.neuron_mask = Mask()

    def init_mask(self, weights_mean: float, neurons_mean: float, weights_std: float = 0,
                  neurons_std: float = 0):
        masks = []
        if self.sparse_weights:
            masks.append(
                self.weight_mask._init_mask(self.conv.weight.shape, weights_mean, weights_std))
        if self.sparse_neurons:
            masks.append(
                self.neuron_mask._init_mask(
                    self.node.v.unsqueeze(0).shape, neurons_mean, neurons_std))
        return masks

    def _pruning(self, flag: bool):
        if self.sparse_weights:
            self.weight_mask._pruning(flag)
        if self.sparse_neurons:
            self.neuron_mask._pruning(flag)

    def set_temp(self, temp: float):
        self.set_weight_temp(temp)
        self.set_neuron_temp(temp)

    def set_weight_temp(self, temp: float):
        if self.sparse_weights:
            self.weight_mask._set_temp(temp)

    def set_neuron_temp(self, temp: float):
        if self.sparse_neurons:
            self.neuron_mask._set_temp(temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mask(self.conv.weight)
        if self.static:
            x = torch.nn.functional.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride,
                                           padding=self.conv.padding, dilation=self.conv.dilation,
                                           groups=self.conv.groups)
            if self.norm is not None:
                x = self.norm(x)
            x.unsqueeze_(0)
            x = x.repeat(self.T, 1, 1, 1, 1)
        else:
            x_shape = [x.shape[0], x.shape[1]]
            x = x.flatten(0, 1)
            x = torch.nn.functional.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride,
                                           padding=self.conv.padding, dilation=self.conv.dilation,
                                           groups=self.conv.groups)
            if self.norm is not None:
                x = self.norm(x)
            x_shape.extend(x.shape[1:])
            x = x.view(x_shape)
        if self.node is not None:
            x = self.node(x)
        x = self.neuron_mask(x)
        return x

    def connects(self, sparse, dense):
        with torch.no_grad():
            if self.sparse_weights:
                weight = self.weight_mask.mask()
            else:
                weight = torch.ones_like(self.conv.weight)
            sparse = torch.nn.functional.conv2d(sparse, weight, bias=None, stride=self.conv.stride,
                                                padding=self.conv.padding,
                                                dilation=self.conv.dilation,
                                                groups=self.conv.groups)
            dense = torch.nn.functional.conv2d(dense, torch.ones_like(self.conv.weight), bias=None,
                                               stride=self.conv.stride, padding=self.conv.padding,
                                               dilation=self.conv.dilation, groups=self.conv.groups)
            sparse = self.neuron_mask.single_step_forward(sparse)
            conn = sparse.sum().item()
            total = dense.sum().item()
            sparse = (sparse != 0).float()
            if self.sparse_neurons:
                neuron_mask = self.neuron_mask.mask()
                if len(neuron_mask.shape) == 5:
                    neuron_mask.squeeze_(0)
                sparse = sparse * neuron_mask
            dense = torch.ones_like(sparse)
            return conn, total, sparse, dense

    def calc_c(self, x: torch.Tensor, prev_layers: List = []):
        # x: [1, C, H, W]
        assert self.conv.stride[0] == self.conv.stride[1]
        stride = self.conv.stride[0]
        assert self.conv.dilation[0] == self.conv.dilation[1] == 1
        assert self.conv.groups == 1
        with torch.no_grad():
            # calc lambda_n for prev layers
            # weight: [C_out, C_in, h, w]
            self.weight_mask.lmbda = x.shape[2] * x.shape[3] / (stride * stride)
            c_prev = self.conv.weight.shape[0] * self.conv.weight.shape[2] * self.conv.weight.shape[
                3] / (stride * stride)
            for layer in prev_layers:
                layer: ConvBlock
                # reshape or flatten
                layer.neuron_mask.lmbda += c_prev

            y = torch.nn.functional.conv2d(x, self.conv.weight, None, self.conv.stride,
                                           self.conv.padding, self.conv.dilation, self.conv.groups)
            return torch.zeros_like(y)
