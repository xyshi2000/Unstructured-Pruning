import torch
from torch import Tensor, nn
from models.submodules.sparse import ConvBlock, Mask
from typing import List
import math


class PenaltyTerm(nn.Module):
    def __init__(self, model: nn.Module, lmbda: float) -> None:
        super(PenaltyTerm, self).__init__()
        self.layers: List[ConvBlock] = []
        self.model = model
        for m in model.modules():
            if isinstance(m, ConvBlock):
                self.layers.append(m)
        self.lmbda = lmbda
        model.calc_c()

    def forward(self) -> Tensor:
        loss = 0
        for layer in self.layers:
            if layer.sparse_neurons:
                loss = loss + (self.lmbda * layer.neuron_mask.lmbda) * (torch.sigmoid(
                    layer.neuron_mask.mask_value * layer.neuron_mask.temp)).sum()
            if layer.sparse_weights:
                loss = loss + (self.lmbda * layer.weight_mask.lmbda) * (torch.sigmoid(
                    layer.weight_mask.mask_value * layer.weight_mask.temp)).sum()
        return loss
