import math
from torch import nn
from models.submodules.sparse import ConvBlock
from typing import List


class TemperatureScheduler:
    def __init__(self, model: nn.Module, init_temp: float, final_temp: float, T0: int,
                 Tmax: int) -> None:
        assert init_temp > 0
        assert init_temp <= final_temp
        assert T0 < Tmax
        self.conv_blocks: List[ConvBlock] = []
        for m in model.modules():
            if isinstance(m, ConvBlock):
                self.conv_blocks.append(m)
                m.set_temp(init_temp)
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.T0 = T0
        self.Tmax = Tmax
        self.current_step = 0
        self.factor = math.log(self.final_temp / self.init_temp)

    def _temp(self):
        if self.current_step < self.T0:
            return self.init_temp
        if self.current_step > self.Tmax:
            return self.final_temp
        return self.init_temp * math.exp(self.factor * (self.current_step - self.T0) /
                                         (self.Tmax - self.T0))

    def step(self):
        self.current_step = self.current_step + 1
        temp = self._temp()
        for block in self.conv_blocks:
            block.set_temp(temp)
        return

    def __str__(self) -> str:
        return 'temperature: {:.3e}'.format(self._temp())


class SplitTemperatureScheduler:
    def __init__(self, model: nn.Module, init_temp_w: float, init_temp_n: float,
                 final_temp_w: float, final_temp_n: float, T0: int, Tmax: int) -> None:
        assert init_temp_w > 0 and init_temp_n > 0
        assert init_temp_w <= final_temp_w and init_temp_n <= final_temp_n
        assert T0 < Tmax
        self.conv_blocks: List[ConvBlock] = []
        for m in model.modules():
            if isinstance(m, ConvBlock):
                self.conv_blocks.append(m)
                m.set_weight_temp(init_temp_w)
                m.set_neuron_temp(init_temp_n)
        self.init_temp_w = init_temp_w
        self.final_temp_w = final_temp_w
        self.init_temp_n = init_temp_n
        self.final_temp_n = final_temp_n
        self.T0 = T0
        self.Tmax = Tmax
        self.current_step = 0
        self.factor_w = math.log(self.final_temp_w / self.init_temp_w)
        self.factor_n = math.log(self.final_temp_n / self.init_temp_n)

    def _temp_w(self):
        if self.current_step < self.T0:
            return self.init_temp_w
        if self.current_step > self.Tmax:
            return self.final_temp_w
        return self.init_temp_w * math.exp(self.factor_w * (self.current_step - self.T0) /
                                           (self.Tmax - self.T0))

    def _temp_n(self):
        if self.current_step < self.T0:
            return self.init_temp_n
        if self.current_step > self.Tmax:
            return self.final_temp_n
        return self.init_temp_n * math.exp(self.factor_n * (self.current_step - self.T0) /
                                           (self.Tmax - self.T0))

    def step(self):
        self.current_step = self.current_step + 1
        temp_w = self._temp_w()
        temp_n = self._temp_n()
        for block in self.conv_blocks:
            block.set_weight_temp(temp_w)
            block.set_neuron_temp(temp_n)
        return

    def __str__(self) -> str:
        return 'temperature: weight: {:.3e}, neuron: {:.3e}'.format(self._temp_w(), self._temp_n())
