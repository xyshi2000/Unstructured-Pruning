import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer
from spikingjelly.clock_driven import surrogate, neuron, functional
from .submodules.sparse import ConvBlock
from .submodules.layers import conv1x1, conv3x3, create_msif, create_mslif
from typing import Union, List

__all__ = [
    'resnet19', 'sew_resnet18_imagenet', 'sew_resnet34_imagenet', 'sew_resnet50_imagenet',
    'sew_resnet101_imagenet', 'sew_resnet152_imagenet', 'sew_resnet18_cifar', 'sew_resnet34_cifar',
    'sew_resnet50_cifar', 'sew_resnet101_cifar', 'sew_resnet152_cifar']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample: ConvBlock = None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, activation=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = create_msif
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.downsample = downsample
        self.conv1 = ConvBlock(conv3x3(inplanes, planes, stride), norm_layer(planes), activation(),
                               sparse_weights=True, sparse_neurons=True)
        self.conv2 = ConvBlock(conv3x3(planes, planes), norm_layer(planes), activation(),
                               sparse_weights=True, sparse_neurons=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        return out

    def connects(self, sparse: torch.Tensor, dense: torch.Tensor):
        conn, total = 0, 0
        id_sparse, id_dense = sparse, dense
        with torch.no_grad():
            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv2.connects(sparse, dense)
            conn, total = conn + c, total + t
            if self.downsample is not None:
                c, t, id_sparse, id_dense = self.downsample.connects(id_sparse, id_dense)
                conn, total = conn + c, total + t
            return conn, total, sparse + id_sparse, dense + id_dense

    def calc_c(self, x: torch.Tensor, prev_layers: List[ConvBlock] = []):
        ident = x
        with torch.no_grad():
            x = self.conv1.calc_c(x, prev_layers)
            x = self.conv2.calc_c(x, [self.conv1])
            if self.downsample is not None:
                ident = self.downsample.calc_c(ident, prev_layers)
            return x + ident


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample: ConvBlock = None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, activation=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = create_msif
        width = int(planes * (base_width / 64.)) * groups

        self.downsample = downsample
        self.conv1 = ConvBlock(conv1x1(inplanes, width), norm_layer(width), activation(),
                               sparse_weights=True, sparse_neurons=True)
        self.conv2 = ConvBlock(conv3x3(width, width, stride, groups, dilation), norm_layer(width),
                               activation(), sparse_weights=True, sparse_neurons=True)
        self.conv3 = ConvBlock(conv1x1(width, planes * self.expansion),
                               norm_layer(planes * self.expansion), activation(),
                               sparse_weights=True, sparse_neurons=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        return out

    def connects(self, sparse: torch.Tensor, dense: torch.Tensor):
        conn, total = 0, 0
        id_sparse, id_dense = sparse, dense
        with torch.no_grad():
            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv2.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv3.connects(sparse, dense)
            conn, total = conn + c, total + t
            if self.downsample is not None:
                c, t, id_sparse, id_dense = self.downsample.connects(id_sparse, id_dense)
                conn, total = conn + c, total + t
            return conn, total, sparse + id_sparse, dense + id_dense

    def calc_c(self, prev_layers: List[ConvBlock] = []):
        ident = x
        with torch.no_grad():
            x = self.conv1.calc_c(x, prev_layers)
            x = self.conv2.calc_c(x, [self.conv1])
            x = self.conv3.calc_c(x, [self.conv2])
            if self.downsample is not None:
                ident = self.downsample.calc_c(ident, prev_layers)
            return x + ident


class SEWResNet_ImageNet(nn.Module):
    def __init__(self, block: Union[BasicBlock, Bottleneck], layers: List[int], num_classes=1000,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer: nn.Module = None, T=4):
        super(SEWResNet_ImageNet, self).__init__()
        self.skip = ['conv1']

        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ConvBlock(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes), create_msif(), static=True, T=T, sparse_weights=False,
            sparse_neurons=False)
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        if num_classes * 10 < 512 * block.expansion:
            self.fc = nn.Linear(512 * block.expansion, num_classes * 10)
            self.boost = nn.AvgPool1d(10, 10)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.boost = None

        self.init_weight()
        if zero_init_residual:
            self.zero_init_blocks()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def zero_init_blocks(self):
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.conv3.norm.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.conv2.norm.weight, 0)

    def _make_layer(self, block: Union[BasicBlock, Bottleneck], planes: int, blocks: int, stride=1,
                    dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBlock(conv1x1(self.inplanes, planes * block.expansion, stride),
                                   norm_layer(planes * block.expansion), create_msif(),
                                   sparse_weights=True, sparse_neurons=True)

        layers: List[Union[BasicBlock, Bottleneck]] = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                  previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                      dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        out = functional.seq_to_ann_forward(x, self.fc)

        if self.boost is not None:
            #### [T, N, L] -> [T, N, C=1, L]
            out = out.unsqueeze(2)
            out = functional.seq_to_ann_forward(out, self.boost).squeeze(2)

        return out

    def forward(self, x: torch.Tensor):
        return self._forward_impl(x)

    def connects(self):
        conn, total = 0, 0
        with torch.no_grad():
            # static conv
            sparse = torch.ones(1, 3, 224, 224, device='cuda')
            dense = torch.ones(1, 3, 224, 224, device='cuda')
            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.maxpool[0](sparse), self.maxpool[0](dense)

            def _connects(layer: List[Union[BasicBlock, Bottleneck]], conn: float, total: float,
                          sparse: torch.Tensor, dense: torch.Tensor):
                for block in layer:
                    c, t, sparse, dense = block.connects(sparse, dense)
                    conn, total = conn + c, total + t
                return conn, total, sparse, dense

            conn, total, sparse, dense = _connects(self.layer1, conn, total, sparse, dense)
            conn, total, sparse, dense = _connects(self.layer2, conn, total, sparse, dense)
            conn, total, sparse, dense = _connects(self.layer3, conn, total, sparse, dense)
            conn, total, sparse, dense = _connects(self.layer4, conn, total, sparse, dense)
            # ignore fc

        return conn, total

    def calc_c(self):
        with torch.no_grad():
            x = torch.ones(1, 3, 224, 224, device='cuda')
            x = self.conv1.calc_c(x)
            x = self.maxpool[0](x)
            prev_layers = [self.conv1]

            def _calc_c(layer: List[Union[BasicBlock, Bottleneck]], x: torch.Tensor,
                        prev_layers: List[ConvBlock]):
                for block in layer:
                    x = block.calc_c(x, prev_layers)
                    if block.downsample is None:
                        if isinstance(block, BasicBlock):
                            prev_layers.append(block.conv2)
                        elif isinstance(block, Bottleneck):
                            prev_layers.append(block.conv3)
                    else:
                        if isinstance(block, BasicBlock):
                            prev_layers = [block.conv2, block.downsample]
                        elif isinstance(block, Bottleneck):
                            prev_layers = [block.conv3, block.downsample]
                return x, prev_layers

            x, prev_layers = _calc_c(self.layer1, x, prev_layers)
            x, prev_layers = _calc_c(self.layer2, x, prev_layers)
            x, prev_layers = _calc_c(self.layer3, x, prev_layers)
            x, prev_layers = _calc_c(self.layer4, x, prev_layers)
            # ignore fc
        return


class SEWResNet_CIFAR(nn.Module):
    def __init__(self, block: Union[BasicBlock, Bottleneck], layers: List[int], num_classes=10,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer: nn.Module = None, T=4):
        super(SEWResNet_CIFAR, self).__init__()
        self.skip = ['conv1']

        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ConvBlock(conv3x3(3, self.inplanes), norm_layer(self.inplanes), create_mslif(),
                               static=True, T=T, sparse_weights=False, sparse_neurons=False)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        if num_classes * 10 < 512 * block.expansion:
            self.fc = nn.Linear(512 * block.expansion, num_classes * 10)
            self.boost = nn.AvgPool1d(10, 10)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.boost = None

        self.init_weight()
        if zero_init_residual:
            self.zero_init_blocks()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def zero_init_blocks(self):
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.conv3.norm.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.conv2.norm.weight, 0)

    def _make_layer(self, block: Union[BasicBlock, Bottleneck], planes: int, blocks: int, stride=1,
                    dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBlock(conv1x1(self.inplanes, planes * block.expansion, stride),
                                   norm_layer(planes * block.expansion), create_msif(),
                                   sparse_weights=True, sparse_neurons=True)

        layers: List[Union[BasicBlock, Bottleneck]] = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                  previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                      dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        out = functional.seq_to_ann_forward(x, self.fc)

        if self.boost is not None:
            #### [T, N, L] -> [T, N, C=1, L]
            out = out.unsqueeze(2)
            out = functional.seq_to_ann_forward(out, self.boost).squeeze(2)

        return out

    def forward(self, x: torch.Tensor):
        return self._forward_impl(x)

    def connects(self):
        conn, total = 0, 0
        with torch.no_grad():
            # static conv
            sparse = torch.ones(1, 3, 32, 32, device='cuda')
            dense = torch.ones(1, 3, 32, 32, device='cuda')
            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t

            def _connects(layer: List[Union[BasicBlock, Bottleneck]], conn: float, total: float,
                          sparse: torch.Tensor, dense: torch.Tensor):
                for block in layer:
                    c, t, sparse, dense = block.connects(sparse, dense)
                    conn, total = conn + c, total + t
                return conn, total, sparse, dense

            conn, total, sparse, dense = _connects(self.layer1, conn, total, sparse, dense)
            conn, total, sparse, dense = _connects(self.layer2, conn, total, sparse, dense)
            conn, total, sparse, dense = _connects(self.layer3, conn, total, sparse, dense)
            conn, total, sparse, dense = _connects(self.layer4, conn, total, sparse, dense)
            # ignore fc

        return conn, total

    def calc_c(self):
        with torch.no_grad():
            x = torch.ones(1, 3, 32, 32, device='cuda')
            x = self.conv1.calc_c(x)
            prev_layers = [self.conv1]

            def _calc_c(layer: List[Union[BasicBlock, Bottleneck]], x: torch.Tensor,
                        prev_layers: List[ConvBlock]):
                for block in layer:
                    x = block.calc_c(x, prev_layers)
                    if block.downsample is None:
                        if isinstance(block, BasicBlock):
                            prev_layers.append(block.conv2)
                        elif isinstance(block, Bottleneck):
                            prev_layers.append(block.conv3)
                    else:
                        if isinstance(block, BasicBlock):
                            prev_layers = [block.conv2, block.downsample]
                        elif isinstance(block, Bottleneck):
                            prev_layers = [block.conv3, block.downsample]
                return x, prev_layers

            x, prev_layers = _calc_c(self.layer1, x, prev_layers)
            x, prev_layers = _calc_c(self.layer2, x, prev_layers)
            x, prev_layers = _calc_c(self.layer3, x, prev_layers)
            x, prev_layers = _calc_c(self.layer4, x, prev_layers)
            # ignore fc
        return


class ResNet19(nn.Module):
    def __init__(self, block: Union[BasicBlock, Bottleneck], layers: List[int], num_classes=10,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer: nn.Module = None, T=2):
        super(ResNet19, self).__init__()
        self.skip = ['conv1']

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ConvBlock(conv3x3(3, self.inplanes), norm_layer(self.inplanes), create_mslif(),
                               static=True, T=T, sparse_weights=False, sparse_neurons=False)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc1 = ConvBlock(conv1x1(512 * block.expansion, 256, bias=True), None, create_mslif(),
                             sparse_neurons=False, sparse_weights=True)
        if num_classes * 10 < 256:
            self.fc2 = ConvBlock(conv1x1(256, num_classes * 10, bias=True), None, None,
                                 sparse_neurons=False, sparse_weights=True)
            self.boost = nn.AvgPool1d(10, 10)
        else:
            self.fc2 = ConvBlock(conv1x1(256, num_classes, bias=True), None, None,
                                 sparse_neurons=False, sparse_weights=True)
            self.boost = None

        self.init_weight()
        if zero_init_residual:
            self.zero_init_blocks()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def zero_init_blocks(self):
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.conv3.norm.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.conv2.norm.weight, 0)

    def _make_layer(self, block: Union[BasicBlock, Bottleneck], planes: int, blocks: int, stride=1,
                    dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBlock(conv1x1(self.inplanes, planes * block.expansion, stride),
                                   norm_layer(planes * block.expansion), create_mslif(),
                                   sparse_weights=True, sparse_neurons=True)

        layers: List[Union[BasicBlock, Bottleneck]] = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                  previous_dilation, norm_layer, create_mslif))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                      dilation=self.dilation, norm_layer=norm_layer, activation=create_mslif))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        out = x.flatten(2)

        if self.boost is not None:
            #### [T, N, L] -> [T, N, C=1, L]
            out = out.unsqueeze(2)
            out = functional.seq_to_ann_forward(out, self.boost).squeeze(2)

        return out

    def forward(self, x: torch.Tensor):
        return self._forward_impl(x)

    def connects(self):
        conn, total = 0, 0
        with torch.no_grad():
            # static conv
            sparse = torch.ones(1, 3, 32, 32, device='cuda')
            dense = torch.ones(1, 3, 32, 32, device='cuda')
            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t

            def _connects(layer: List[Union[BasicBlock, Bottleneck]], conn: float, total: float,
                          sparse: torch.Tensor, dense: torch.Tensor):
                for block in layer:
                    c, t, sparse, dense = block.connects(sparse, dense)
                    conn, total = conn + c, total + t
                return conn, total, sparse, dense

            conn, total, sparse, dense = _connects(self.layer1, conn, total, sparse, dense)
            conn, total, sparse, dense = _connects(self.layer2, conn, total, sparse, dense)
            conn, total, sparse, dense = _connects(self.layer3, conn, total, sparse, dense)

            sparse, dense = self.avgpool(sparse), self.avgpool(dense)
            c, t, sparse, dense = self.fc1.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.fc2.connects(sparse, dense)
            conn, total = conn + c, total + t

        return conn, total

    def calc_c(self):
        with torch.no_grad():
            x = torch.ones(1, 3, 32, 32, device='cuda')
            x = self.conv1.calc_c(x)
            prev_layers = [self.conv1]

            def _calc_c(layer: List[Union[BasicBlock, Bottleneck]], x: torch.Tensor,
                        prev_layers: List[ConvBlock]):
                for block in layer:
                    x = block.calc_c(x, prev_layers)
                    if block.downsample is None:
                        if isinstance(block, BasicBlock):
                            prev_layers.append(block.conv2)
                        elif isinstance(block, Bottleneck):
                            prev_layers.append(block.conv3)
                    else:
                        if isinstance(block, BasicBlock):
                            prev_layers = [block.conv2, block.downsample]
                        elif isinstance(block, Bottleneck):
                            prev_layers = [block.conv3, block.downsample]
                return x, prev_layers

            x, prev_layers = _calc_c(self.layer1, x, prev_layers)
            x, prev_layers = _calc_c(self.layer2, x, prev_layers)
            x, prev_layers = _calc_c(self.layer3, x, prev_layers)

            x = self.avgpool(x.unsqueeze(0)).squeeze(0)
            x = self.fc1.calc_c(x, prev_layers)
            x = self.fc2.calc_c(x, [self.fc1])
        return


def resnet19(**kwargs):
    return ResNet19(BasicBlock, [3, 3, 2], **kwargs)


def _sew_resnet_imagenet(block, layers, **kwargs):
    model = SEWResNet_ImageNet(block, layers, **kwargs)
    return model


def sew_resnet18_imagenet(**kwargs):
    return _sew_resnet_imagenet(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34_imagenet(**kwargs):
    return _sew_resnet_imagenet(BasicBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50_imagenet(**kwargs):
    return _sew_resnet_imagenet(Bottleneck, [3, 4, 6, 3], **kwargs)


def sew_resnet101_imagenet(**kwargs):
    return _sew_resnet_imagenet(Bottleneck, [3, 4, 23, 3], **kwargs)


def sew_resnet152_imagenet(**kwargs):
    return _sew_resnet_imagenet(Bottleneck, [3, 8, 36, 3], **kwargs)


def _sew_resnet_cifar(block, layers, **kwargs):
    model = SEWResNet_CIFAR(block, layers, **kwargs)
    return model


def sew_resnet18_cifar(**kwargs):
    return _sew_resnet_cifar(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34_cifar(**kwargs):
    return _sew_resnet_cifar(BasicBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50_cifar(**kwargs):
    return _sew_resnet_cifar(Bottleneck, [3, 4, 6, 3], **kwargs)


def sew_resnet101_cifar(**kwargs):
    return _sew_resnet_cifar(Bottleneck, [3, 4, 23, 3], **kwargs)


def sew_resnet152_cifar(**kwargs):
    return _sew_resnet_cifar(Bottleneck, [3, 8, 36, 3], **kwargs)
