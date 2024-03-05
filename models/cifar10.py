import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from .submodules.sparse import ConvBlock
from .submodules.layers import conv1x1, conv3x3, create_mslif

__all__ = ['Cifar10Net', 'Cifar10ADMMNet', 'Cifar10ANNNet']


class Cifar10Net(nn.Module):
    def __init__(self, T=8, base_width=256, num_classes=10):
        super().__init__()
        self.skip = ['static_conv']

        self.T = T
        self.static_conv = ConvBlock(conv3x3(3, base_width), nn.BatchNorm2d(base_width),
                                     create_mslif(), static=True, T=T, sparse_weights=True,
                                     sparse_neurons=False)
        self.conv1 = ConvBlock(conv3x3(base_width, base_width), nn.BatchNorm2d(base_width),
                               create_mslif(), sparse_weights=True, sparse_neurons=True)
        self.conv2 = ConvBlock(conv3x3(base_width, base_width), nn.BatchNorm2d(base_width),
                               create_mslif(), sparse_weights=True, sparse_neurons=True)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(conv3x3(base_width, base_width), nn.BatchNorm2d(base_width),
                               create_mslif(), sparse_weights=True, sparse_neurons=True)
        self.conv4 = ConvBlock(conv3x3(base_width, base_width), nn.BatchNorm2d(base_width),
                               create_mslif(), sparse_weights=True, sparse_neurons=True)
        self.conv5 = ConvBlock(conv3x3(base_width, base_width), nn.BatchNorm2d(base_width),
                               create_mslif(), sparse_weights=True, sparse_neurons=True)
        self.maxpool5 = nn.MaxPool2d(2, 2)
        self.dp1 = layer.MultiStepDropout(0.5)
        self.fc1 = ConvBlock(conv1x1(base_width * 8 * 8, (base_width // 2) * 4 * 4), None,
                             create_mslif(), sparse_weights=True, sparse_neurons=False)
        self.dp2 = layer.MultiStepDropout(0.5)
        self.fc2 = ConvBlock(conv1x1((base_width // 2) * 4 * 4, num_classes * 10), None,
                             create_mslif(), sparse_weights=True, sparse_neurons=False)
        self.boost = nn.AvgPool1d(10, 10)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def connects(self):
        conn, total = 0, 0
        with torch.no_grad():
            sparse = torch.ones(1, 3, 32, 32, device='cuda')
            dense = torch.ones(1, 3, 32, 32, device='cuda')
            c, t, sparse, dense = self.static_conv.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv2.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.maxpool2(sparse), self.maxpool2(dense)
            c, t, sparse, dense = self.conv3.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv4.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv5.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.maxpool5(sparse), self.maxpool5(dense)
            sparse, dense = sparse.view(1, -1, 1, 1), dense.view(1, -1, 1, 1)
            c, t, sparse, dense = self.fc1.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.fc2.connects(sparse, dense)
            conn, total = conn + c, total + t
        return conn, total

    def calc_c(self):
        with torch.no_grad():
            x = torch.ones(1, 3, 32, 32, device='cuda')
            x = self.static_conv.calc_c(x)
            x = self.conv1.calc_c(x, [self.static_conv])
            x = self.conv2.calc_c(x, [self.conv1])
            x = self.maxpool2(x)
            x = self.conv3.calc_c(x, [self.conv2])
            x = self.conv4.calc_c(x, [self.conv3])
            x = self.conv5.calc_c(x, [self.conv4])
            x = self.maxpool5(x)
            x = x.view(1, -1, 1, 1)
            x = self.fc1.calc_c(x, [self.conv5])
            x = self.fc2.calc_c(x, [self.fc1])

    def forward(self, x):
        x = self.static_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = functional.seq_to_ann_forward(x, self.maxpool2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = functional.seq_to_ann_forward(x, self.maxpool5)
        x = x.view(x.shape[0], x.shape[1], -1, 1, 1)
        #### [T, N, C, H, W] -> [T, N, CxHxW, 1, 1]
        #x = self.dp1(x)
        x = self.fc1(x)
        #x = self.dp2(x)
        x = self.fc2(x)
        x = x.flatten(2)

        x = x.unsqueeze(2)
        #### [T, N, L] -> [T, N, C=1, L]
        out = functional.seq_to_ann_forward(x, self.boost).squeeze(2)

        return out


class Cifar10Net2(Cifar10Net):
    def __init__(self, T=8, base_width=256, num_classes=10):
        super().__init__(T, base_width, num_classes)
        self.fc2 = ConvBlock(conv1x1((base_width // 2) * 4 * 4, num_classes * 10), None, None,
                             sparse_weights=True, sparse_neurons=False)


class Cifar10ANNNet(nn.Module):
    def __init__(self, T=1):
        super().__init__()
        self.skip = ['static_conv']

        self.static_conv = nn.Sequential(conv3x3(3, 256), nn.BatchNorm2d(256), nn.ReLU())
        self.conv1 = nn.Sequential(conv3x3(256, 256), nn.BatchNorm2d(256), nn.ReLU())
        self.conv2 = nn.Sequential(conv3x3(256, 256), nn.BatchNorm2d(256), nn.ReLU())
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(conv3x3(256, 256), nn.BatchNorm2d(256), nn.ReLU())
        self.conv4 = nn.Sequential(conv3x3(256, 256), nn.BatchNorm2d(256), nn.ReLU())
        self.conv5 = nn.Sequential(conv3x3(256, 256), nn.BatchNorm2d(256), nn.ReLU())
        self.maxpool5 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 128 * 4 * 4, bias=False)
        self.fc2 = nn.Linear(128 * 4 * 4, 100, bias=False)
        self.boost = nn.AvgPool1d(10, 10)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def connects(self):
        return 0, 0

    def calc_c(self):
        pass

    def forward(self, x):
        x = self.static_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.unsqueeze(1)
        #### [N, L] -> [N, C=1, L]
        out = self.boost(x).squeeze(1).unsqueeze(0)

        return out


class Cifar10ADMMNet(nn.Module):
    def __init__(self, T=8, base_width=256, num_classes=10):
        super().__init__()
        self.skip = ['static_conv']

        self.T = T
        self.static_conv = ConvBlock(conv3x3(3, 128), nn.BatchNorm2d(128), create_mslif(),
                                     static=True, T=T, sparse_weights=False, sparse_neurons=True)
        self.conv1 = ConvBlock(conv3x3(128, 256, 2), nn.BatchNorm2d(256), create_mslif(),
                               sparse_weights=True, sparse_neurons=True)
        self.conv2 = ConvBlock(conv3x3(256, 256), nn.BatchNorm2d(256), create_mslif(),
                               sparse_weights=True, sparse_neurons=True)
        self.conv3 = ConvBlock(conv3x3(256, 512, 2), nn.BatchNorm2d(512), create_mslif(),
                               sparse_weights=True, sparse_neurons=True)
        self.conv4 = ConvBlock(conv3x3(512, 512), nn.BatchNorm2d(512), create_mslif(),
                               sparse_weights=True, sparse_neurons=True)
        self.conv5 = ConvBlock(conv3x3(512, 1024), nn.BatchNorm2d(1024), create_mslif(),
                               sparse_weights=True, sparse_neurons=True)
        self.conv6 = ConvBlock(conv3x3(1024, 2048, 2), nn.BatchNorm2d(2048), create_mslif(),
                               sparse_weights=True, sparse_neurons=True)
        self.fc1 = ConvBlock(conv1x1(2048 * 4 * 4, 1024), None, create_mslif(), sparse_weights=True,
                             sparse_neurons=False)
        self.fc2 = ConvBlock(conv1x1(1024, 512), None, create_mslif(), sparse_weights=True,
                             sparse_neurons=False)
        self.fc3 = ConvBlock(conv1x1(512, 100), None, create_mslif(), sparse_weights=True,
                             sparse_neurons=False)
        self.boost = nn.AvgPool1d(10, 10)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def connects(self):
        conn, total = 0, 0
        with torch.no_grad():
            sparse = torch.ones(1, 3, 32, 32, device='cuda')
            dense = torch.ones(1, 3, 32, 32, device='cuda')
            c, t, sparse, dense = self.static_conv.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv2.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv3.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv4.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv5.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv6.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = sparse.view(1, -1, 1, 1), dense.view(1, -1, 1, 1)
            c, t, sparse, dense = self.fc1.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.fc2.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.fc3.connects(sparse, dense)
            conn, total = conn + c, total + t
        return conn, total

    def calc_c(self):
        raise NotImplementedError

    def forward(self, x):
        x = self.static_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.shape[0], x.shape[1], -1, 1, 1)
        #### [T, N, C, H, W] -> [T, N, CxHxW, 1, 1]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.flatten(2)

        x = x.unsqueeze(2)
        #### [T, N, L] -> [T, N, C=1, L]
        out = functional.seq_to_ann_forward(x, self.boost).squeeze(2)

        return out
