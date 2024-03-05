import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from .submodules.sparse import ConvBlock
from .submodules.layers import conv1x1, conv3x3, create_mslif

__all__ = ['VGGSNN']


class VGGSNN(nn.Module):
    def __init__(self):
        super(VGGSNN, self).__init__()
        self.skip = ['conv1']

        self.conv1 = ConvBlock(conv3x3(2, 64), nn.BatchNorm2d(64), create_mslif(),
                               sparse_neurons=False, sparse_weights=True)
        self.conv2 = ConvBlock(conv3x3(64, 128), nn.BatchNorm2d(128), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv3 = ConvBlock(conv3x3(128, 256), nn.BatchNorm2d(256), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.conv4 = ConvBlock(conv3x3(256, 256), nn.BatchNorm2d(256), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv5 = ConvBlock(conv3x3(256, 512), nn.BatchNorm2d(512), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.conv6 = ConvBlock(conv3x3(512, 512), nn.BatchNorm2d(512), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.conv7 = ConvBlock(conv3x3(512, 512), nn.BatchNorm2d(512), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.conv8 = ConvBlock(conv3x3(512, 512), nn.BatchNorm2d(512), create_mslif(),
                               sparse_neurons=True, sparse_weights=True)
        self.pool4 = nn.AvgPool2d(2, 2)

        W = int(48 / 2 / 2 / 2 / 2)

        self.classifier = ConvBlock(conv1x1(512 * W * W, 100), None, None,
                                    sparse_neurons=False, sparse_weights=True)
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
            sparse = torch.ones(1, 2, 48, 48, device='cuda')
            dense = torch.ones(1, 2, 48, 48, device='cuda')
            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv2.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool1(sparse), self.pool1(dense)
            c, t, sparse, dense = self.conv3.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv4.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool2(sparse), self.pool2(dense)
            c, t, sparse, dense = self.conv5.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv6.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool3(sparse), self.pool3(dense)
            c, t, sparse, dense = self.conv7.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv8.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool4(sparse), self.pool4(dense)
            sparse, dense = sparse.view(1, -1, 1, 1), dense.view(1, -1, 1, 1)
            c, t, sparse, dense = self.classifier.connects(sparse, dense)
            conn, total = conn + c, total + t
        return conn, total

    def calc_c(self):
        with torch.no_grad():
            x = torch.ones(1, 2, 48, 48, device='cuda')
            x = self.conv1.calc_c(x)
            x = self.conv2.calc_c(x, [self.conv1])
            x = self.pool1(x)
            x = self.conv3.calc_c(x, [self.conv2])
            x = self.conv4.calc_c(x, [self.conv3])
            x = self.pool2(x)
            x = self.conv5.calc_c(x, [self.conv4])
            x = self.conv6.calc_c(x, [self.conv5])
            x = self.pool3(x)
            x = self.conv7.calc_c(x, [self.conv6])
            x = self.conv8.calc_c(x, [self.conv7])
            x = self.pool4(x)
            x = x.view(1, -1, 1, 1)
            x = self.classifier.calc_c(x, [self.conv8])
            return

    def forward(self, x: torch.Tensor):
        x = x.transpose(0, 1)
        #### [N, T, C, H, W] -> [T, N, C, H, W]
        x = self.conv1(x)
        x = self.conv2(x)
        x = functional.seq_to_ann_forward(x, self.pool1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = functional.seq_to_ann_forward(x, self.pool2)
        x = self.conv5(x)
        x = self.conv6(x)
        x = functional.seq_to_ann_forward(x, self.pool3)
        x = self.conv7(x)
        x = self.conv8(x)
        x = functional.seq_to_ann_forward(x, self.pool4)
        x = x.view(x.shape[0], x.shape[1], -1, 1, 1)
        x = self.classifier(x)
        x = x.flatten(2).unsqueeze(2)
        #### [T, N, L] -> [T, N, C=1, L]
        out = functional.seq_to_ann_forward(x, self.boost).squeeze(2)
        return out
