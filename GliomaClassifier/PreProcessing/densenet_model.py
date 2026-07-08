import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size, dropout_prob):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ("norm1", nn.BatchNorm3d(in_channels)),
            ("relu1", nn.ReLU(inplace=True)),
            ("conv1", nn.Conv3d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
            ("norm2", nn.BatchNorm3d(bn_size * growth_rate)),
            ("relu2", nn.ReLU(inplace=True)),
            ("conv2", nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))
        self.dropout_prob = dropout_prob

    def forward(self, x):
        new_features = self.layers(x)
        if self.dropout_prob > 0:
            new_features = F.dropout(new_features, p=self.dropout_prob, training=self.training)
        return torch.cat([x, new_features], dim=1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, dropout_prob):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size,
                dropout_prob
            )
            self.add_module(f"denselayer{i + 1}", layer)


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(OrderedDict([
            ("norm", nn.BatchNorm3d(in_channels)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv", nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)),
            ("pool", nn.AvgPool3d(kernel_size=2, stride=2)),
        ]))


class DenseNet1213D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        bn_size=4,
        dropout_prob=0.0,
    ):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv3d(
                in_channels,
                init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )),
            ("norm0", nn.BatchNorm3d(init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = init_features

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(
                    in_channels=num_features,
                    out_channels=num_features // 2
                )
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm3d(num_features))

        self.class_layers = nn.Sequential(OrderedDict([
            ("relu", nn.ReLU(inplace=True)),
            ("pool", nn.AdaptiveAvgPool3d(1)),
            ("flatten", nn.Flatten(1)),
            ("out", nn.Linear(num_features, out_channels)),
        ]))

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.class_layers(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


def densenet121_3d(in_channels=1, out_channels=2):
    return DenseNet1213D(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
    )