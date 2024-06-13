import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, growth_rate * 4, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate * 4)
        self.conv2 = nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)
        return out

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        img_channels = 1
        num_classes = 22
        growth_rate = 16
        num_init_features = 64

        self.init_conv = nn.Sequential(
            nn.Conv2d(img_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_features = num_init_features
        self.dense_block1, num_features = self._make_dense_block(num_features, growth_rate, 6)
        self.trans_layer1, num_features = self._make_transition_layer(num_features)

        self.dense_block2, num_features = self._make_dense_block(num_features, growth_rate, 12)
        self.trans_layer2, num_features = self._make_transition_layer(num_features)

        self.dense_block3, num_features = self._make_dense_block(num_features, growth_rate, 24)
        self.trans_layer3, num_features = self._make_transition_layer(num_features)

        self.dense_block4, num_features = self._make_dense_block(num_features, growth_rate, 16)

        self.final_bn = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(num_features * 7 * 7, num_classes)

    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        layers = []
        num_features = in_channels
        for i in range(num_layers):
            layers.append(DenseLayer(num_features, growth_rate))
            num_features += growth_rate
        return nn.Sequential(*layers), num_features

    def _make_transition_layer(self, in_channels):
        out_channels = in_channels // 2
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ), out_channels

    def forward(self, x):
        x = self.init_conv(x)
        x = self.dense_block1(x)
        x = self.trans_layer1(x)
        x = self.dense_block2(x)
        x = self.trans_layer2(x)
        x = self.dense_block3(x)
        x = self.trans_layer3(x)
        x = self.dense_block4(x)
        x = self.final_bn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.classifier(x)
        return x