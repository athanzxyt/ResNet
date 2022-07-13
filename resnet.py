"""
Implementation of
K. He, X. Zhang, S. Ren, and J. Sun.
Deep Residual Learning for Image Recognition.
CVPR, 2016.
in PyTorch

Author: Athan Zhang @athanzxyt
https://github.com/athanzxyt 
"""

import torch
import torch.nn as nn

class block_2layer(nn.Module):
    # Base Block used for Shallow Architectures
    # ResNet18 and ResNet34
    def __init__(self,
                 in_channels,
                 out_channels,
                 identity_shortcut=None,
                 stride=1):
        super(block_2layer, self).__init__()
        self.expansion = 1

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity_shortcut = identity_shortcut

    def forward(self, x):
        identity = x
        x = self.layers(x)
        if self.identity_shortcut:
            residual = self.identity_shortcut(identity)

        x += residual
        x = nn.ReLU(x)
        return x

class block_3layer(nn.Module):
    # Bottleneck Block used for Deep Architectures
    # ResNet50 and beyond
    def __init__(self,
                 in_channels,
                 out_channels,
                 identity_shortcut=None,
                 stride=1):
        super(block_3layer, self).__init__()
        self.expansion = 4

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels*self.expansion)
        )
        self.identity_shortcut = identity_shortcut
    
    def forward(self, x):
        identity = x
        x = self.layers(x)
        if self.identity_shortcut:
            residual = self.identity_shortcut(identity)

        x += residual
        x = nn.ReLU(x)
        return x

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 channels,
                 num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.initial_layers = nn.Sequential(
            nn.Conv2d(channels, self.in_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.resnet_layers = nn.Sequential(
            self._make_layer(block, layers[0], 64, stride=1),
            self._make_layer(block, layers[1], 128, stride=2),
            self._make_layer(block, layers[2], 256, stride=2),
            self._make_layer(block, layers[3], 512, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.resnet_layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_blocks, out_channels, stride):
        identity_shortcut = None
        layers = []

        if stride != 1 or self.in_channels != self.out_channels*block.expansion:
            identity_shortcut = nn.Sequentual(
                nn.Conv2d(self.in_channels, out_channels*block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.out_channels*block.expansion)
            )
        
        layers.append(block(self.in_channels, out_channels, identity_shortcut, stride))
        self.in_channels = out_channels*block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

def ResNet18(channels=3, num_classes=1000):
    return ResNet(block_2layer, [2,2,2,2], channels, num_classes)

def ResNet34(channels=3, num_classes=1000):
    return ResNet(block_2layer, [3,4,6,3], channels, num_classes)

def ResNet50(channels=3, num_classes=1000):
    return ResNet(block_3layer, [3,4,6,3], channels, num_classes)

def ResNet101(channels=3, num_classes=100):
    return ResNet(block_3layer, [3,4,23,3], channels, num_classes)

def ResNet152(channels=3, num_classes=1000):
    return ResNet(block_3layer, [3,8,36,3], channels, num_classes)
