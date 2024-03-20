# -*- coding: utf-8 -*-

"""MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["mobilenetv2"]


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(self, in_planes, out_planes, expansion, stride,batchn):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        if(batchn):
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.GroupNorm(4,planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
            bias=False,
        )
        if(batchn):
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn2 = nn.GroupNorm(4,planes)

        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        if(batchn):
            self.bn3 = nn.BatchNorm2d(out_planes)
        else:
            self.bn3 = nn.GroupNorm(4,out_planes)

        self.shortcut = nn.Sequential()

        if(batchn):
            norm = nn.BatchNorm2d(out_planes)
        else:
            norm = nn.GroupNorm(4,out_planes)


        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                norm,
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self,feature_maps,input_shape,num_classes,batchn=False):
        super(MobileNetV2, self).__init__()

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        if(batchn): 
            self.bn1 = nn.BatchNorm2d(32)
        else:
            self.bn1 = nn.GroupNorm(4,32)
            
        self.layers = self._make_layers(in_planes=32,batchn=batchn)

        self.conv2 = nn.Conv2d(
            320, 1280, kernel_size=1, stride=1, padding=0, bias=False
        )
        if(batchn):
            self.bn2 = nn.BatchNorm2d(1280)
        else:
            self.bn2 = nn.GroupNorm(4,1280)
            
        self.linear = nn.Linear(1280, num_classes)

        self.features_dim = 1280

    def _make_layers(self, in_planes,batchn):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride,batchn))
                in_planes = out_planes
        return layers

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        for layer in self.layers:
            out = layer(out)

        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        return out,features


def mobilenetv2(feature_maps,input_shape,num_classes,batchn=False):
    return MobileNetV2(feature_maps,input_shape,num_classes,batchn=batchn)


if __name__ == "__main__":
    net = MobileNetV2(0,None,10,False)

    x = torch.randn(1, 3, 32, 32)
    y,_ = net(x)
    print(y.shape)
