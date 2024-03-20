# -*- coding: utf-8 -*-

"""ShuffleNetV2 in PyTorch.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["shufflenetv2"]


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5,batchn=False):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        if(batchn):
            self.bn1 = nn.BatchNorm2d(in_channels)
        else:
            self.bn1 = nn.GroupNorm(4,in_channels) 
        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        if(batchn):
            self.bn2 = nn.BatchNorm2d(in_channels)
        else:
            self.bn2 = nn.GroupNorm(4,in_channels) 
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        if(batchn):
            self.bn3 = nn.BatchNorm2d(in_channels)
        else:
            self.bn3 = nn.GroupNorm(4,in_channels)        
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels,batchn):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        if(batchn):
            self.bn1 = nn.BatchNorm2d(in_channels)
        else:
            self.bn1 = nn.GroupNorm(4,in_channels)

        self.conv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        if(batchn):
            self.bn2 = nn.BatchNorm2d(mid_channels)
        else:
            self.bn2 = nn.GroupNorm(4,mid_channels)  
        # right
        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        if(batchn):
            self.bn3 = nn.BatchNorm2d(mid_channels)
        else:
            self.bn3 = nn.GroupNorm(4,mid_channels)  
        self.conv4 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=mid_channels,
            bias=False,
        )
        if(batchn):
            self.bn4 = nn.BatchNorm2d(mid_channels)
        else:
            self.bn4 = nn.GroupNorm(4,mid_channels) 
        self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        if(batchn):
            self.bn5 = nn.BatchNorm2d(mid_channels)
        else:
            self.bn5 = nn.GroupNorm(4,mid_channels) 

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)))
        # right
        out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))
        # concat
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, net_size, num_classes, batchn=False):
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]["out_channels"]
        num_blocks = configs[net_size]["num_blocks"]

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
        if(batchn):
            self.bn1 = nn.BatchNorm2d(24)
        else:
            self.bn1 = nn.GroupNorm(4,24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0],batchn)
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1],batchn)
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2],batchn)
        self.conv2 = nn.Conv2d(
            out_channels[2],
            out_channels[3],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        if(batchn):
            self.bn2 = nn.BatchNorm2d(out_channels[3])
        else:
            self.bn2 = nn.GroupNorm(4,out_channels[3])
        self.linear = nn.Linear(out_channels[3], num_classes)

        self.features_dim = out_channels[3]

    def _make_layer(self, out_channels, num_blocks,batchn):
        layers = [DownBlock(self.in_channels, out_channels,batchn)]
        for _ in range(num_blocks):
            layers.append(BasicBlock(out_channels,batchn=batchn))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.max_pool2d(out, 3, stride=2, padding=1)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = F.relu(self.bn2(self.conv2(out3)))
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        out = self.linear(features)

        return out,features


configs = {
    "0.5": {"out_channels": (48, 96, 192, 1024), "num_blocks": (3, 7, 3)},
    "1": {"out_channels": (116, 232, 464, 1024), "num_blocks": (3, 7, 3)},
    "1.5": {"out_channels": (176, 352, 704, 1024), "num_blocks": (3, 7, 3)},
    "2": {"out_channels": (224, 488, 976, 2048), "num_blocks": (3, 7, 3)},
}


def shufflenetv2(feature_maps,input_shape,num_classes,batchn):
       
    return ShuffleNetV2(net_size="1",num_classes=num_classes,batchn=batchn)


if __name__ == "__main__":
    net = shufflenetv2(0,0,10,False)
    x = torch.randn(2, 3, 32, 32)
    y,_ = net(x)
    print(y.shape)
