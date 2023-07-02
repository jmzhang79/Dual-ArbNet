
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
from argparse import Namespace
import random 
import math

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
 
        assert  kernel_size in (3,7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, inplanes, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(inplanes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x*self.ca(x)
        result = out*self.sa(out)
        return result

class Res_Conv(nn.Module):
    def __init__(self, inChannels,  kSize=3):
        super(Res_Conv, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(inChannels, inChannels, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])
        
    def forward(self, x):
        out = self.conv(x)
        return out + x

class ResBlock(nn.Module):
    def __init__(self, channels, nConvLayers, kSize=3):
        super(ResBlock, self).__init__()

        convs = [Res_Conv(channels) for i in range(nConvLayers)]
        self.convs = nn.Sequential(*convs)
        self.CBAM = CBAM(channels)


    def forward(self, x):
        return self.CBAM(self.convs(x)) + x