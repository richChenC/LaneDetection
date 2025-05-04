from 车道线检测.model.model import parsingNet

import torch
import torch.nn as nn
import numpy as np

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def initialize_weights(*models):
    for model in models:
        real_init_weights(model)

def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unknown module', m)

# 你需要有resnet实现，建议直接用torchvision.models.resnet18/34/50
from torchvision.models import resnet18, resnet34, resnet50

def resnet(backbone='18', pretrained=True):
    if backbone == '18':
        return resnet18(pretrained=pretrained)
    elif backbone == '34':
        return resnet34(pretrained=pretrained)
    elif backbone == '50':
        return resnet50(pretrained=pretrained)
    else:
        raise ValueError('Unsupported backbone')

class parsingNet(nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='18', cls_dim=(37, 10, 4), use_aux=False):
        super(parsingNet, self).__init__()
        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim  # (num_gridding, num_cls_per_lane, num_of_lanes)
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)
        self.model = resnet(backbone, pretrained=pretrained)
        if self.use_aux:
            self.aux_header2 = nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_header3 = nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_header4 = nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_combine = nn.Sequential(
                conv_bn_relu(384, 256, 3, padding=2, dilation=2),
                conv_bn_relu(256, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=4, dilation=4),
                nn.Conv2d(128, cls_dim[-1] + 1, 1)
            )
            initialize_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)
        self.cls = nn.Sequential(
            nn.Linear(1800, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.total_dim),
        )
        self.pool = nn.Conv2d(512, 8, 1) if backbone in ['34', '18'] else nn.Conv2d(2048, 8, 1)
        initialize_weights(self.cls)
    def forward(self, x):
        x2, x3, fea = self.model(x)
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear')
            x4 = self.aux_header4(fea)
            x4 = nn.functional.interpolate(x4, scale_factor=4, mode='bilinear')
            aux_seg = torch.cat([x2, x3, x4], dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None
        fea = self.pool(fea).view(-1, 1800)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        if self.use_aux:
            return group_cls, aux_seg
        return group_cls 