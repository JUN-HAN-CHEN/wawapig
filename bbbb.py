import os
import time
import copy
import torch
import torchvision
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:width_mult=1.4
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



def model_parser(model, fixed_weight=False, dropout_rate=0.0, bayesian = False):
    base_model = None

    if model == 'Googlenet':
        base_model = models.inception_v3(pretrained=True)
        network = GoogleNet(base_model, fixed_weight, dropout_rate)
    elif model == 'Resnet':
        base_model = models.resnet34(pretrained=True)
        network = ResNet(base_model, fixed_weight, dropout_rate, bayesian)
    elif model == 'ResnetSimple':
        base_model = models.resnet34(pretrained=True)
        network = ResNetSimple(base_model, fixed_weight)
    #TODO    
    elif model == 'MobileNetV2':
        base_model = models.mobilenet_v2( width_mult=1.4)
        # print(base_model)
        network = MobileNetV2(base_model, fixed_weight)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
   
    else:
        assert 'Unvalid Model'
    return network

class MobileNetV2(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.5, bayesian = True):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        norm_layer = nn.BatchNorm2d

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        feat_in = base_model.classifier[1].in_features

        # self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # self.base_model = base_model
        model = []
        hank = [1]
        for i in range(18):
            model.append(base_model.features[i])
            # if i in hank:
            #     model.append(base_model.features[i])
        # model.append(base_model.features[18])
        model.append(block(int(320*1.4), int(640*1.4), 2, expand_ratio=6, norm_layer=norm_layer))
        # model.append(block(int(640*1.4), int(960*1.4), 1, expand_ratio=6, norm_layer=norm_layer))
        model.append(ConvBNReLU(int(640*1.4), int(1280*1.4), kernel_size=1, norm_layer=norm_layer))
        self.base_model = nn.Sequential(*model)
        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_last = nn.Linear(feat_in, 1024, bias=True)
        # self.fc_middle = nn.Linear(2048, 1024, bias=True)
        self.fc_position = nn.Linear(1024, 3, bias=True)
        self.fc_rotation = nn.Linear(1024, 4, bias=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        init_modules = [self.fc_last, self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.base_model(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        fv= self.fc_last(x)
        x = F.relu(fv)
        # x= self.fc_middle(x)
        # x = F.relu(x)
        # dropout_on = self.training or self.bayesian
        # print(self.dropout_rate, dropout_on)
        # if self.dropout_rate > 0:
        #     x = F.dropout(x, p=self.dropout_rate, training=dropout_on)
        x = self.dropout(x)
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, fv
model = model_parser('MobileNetV2', False, 0.5, True)
print(model)
summary(model.cuda(), input_size=(3, 256, 256))
