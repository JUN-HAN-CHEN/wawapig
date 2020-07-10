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
import ipdb

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
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
        # base_model = models.mobilenet_v2(pretrained=True)
        base_model = models.mobilenet_v2(width_mult=1.6)
        print('fuck')
        network = MobileNetV2(base_model, fixed_weight)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    elif model == "Densenet":
        base_model = models.densenet121(pretrained=True)
        network = Densenet(base_model, fixed_weight, dropout_rate, bayesian)       
    else:
        assert 'Unvalid Model'

    return network


class PoseLoss(nn.Module):
    def __init__(self, device, sx=0.0, sq=0.0, learn_beta=False):
        super(PoseLoss, self).__init__()
        self.learn_beta = learn_beta

        if not self.learn_beta:
            self.sx = 0
            self.sq = -6.25
            
        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)

        # if learn_beta:
        #     self.sx.requires_grad = True
        #     self.sq.requires_grad = True
        #
        # self.sx = self.sx.to(device)
        # self.sq = self.sq.to(device)

        self.loss_print = None

    def forward(self, pred_x, pred_q, target_x, target_q):
        pred_q = F.normalize(pred_q, p=2, dim=1)
        loss_x = F.l1_loss(pred_x, target_x, reduction="sum")
        loss_q = F.l1_loss(pred_q, target_q, reduction='sum')
        # loss_x = F.mse_loss(pred_x, target_x)
        # loss_q = F.mse_loss(pred_q, target_q)

            
        loss = torch.exp(-self.sx)*loss_x \
               + self.sx \
               + torch.exp(-self.sq)*loss_q \
               + self.sq

        self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

        return loss, loss_x.item(), loss_q.item()
# class MobileNetV22(nn.Module):
#     def __init__(self, base_model, fixed_weight=False, dropout_rate=0.5, bayesian = True):
#         super(MobileNetV22, self).__init__()
#         block = InvertedResidual
#         norm_layer = nn.BatchNorm2d

#         self.bayesian = bayesian
#         self.dropout_rate = dropout_rate
#         feat_in = base_model.classifier[1].in_features

#         self.base_model = nn.Sequential(*list(base_model.children())[:-1])
#         # self.base_model = base_model

#         if fixed_weight:
#             for param in self.base_model.parameters():
#                 param.requires_grad = False

#         self.fc_last = nn.Linear(feat_in, 2048, bias=True)
#         self.fc_middle = nn.Linear(2048, 1024, bias=True)
#         self.fc_position = nn.Linear(2048, 3, bias=True)
#         self.fc_rotation = nn.Linear(2048, 4, bias=True)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         init_modules = [self.fc_last, self.fc_position, self.fc_rotation, self.fc_middle]

#         for module in init_modules:
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#                 nn.init.kaiming_normal_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)
#         self.dropout = nn.Dropout(self.dropout_rate)

#     def forward(self, x):
#         x = self.base_model(x)
#         x = self.gap(x)
#         fv = x.view(x.size(0), -1)
#         x= self.fc_last(fv)
#         x = F.relu(x)
#         # x= self.fc_middle(x)
#         # x = F.relu(x)
#         # dropout_on = self.training or self.bayesian
#         # print(self.dropout_rate, dropout_on)
#         # if self.dropout_rate > 0:
#         #     x = F.dropout(x, p=self.dropout_rate, training=dropout_on)
#         # x = self.dropout(x)
#         x = F.dropout(x, p=0.5, training=True) 
#         position = self.fc_position(x)
#         rotation = self.fc_rotation(x)

#         return position, rotation, fv
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
        for i in range(18):
            model.append(base_model.features[i])
            # model.append(base_model.features[i])
        model.append(base_model.features[18])
        # model.append(block(int(320*1.6), int(640*1.6), 2, expand_ratio=6, norm_layer=norm_layer))
        # model.append(block(int(640*1.6), int(960*1.6), 1, expand_ratio=6, norm_layer=norm_layer))
        # model.append(ConvBNReLU(int(960*1.6), int(1280*1.6), kernel_size=1, norm_layer=norm_layer))
        # model.append(ConvBNReLU(320, 1280, kernel_size=1, norm_layer=norm_layer))
        # model.append(block(320, 640, 2, expand_ratio=6, norm_layer=norm_layer))
        # model.append(block(640, 960, 1, expand_ratio=6, norm_layer=norm_layer))
        # model.append(ConvBNReLU(960, 1280, kernel_size=1, norm_layer=norm_layer))
        self.base_model = nn.Sequential(*model)
        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False
        abc = 4096
        self.fc_last = nn.Linear(feat_in, abc, bias=True)
        self.fc_middle = nn.Linear(abc, int(abc/2), bias=True)
        self.fc_middle2 = nn.Linear(int(abc/2), int(abc/4), bias=True)
        self.fc_position = nn.Linear(int(abc/4), 3, bias=True)
        self.fc_rotation = nn.Linear(int(abc/4), 4, bias=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        init_modules = [self.fc_last, self.fc_position, self.fc_rotation, self.fc_middle, self.fc_middle2]

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
        x= self.fc_last(x)
        x = F.relu(x)
        x= self.fc_middle(x)
        x = F.relu(x)
        fv = self.fc_middle2(x)
        x = F.relu(fv)
        # dropout_on = self.training or self.bayesian
        # print(self.dropout_rate, dropout_on)
        # if self.dropout_rate > 0:
        #     x = F.dropout(x, p=self.dropout_rate, training=dropout_on)
        x = self.dropout(x)
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, fv        
# class MobileNetV2(nn.Module):
#     def __init__(self, base_model, fixed_weight=False, dropout_rate=0.5, bayesian = True):
#         super(MobileNetV2, self).__init__()
#         block = InvertedResidual
#         norm_layer = nn.BatchNorm2d

#         self.bayesian = bayesian
#         self.dropout_rate = dropout_rate
#         feat_in = base_model.classifier[1].in_features

#         self.base_model = nn.Sequential(*list(base_model.children())[:-1])
#         # self.base_model = base_model

#         if fixed_weight:
#             for param in self.base_model.parameters():
#                 param.requires_grad = False

#         self.fc_last = nn.Linear(feat_in, 2048, bias=True)
#         self.fc_middle = nn.Linear(2048, 1024, bias=True)
#         self.fc_position = nn.Linear(2048, 3, bias=True)
#         self.fc_rotation = nn.Linear(2048, 4, bias=True)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         init_modules = [self.fc_last, self.fc_position, self.fc_rotation, self.fc_middle]

#         for module in init_modules:
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#                 nn.init.kaiming_normal_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)
#         self.dropout = nn.Dropout(self.dropout_rate)

#     def forward(self, x):
#         x = self.base_model(x)
#         x = self.gap(x)
#         x = x.view(x.size(0), -1)
#         fv = self.fc_last(x)
#         x = F.relu(x)
#         print('fuck')
#         # x= self.fc_middle(x)
#         # x = F.relu(x)
#         # dropout_on = self.training or self.bayesian
#         # print(self.dropout_rate, dropout_on)
#         # if self.dropout_rate > 0:
#         #     x = F.dropout(x, p=self.dropout_rate, training=dropout_on)
#         # x = self.dropout(x)
#         x = F.dropout(x, p=0.5, training=True) 
#         position = self.fc_position(x)
#         rotation = self.fc_rotation(x)

#         return position, rotation, fv
# # class a(nn.Module):
#     def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0, bayesian = False):
#         super(a, self).__init__()

#         self.bayesian = bayesian
#         self.dropout_rate = dropout_rate
#         init_modules = [self.fc_last, self.fc_position, self.fc_rotation]

#         for module in init_modules:
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#                 nn.init.kaiming_normal_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)

#     def forward(self, x):
#         x = self.base_model(x)
#         x = x.view(x.size(0), -1)
#         x= self.fc_last(x)
#         x_fully = F.relu(x)

#         dropout_on = self.training or self.bayesian
#         if self.dropout_rate > 0:
#             x = F.dropout(x, p=self.dropout_rate, training=dropout_on)
        
#         position = self.fc_position(x)
#         rotation = self.fc_rotation(x)

#         return position, rotation, x_fully

        # return position, rotation, 0

class wawapig(nn.Module):
    def __init__(self, base_model):
        super(wawapig, self).__init__()
        self.fc_position = base_model.fc_position
        self.fc_rotation = base_model.fc_rotation

    def forward(self, x):
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=True)  
        # ipdb.set_trace()  
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation
class Densenet(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0, bayesian = False):
        super(Densenet, self).__init__()

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        feat_in = base_model.classifier.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # self.base_model = base_model

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(2048, 3, bias=True)
        self.fc_rotation = nn.Linear(2048, 4, bias=True)

        init_modules = [self.fc_last, self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # nn.init.normal_(self.fc_last.weight, 0, 0.01)
        # nn.init.constant_(self.fc_last.bias, 0)
        #
        # nn.init.normal_(self.fc_position.weight, 0, 0.5)
        # nn.init.constant_(self.fc_position.bias, 0)
        #
        # nn.init.normal_(self.fc_rotation.weight, 0, 0.01)
        # nn.init.constant_(self.fc_rotation.bias, 0)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.base_model(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x_fully= self.fc_last(x)
        x = F.relu(x_fully)

        dropout_on = self.training or self.bayesian
        # print(self.dropout_rate, dropout_on)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=dropout_on)
        # x = self.dropout(x)
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x_fully

class ResNet(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0, bayesian = False):
        super(ResNet, self).__init__()

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # self.base_model = base_model

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(2048, 3, bias=True)
        self.fc_rotation = nn.Linear(2048, 4, bias=True)

        init_modules = [self.fc_last, self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # nn.init.normal_(self.fc_last.weight, 0, 0.01)
        # nn.init.constant_(self.fc_last.bias, 0)
        #
        # nn.init.normal_(self.fc_position.weight, 0, 0.5)
        # nn.init.constant_(self.fc_position.bias, 0)
        #
        # nn.init.normal_(self.fc_rotation.weight, 0, 0.01)
        # nn.init.constant_(self.fc_rotation.bias, 0)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x_fully= self.fc_last(x)
        x = F.relu(x_fully)

        dropout_on = self.training or self.bayesian
        # print(self.dropout_rate, dropout_on)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=dropout_on)
        # x = self.dropout(x)
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x_fully


class ResNetSimple(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0):
        super(ResNetSimple, self).__init__()
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # self.base_model = base_model

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(feat_in, 3, bias=False)
        self.fc_rotation = nn.Linear(feat_in, 4, bias=False)

        init_modules = [self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight.data)
                if module.bias is not None:
                    nn.init.constant(module.bias.data, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation

class GoogleNet(nn.Module):
    """ PoseNet using Inception V3 """
    def __init__(self, base_model, fixed_weight=False, dropout_rate = 0.0):
        super(GoogleNet, self).__init__()
        self.dropout_rate =dropout_rate

        model = []
        model.append(base_model.Conv2d_1a_3x3)
        model.append(base_model.Conv2d_2a_3x3)
        model.append(base_model.Conv2d_2b_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Conv2d_3b_1x1)
        model.append(base_model.Conv2d_4a_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Mixed_5b)
        model.append(base_model.Mixed_5c)
        model.append(base_model.Mixed_5d)
        model.append(base_model.Mixed_6a)
        model.append(base_model.Mixed_6b)
        model.append(base_model.Mixed_6c)
        model.append(base_model.Mixed_6d)
        model.append(base_model.Mixed_6e)
        model.append(base_model.Mixed_7a)
        model.append(base_model.Mixed_7b)
        model.append(base_model.Mixed_7c)
        self.base_model = nn.Sequential(*model)

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Out 2
        self.pos2 = nn.Linear(2048, 3, bias=True)
        self.ori2 = nn.Linear(2048, 4, bias=True)

    def forward(self, x):
        # 299 x 299 x 3
        x = self.base_model(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        pos = self.pos2(x)
        ori = self.ori2(x)

        return pos, ori, 0
