# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

#---------------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models as tv
# import torchvision


# class VGG(torch.nn.Module):
#     def __init__(self, requires_grad=False, pretrained=True):
#         super(VGG, self).__init__()
        
#         vgg_pretrained_features = tv.vgg19(pretrained=pretrained).features
            
#         # print(vgg_pretrained_features)
#         self.stage1 = torch.nn.Sequential()
#         self.stage2 = torch.nn.Sequential()
#         self.stage3 = torch.nn.Sequential()
#         self.stage4 = torch.nn.Sequential()
#         self.stage5 = torch.nn.Sequential()

#         # vgg19
#         for x in range(0,4):
#             self.stage1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(4, 9):
#             self.stage2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(9, 18):
#             self.stage3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(18, 27):
#             self.stage4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(27, 36):
#             self.stage5.add_module(str(x), vgg_pretrained_features[x])                
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False
        
#         self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
#         self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

#         self.chns = [64,128,256,512,512]

  
#     def get_features(self, x):
#         # normalize the data
#         h = (x-self.mean)/self.std
        
#         h = self.stage1(h)
#         h_relu1_2 = h
        
#         h = self.stage2(h)
#         h_relu2_2 = h
        
#         h = self.stage3(h)
#         h_relu3_3 = h
        
#         h = self.stage4(h)
#         h_relu4_3 = h

#         h = self.stage5(h)
#         h_relu5_3 = h

#         # get the features of each layer
#         outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

#         return outs
       
#     def forward(self, x):
#         feats_x = self.get_features(x)
#         return feats_x



# class FDL_loss(torch.nn.Module):
#     def __init__(
#         self, loss_weight=1.0, reduction='mean',patch_size=5, stride=1, num_proj=256, model="VGG", phase_weight=1.0
#     ):
#         """
#         patch_size, stride, num_proj: SWD slice parameters
#         model: feature extractor, support VGG, ResNet, Inception, EffNet
#         phase_weight: weight for phase branch
#         """

#         super(FDL_loss, self).__init__()
#         if model == "ResNet":
#             self.model = ResNet()
#         elif model == "EffNet":
#             self.model = EffNet()
#         elif model == "Inception":
#             self.model = Inception()
#         elif model == "VGG":
#             self.model = VGG()
#         else:
#             assert "Invalid model type! Valid models: VGG, Inception, EffNet, ResNet"
#         self.loss_weight = loss_weight
#         self.reduction = reduction 
#         self.phase_weight = phase_weight
#         self.stride = stride
#         for i in range(len(self.model.chns)):
#             rand = torch.randn(num_proj, self.model.chns[i], patch_size, patch_size)
#             rand = rand / rand.view(rand.shape[0], -1).norm(dim=1).unsqueeze(
#                 1
#             ).unsqueeze(2).unsqueeze(3)
#             self.register_buffer("rand_{}".format(i), rand)

#         # print all the parameters

#     def forward_once(self, x, y, idx):
#         """
#         x, y: input image tensors with the shape of (N, C, H, W)
#         """
#         rand = self.__getattr__("rand_{}".format(idx))
#         projx = F.conv2d(x, rand, stride=self.stride)
#         projx = projx.reshape(projx.shape[0], projx.shape[1], -1)
#         projy = F.conv2d(y, rand, stride=self.stride)
#         projy = projy.reshape(projy.shape[0], projy.shape[1], -1)

#         # sort the convolved input
#         projx, _ = torch.sort(projx, dim=-1)
#         projy, _ = torch.sort(projy, dim=-1)

#         # compute the mean of the sorted convolved input
#         s = torch.abs(projx - projy).mean([1, 2])

#         return s

#     def forward(self, x, y):
#         x = self.model(x)
#         y = self.model(y)
#         score = []
#         for i in range(len(x)):
#             # Transform to Fourier Space
#             fft_x = torch.fft.fftn(x[i], dim=(-2, -1))
#             fft_y = torch.fft.fftn(y[i], dim=(-2, -1))

#             # get the magnitude and phase of the extracted features
#             x_mag = torch.abs(fft_x)
#             x_phase = torch.angle(fft_x)
#             y_mag = torch.abs(fft_y)
#             y_phase = torch.angle(fft_y)

#             s_amplitude = self.forward_once(x_mag, y_mag, i)
#             s_phase = self.forward_once(x_phase, y_phase, i)

#             score.append(s_amplitude + s_phase * self.phase_weight)

#         score = sum(score)  # sumup between different layers
#         score = score.mean()  # mean within batch
#         return score  # the bigger the score, the bigger the difference between the two images
