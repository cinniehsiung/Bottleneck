# modified from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
# based on Soatto disentanglement page 24
# omitting batch norm for now

import torch
import torch.nn as nn
from layers import LogNormalDropout, LogNormalDropoutSingle

class AlexNet(nn.Module):

    def __init__(self, device, B, max_alpha, num_classes=10, use_bn=True):
        super(AlexNet, self).__init__()

        # for the IB Langrangian, to be able to reference alpha
        self.dropout_layers_features = []#[0, 3]
        self.dropout_layers_classifier = [6]#[0, 3]
        
        # define the layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            #LogNormalDropoutSingle(device=device, shape=(B, 64, 28, 28), max_alpha= 0.7, 
            #    module=nn.Conv2d, in_channels=3, out_channels=64, kernel_size=5, padding=2, bias=not use_bn),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=not use_bn),
            #LogNormalDropoutSingle(device=device, shape=(B, 64, 14, 14), max_alpha= 0.7, 
            #    module=nn.Conv2d, in_channels=64, out_channels=64, kernel_size=5, padding=2, bias=not use_bn),
            nn.BatchNorm2d(num_features=64) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(7*7*64, 384, bias=not use_bn),
            #LogNormalDropoutSingle(device=device, shape=(B, 384), max_alpha=max_alpha, 
            #    module=nn.Linear, in_features=7*7*64, out_features=384),
            nn.BatchNorm1d(num_features=384) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192, bias=not use_bn),
            #LogNormalDropoutSingle(device=device, shape=(B, 192), max_alpha=max_alpha, 
            #    module=nn.Linear, in_features=384, out_features=192),
            nn.BatchNorm1d(num_features=192) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            #nn.Linear(192, 10),
            LogNormalDropoutSingle(device=device, shape=(B, 10), max_alpha=max_alpha, 
                module=nn.Linear, in_features=192, out_features=10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 

    def getIw(self):
        Iw_features = sum(self.features[idx].Iw for idx in self.dropout_layers_features)
        Iw_classifier = sum(self.classifier[idx].Iw for idx in self.dropout_layers_classifier)
        return Iw_features+Iw_classifier
