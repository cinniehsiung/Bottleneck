# modified from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
# based on Soatto disentanglement page 24
# omitting batch norm for now

import torch
import torch.nn as nn
from layers import LogNormalDropout 

class AlexNet(nn.Module):

    def __init__(self, B, num_classes=10, use_bn=True):
        super(AlexNet, self).__init__()

        # for the IB Langrangian, to be able to reference alpha
        #self.dropout = LogNormalDropout(shape=(B,10))
        self.dropout = LogNormalDropout(shape=(B, 64, 14, 14), max_alpha= 0.7, kernel_size=5, padding=2)
        
        # define the layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            self.dropout,
            nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=not use_bn),
            nn.BatchNorm2d(num_features=64) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(7*7*64, 384, bias=not use_bn),
            nn.BatchNorm1d(num_features=384) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192, bias=not use_bn),
            nn.BatchNorm1d(num_features=192) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(192, 10),
            nn.Softmax(dim=1),
            #self.dropout,
            # nn.CrossEntropyLoss expects raw logits
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 

    def getAlpha(self):
        return self.dropout.alpha

    def getIw(self):
        return self.dropout.Iw
