# modified from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
# based on Soatto disentanglement page 24
# omitting batch norm for now

import torch
import torch.nn as nn
from layers import LogNormalDropout 

class AlexNet(nn.Module):

    def __init__(self, B, num_classes=10, use_bn=True):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
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
            LogNormalDropout(shape=(B, 10)),
            # nn.CrossEntropyLoss expects raw logits
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 
