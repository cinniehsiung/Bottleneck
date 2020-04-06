# modified from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
# based on Soatto disentanglement page 24
# omitting batch norm for now

import torch
import torch.nn as nn
from layers import LogNormalDropout 

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            LogNormalDropout(alpha=0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            #BN
            LogNormalDropout(alpha = 0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(7*7*64, 384),
            LogNormalDropout(alpha=0.2),
            # BN
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            LogNormalDropout(alpha=0.2),
            # BN
            nn.ReLU(inplace=True),
            nn.Linear(192, 10),
            LogNormalDropout(alpha=0.2)
            # nn.Softmax(dim=1) # using CrossEntropyLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 
