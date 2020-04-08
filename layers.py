import torch
import torch.nn as nn
from torch.autograd import Variable

class LogNormalDropout(nn.Module):
    def __init__(self, shape):
        super(LogNormalDropout, self).__init__()
        self.register_buffer('noise', torch.empty(shape))
        self.alpha = nn.Parameter(torch.tensor(1))
        
    def forward(self, x):
        """
        Sample noise   e ~ log N(-alpha/2, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            mean, std = -self.alpha/2.0, self.alpha**0.5
            Z = self.noise.normal_(0, 1)
            epsilon  = torch.exp(mean+std*Z)
            return x * epsilon
        else:
            return x
