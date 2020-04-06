import torch
import torch.nn as nn
from torch.autograd import Variable

class LogNormalDropout(nn.Module):
    def __init__(self, shape, alpha):
        super(LogNormalDropout, self).__init__()
        self.register_buffer('noise', torch.empty(shape))
        self.alpha = alpha
        
    def forward(self, x):
        """
        Sample noise   e ~ log N(-alpha/2, alpha)
        Multiply noise h = h_ * e
        """
        if self.train() and self.alpha:
            mean, std = -self.alpha/2.0, self.alpha**0.5
            epsilon = self.noise.log_normal_(mean, std)
            return x * epsilon
        else:
            return x
