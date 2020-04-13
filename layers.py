import torch
import torch.nn as nn
from torch.autograd import Variable


class LogNormalDropout(nn.Module):
    def __init__(self, shape, max_alpha = 0.7, kernel_size=5, stride=1, padding=2):
        super(LogNormalDropout, self).__init__()
        self.register_buffer('noise', torch.empty(shape))

        # should match previous layer
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # copying constants from https://github.com/ucla-vision/information-dropout/blob/master/cifar.py
        self.max_alpha = max_alpha
        self.eps = 0.001


        #self.alpha = nn.Parameter(torch.tensor(0.5))
        self.channels = shape[1]
        
    def forward(self, x):
        """
        Sample noise   e ~ log N(-alpha/2, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            Conv2d =  nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, 
                    stride=self.stride, padding = self.padding)
            Sigmoid = nn.Sigmoid()

            # calculate alpha
            self.alpha = self.max_alpha*Sigmoid(Conv2d.forward(x)) + self.eps

            # calculate information in the weights
            self.Iw = - torch.log(self.alpha/(self.max_alpha+self.eps))

            # perform dropout using reparametrization trick
            mean, std = -self.alpha/2.0, self.alpha**0.5
            Z = self.noise.normal_(0, 1)
            epsilon  = torch.exp(mean+std*Z)
            return x * epsilon
        else:
            return x
