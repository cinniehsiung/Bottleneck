import torch
import torch.nn as nn


class LogNormalDropout(nn.Module):
    def __init__(self, device, shape, max_alpha, module, **params):
        super(LogNormalDropout, self).__init__()
        self.device = device
        self.register_buffer('noise', torch.empty(shape))

        # copying constants from https://github.com/ucla-vision/information-dropout/blob/master/cifar.py
        self.max_alpha = max_alpha
        self.eps = 1e-5

        self.layer = module(**params)
        self.layer_noise = module(**params)

        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Sample noise   e ~ log N(-alpha/2, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # calculate alpha
            self.alpha = self.max_alpha*self.Sigmoid(self.layer_noise.forward(x)) + self.eps

            # calculate information in the weights
            self.Iw = - torch.sum(torch.log(self.alpha/(self.max_alpha+self.eps)))/x.shape[0]
            #self.Iw = - torch.sum(torch.log(self.alpha))/x.shape[0]

            # perform dropout using reparametrization trick
            mean, std = -self.alpha/2.0, self.alpha**0.5
            Z = self.noise.normal_(0, 1)
            epsilon  = torch.exp(mean+std*Z)
            return self.layer.forward(x) * epsilon
        else:
            return self.layer.forward(x)
        
class LogNormalDropoutSingle(nn.Module):
    def __init__(self, device, shape, max_alpha, module, **params):
        super(LogNormalDropoutSingle, self).__init__()
        self.device = device
        self.register_buffer('noise', torch.empty(shape))

        # copying constants from https://github.com/ucla-vision/information-dropout/blob/master/cifar.py
        self.max_alpha = max_alpha
        self.eps = 1e-5

        self.layer = module(**params)
        #self.layer_noise = module(**params)

        self.Sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        """
        Sample noise   e ~ log N(-alpha/2, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # calculate alpha
            self.alpha = self.max_alpha*self.Sigmoid(self.weight) + self.eps

            # calculate information in the weights
            self.Iw = - torch.sum(torch.log(self.alpha/(self.max_alpha+self.eps)))/x.shape[0]
            #self.Iw = - torch.sum(torch.log(self.alpha))/x.shape[0]

            # perform dropout using reparametrization trick
            mean, std = -self.alpha/2.0, self.alpha**0.5
            Z = self.noise.normal_(0, 1)
            epsilon  = torch.exp(mean+std*Z)
            return self.layer.forward(x) * epsilon
        else:
            return self.layer.forward(x)
        
