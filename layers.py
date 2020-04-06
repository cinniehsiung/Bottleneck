import torch
import torch.nn as nn
from torch.autograd import Variable

class LogNormalDropout(nn.Module):
    def __init__(self, alpha):
        super(LogNormalDropout, self).__init__()
        self.alpha = alpha #torch.Tensor([alpha])
        
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train() and self.alpha:
            # lognormal(mean = alpha, var = alpha)
            # ._log_normal(mean, std)
            epsilon = torch.empty(x.size()).log_normal_(-self.alpha/2.0, self.alpha**0.5)

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x
