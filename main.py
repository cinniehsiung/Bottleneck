# modified from https://github.com/icpm/pytorch-cifar10/blob/master/main.py

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import csv
import pickle
import argparse
from itertools import product

from alexnet import AlexNet
from tqdm import tqdm


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
EPS = 1e-8

def main():
    # Soatto entanglement p23:
    # sgd with initial lr = [0.02, 0.005], momentum 0.9
    # decay every 140 epochs
    # batch size 500
    # dataset size N varies from 100 to 50k
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--name', default='alexnet')
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--epochs', default=360, type=int)
    parser.add_argument('--patience', default=-1, type=int, help='epochs to wait for early stopping; default no early stopping')
    parser.add_argument('--N', default=1000, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--batchnorm', dest='batchnorm', action='store_true')
    parser.add_argument('--no-batchnorm', dest='batchnorm', action='store_false')
    parser.set_defaults(batchnorm=True)
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool)
    args = parser.parse_args()

    if args.beta >= 0:
        solver = Solver(args)
        solver.run()
    else:
        print('Grid search')
        bs = np.arange(-3.5, 3, 0.25)
        ns = np.arange(2, 5, 0.25)
        grid_vals = product(bs,ns)
        results = np.zeros_like(grid_vals)
        for i, j in product(np.arange(len(bs)), np.arange(len(ns))):
            b, n = bs[i], ns[j]
            args.beta = 10**b
            args.N = int(10**n)
            args.name = "beta: {}, \t N: {}".format(args.beta, args.N)
            print(args.name)
            
            if args.N < 500:
                args.batch_size = 100

            solver = Solver(args)
            results[i, j] = solver.run()

        pickle.dump(results, open("results.p", "wb"))


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.name = config.name
        self.lr = config.lr
        self.momentum = config.momentum
        self.beta = config.beta
        self.epochs = config.epochs
        self.patience = config.patience
        self.N = config.N
        self.batch_size = config.batch_size
        self.use_bn = config.batchnorm
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        # ToTensor scales pixel values from [0,255] to [0,1]
        mean_var = (125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255, 66.7/255)
        transform = transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor(), transforms.Normalize(*mean_var, inplace=True)])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False)

        assert self.N <= 50000
        if self.N < 50000:
            train_set.data = train_set.data[:self.N]
            # downsize the test set to improve speed for small N
            test_set.data = test_set.data[:self.N]

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = AlexNet(B=self.batch_size, use_bn=self.use_bn).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=140)
        self.criterion = nn.NLLLoss().to(self.device)

    def train(self):
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        pbar = tqdm(self.train_loader)
        for batch_num, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(torch.log(output+EPS), target) - 0.5*self.beta*torch.log(self.model.getAlpha())
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            pbar.set_description('Train')
            # pbar.set_description('Train epoch {}/{}'.format(epoch, self.epochs))
            pbar.set_postfix(loss=train_loss, acc=100. * train_correct / total, total=total, alpha=self.model.getAlpha().item())

        return train_loss, train_correct / total

    def test(self):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.test_loader)
            for batch_num, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                #loss = self.criterion(output, target)
                loss = self.criterion(torch.log(output + EPS), target) - 0.5*self.beta*torch.log(self.model.getAlpha())
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                pbar.set_description(' Test')
                pbar.set_postfix(loss=test_loss, acc=100. * test_correct / total, total=total, alpha=self.model.getAlpha().item())

        return test_loss, test_correct / total

    def save(self, name=None):
        model_out_path = (name or self.name) + ".pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        results = []
        best_acc, best_ep = -1, -1
        for epoch in tqdm(range(1, self.epochs + 1)):
            # print("\n===> epoch: %d/200" % epoch)
            train_loss, train_acc = self.train()
            self.scheduler.step(epoch)
            test_loss, test_acc = self.test()
            results.append([train_loss, train_acc, test_loss, test_acc])

            if self.patience >= 0: # early stopping
                if test_acc > best_acc:
                    best_acc, best_ep = test_acc, epoch
                if best_ep < epoch - patience:
                    break

        with open(self.name + '.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['train_loss', 'train_acc', 'test_loss', 'test_acc'])
            w.writerows(results)
        self.save()

        return best_acc


if __name__ == '__main__':
    main()
