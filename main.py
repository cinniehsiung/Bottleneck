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

DOWNLOAD = False
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
    parser.add_argument('--max_alpha', default=0.7, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--epochs', default=360, type=int)
    parser.add_argument('--patience', default=20, type=int, help='epochs to wait for early stopping; default no early stopping')
    parser.add_argument('--N', default=1000, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--random_labels', type=int)
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=int)
    args = parser.parse_args()
    args.batchnorm = True

    args.name += ('_rand' if args.random_labels else '_norand')
    if args.beta >= 0:
        solver = Solver(args)
        solver.run()
    else:
        print('Grid search')
        with open(args.name + '.csv', 'a') as f:
            w = csv.writer(f)
            w.writerow(['N', 'beta', 'train_acc', 'test_acc', 'train_loss',
                'test_loss', 'train_ce', 'test_ce', 'train_Iw', 'test_Iw', 'train_a', 'test_a'])
        bs = np.arange(-3.5, 3.1, 1)
        ns = np.arange(2, 4.51, 1)
        bns = []
        train_acc = -1*np.ones((len(bs), len(ns)))
        test_acc = -1*np.ones((len(bs), len(ns)))
        best_lr = np.zeros((len(bs), len(ns)))
        for i, j in product(range(len(bs)), range(len(ns))):
            b, n = bs[i], ns[j]
            args.beta = 10**b
            args.N = int(10**n)
            bns.append(bs[i] - ns[j])
            print("beta: {}, \t N: {}".format(args.beta, args.N))
            
            args.batch_size = min(args.N, 500)
            for lr in [0.005, 0.02]:
                args.lr = lr
                curr_train, curr_test = Solver(args).run()
                if curr_test >= test_acc[i, j]:
                    train_acc[i,j], test_acc[i, j], best_lr[i, j] = curr_train, curr_test, lr

            data = {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'best_lr': best_lr,
                'bs': bs,
                'ns': ns,
                'bns' : bns
            }
            pickle.dump(data, open(args.name+".p", "wb"))


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.name = config.name
        self.lr = config.lr
        self.momentum = config.momentum
        self.beta = config.beta
        self.max_alpha = config.max_alpha
        self.epochs = config.epochs
        self.patience = config.patience
        self.N = config.N
        self.batch_size = config.batch_size
        self.random_labels = config.random_labels
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
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=DOWNLOAD, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=DOWNLOAD, transform=transform)
        
        if self.random_labels:
            np.random.shuffle(train_set.targets)
            np.random.shuffle(test_set.targets)

        assert self.N <= 50000
        if self.N < 50000:
            train_set.data = train_set.data[:self.N]
            # downsize the test set to improve speed for small N
            test_set.data = test_set.data[:self.N]

        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = AlexNet(device=self.device, B=self.batch_size,
            max_alpha=self.max_alpha, use_bn=self.use_bn).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=140)
        self.criterion = nn.NLLLoss().to(self.device)


    def getIw(self):
        # Iw should be normalized with respect to N
        # via reparameterization, we optimize alpha with only 1920 dimensions
        # but Iw should scale with the dimension of the weights
        return 7*7*64*384/1920*self.model.getIw()/self.batch_size

    def do_batch(self, train, epoch):
        loader = self.train_loader if train else self.test_loader
        total_ce, total_Iw, total_loss = 0, 0, 0
        total_correct = 0
        total = 0
        pbar = tqdm(loader)
        num_batches = len(loader)
        for batch_num, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            if train:
                self.optimizer.zero_grad()
            output = self.model(data)
            # NLLLoss is averaged across observations for each minibatch
            ce = self.criterion(torch.log(output+EPS), target)
            Iw = self.getIw()
            loss = ce + 0.5*self.beta*Iw
            if train:
                loss.backward()
                self.optimizer.step()
            total_ce += ce.item()
            total_Iw += Iw.item()
            total_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            total += target.size(0)

            a = self.model.get_a()
            pbar.set_description('Train' if train else 'Test')
            pbar.set_postfix(N=self.N, b=self.beta, ep=epoch, acc=100. * total_correct/total,
                loss=total_loss/num_batches, ce=total_ce/num_batches, Iw=total_Iw/num_batches, a=a)
        return total_correct/total, total_loss/num_batches, total_ce/num_batches, total_Iw/num_batches, a

    def train(self, epoch):
        self.model.train()
        return self.do_batch(train=True, epoch=epoch)

    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            return self.do_batch(train=False, epoch=epoch)

    def save(self, name=None):
        model_out_path = (name or self.name) + ".pth"
        # torch.save(self.model, model_out_path)
        # print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        results = []
        best_acc, best_ep = -1, -1
        for epoch in range(1, self.epochs + 1):
            # print("\n===> epoch: %d/200" % epoch)
            train_acc, train_loss, train_ce, train_Iw, train_a = self.train(epoch)
            self.scheduler.step(epoch)
            test_acc, test_loss, test_ce, test_Iw, test_a = self.test(epoch)
            results.append([self.N, self.beta, train_acc, test_acc, train_loss,
                test_loss, train_ce, test_ce, train_Iw, test_Iw, train_a, test_a])

            if test_acc > best_acc:
                best_acc, best_ep = test_acc, epoch
            if self.patience >= 0: # early stopping
                if best_ep < epoch - self.patience:
                    break

        with open(self.name + '.csv', 'a') as f:
            w = csv.writer(f)
            w.writerows(results)
        self.save()

        return train_acc, test_acc


if __name__ == '__main__':
    main()
