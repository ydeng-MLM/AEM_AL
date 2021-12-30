import torch
from torch import nn

import numpy as np

from model import SimpleNN, MLP

class MLP_toy(object):
    def __init__(self, train_loader, test_loader):
        self.linear = [1, 50, 50, 1]
        self.LR = 1e-3
        self.LR_decacy = 0.2
        self.WD = 1e-5
        self.epoch = 100
        self.loss_fn = nn.MSELoss()
        self.model = SimpleNN(self.linear).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR, weight_decay=self.WD)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=self.LR_decacy,
                                                               patience=10, threshold=1e-4)

        self.trainloader = train_loader
        self.testloader = test_loader

    def train(self, verbose=True):
        for t in range(self.epoch):
            self.model.train()
            size = len(self.trainloader.dataset)
            for batch, (X, y) in enumerate(self.trainloader):
                # Compute prediction and loss
                X = X.cuda()
                y = y.cuda()
                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if verbose:
                if t%10==0:
                    loss = loss.item()
                    print("Epoch "+str(t)+"\n-------------------------------")
                    print(f"loss: {loss:>7f}")

            self.model.eval()
            size = len(self.testloader.dataset)
            num_batches = len(self.testloader)
            test_loss = 0

            with torch.no_grad():
                for X, y in self.testloader:
                    X = X.cuda()
                    y = y.cuda()
                    pred = self.model(X)
                    test_loss += self.loss_fn(pred, y).item()

            test_loss /= num_batches
            # if t%10==0:
            # print(f"Test Error Avg loss: {test_loss:>8f} \n")

            self.scheduler.step(test_loss)

        if verbose:
            print('Model training done!')

    def predict(self):
        self.model.eval()
        size = len(self.testloader.dataset)
        num_batches = len(self.testloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in self.testloader:
                X = X.cuda()
                y = y.cuda()
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()

        test_loss /= num_batches
        return test_loss

    def eval_pool(self):
        self.model.eval()
        size = len(self.testloader.dataset)
        num_batches = len(self.testloader)
        test_loss = np.array([])

        with torch.no_grad():
            for X, y in self.testloader:
                X = X.cuda()
                y = y.cuda()
                pred = self.model(X)
                test_loss = np.append(test_loss, self.loss_fn(pred, y).item())

        return test_loss

    def load(self):
        checkpoint = torch.load('model/fixed_init_toy.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class MLP_nano(object):
    def __init__(self, train_loader, test_loader):
        self.linear = [8, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 201]
        self.LR = 1e-4
        self.LR_decacy = 0.2
        self.WD = 1e-3
        self.epoch = 500
        self.loss_fn = nn.MSELoss()
        self.model = MLP(self.linear).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR, weight_decay=self.WD)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=self.LR_decacy,
                                                               patience=10, threshold=1e-4)

        self.trainloader = train_loader
        self.testloader = test_loader

    def train(self, verbose=True):
        for t in range(self.epoch):
            self.model.train()
            size = len(self.trainloader.dataset)
            for batch, (X, y) in enumerate(self.trainloader):
                # Compute prediction and loss
                X = X.cuda()
                y = y.cuda()
                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if verbose:
                if t%10==0:
                    loss = loss.item()
                    print("Epoch "+str(t)+"\n-------------------------------")
                    print(f"loss: {loss:>7f}")

            self.model.eval()
            size = len(self.testloader.dataset)
            num_batches = len(self.testloader)
            test_loss = 0

            with torch.no_grad():
                for X, y in self.testloader:
                    X = X.cuda()
                    y = y.cuda()
                    pred = self.model(X)
                    test_loss += self.loss_fn(pred, y).item()

            test_loss /= num_batches
            # if t%10==0:
            # print(f"Test Error Avg loss: {test_loss:>8f} \n")

            self.scheduler.step(test_loss)

        if verbose:
            print('Model training done!')

    def predict(self):
        self.model.eval()
        size = len(self.testloader.dataset)
        num_batches = len(self.testloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in self.testloader:
                X = X.cuda()
                y = y.cuda()
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()

        test_loss /= num_batches
        return test_loss

    def eval_pool(self):
        self.model.eval()
        size = len(self.testloader.dataset)
        num_batches = len(self.testloader)
        test_loss = np.array([])

        with torch.no_grad():
            for X, y in self.testloader:
                X = X.cuda()
                y = y.cuda()
                pred = self.model(X)
                test_loss = np.append(test_loss, self.loss_fn(pred, y).item())

        return test_loss

    def load(self):
        checkpoint = torch.load('model/fixed_init_nano.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])