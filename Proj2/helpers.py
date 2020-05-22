from torch import empty
import torch
import statistics
import math

from modules import LossMSE

class SGD():
    def __init__(self, params, lr = 1e-2):
        self.lr = lr
        self.params = params
    def step(self):
        for p, grad in self.params:
            p.add_(-self.lr * grad)
    def zero_grad(self):
        for p, grad in self.params:
            grad.zero_()

def generate_sets(size = 1000):
    center = 0.5
    radius = 1 / math.sqrt(2*math.pi)
    train_set  = empty(size, 2).uniform_(0, 1)
    test_set = empty(size, 2).uniform_(0, 1)
    train_target = ((center - train_set).norm(dim = 1) <= radius) * 1
    test_target = ((center - test_set).norm(dim = 1) <= radius) * 1
    return train_set, train_target.int(), test_set, test_target.int()

def nb_classification_errors(model, test_input, test_target, batch):
    nb_errors = 0
    for b in range(0, test_input.size(0), batch):
        output = model.forward(test_input.narrow(0,b, batch))
        _, predicted_class = output.max(1)
        for i in range(batch):
            if test_target[b + i, predicted_class[i]] <= 0 :
                nb_errors = nb_errors + 1
    return nb_errors

def ohe(train_targets, test_targets):
    new_train = empty((train_targets.size()[0],2)).zero_()
    new_test = empty((train_targets.size()[0],2)).zero_()
    for i,b in enumerate(train_targets.tolist()):
        new_train[i,b] = 1
    for i,b in enumerate(test_targets.tolist()):
        new_test[i,b] = 1
    return new_train, new_test


def normalize(train_input, test_input):
    return((train_input - train_input.mean(0))/train_input.std(0), (test_input - train_input.mean(0))/train_input.std(0))


def train_model(model, train_input, train_target, test_input, test_target, lr = 1e-2, num_epoch = 25, batch = 100):
    criterion =LossMSE()
    optimizer = SGD(model.param(), lr=lr)
    losses_train = []
    losses_test = []
    n = train_input.size(0)
    for e in range(num_epoch):
        running_loss = []
        test_loss = []

        for b in range(0, train_input.size(0), batch):
            output = model.forward(train_input.narrow(0, b, batch))
            loss = criterion.forward(output, train_target.narrow(0, b, batch))
            optimizer.zero_grad()
            model.backward(criterion.backward())
            optimizer.step()
            running_loss.append(loss)
        losses_train.append(torch.Tensor(running_loss).mean().tolist())

        for b in range(0, test_input.size(0), batch):
            test_loss.append(criterion.forward(model.forward(test_input.narrow(0, b, batch)), test_target.narrow(0, b, batch)))
        losses_test.append(torch.Tensor(test_loss).mean().tolist())

    return losses_train, losses_test
