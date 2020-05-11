#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 08:51:08 2020

@author: thomas
"""
### Python modules
import math
from torch import empty

### Project modules
import modules


### Stochastic gradient descent
class SGD():
    def __init__(self, params, lr):
        self.lr = lr
        self.params = params
    def step(self):
        for p, grad in self.params:
            p.add_(-self.lr * grad)
    def zero_grad(self):
        for p, grad in self.params:
            grad.zero_()


def generate_sets(size = 1000, seed = 1):
    center = 0.5
    radius = 1 / math.sqrt(2*math.pi)
    train_set  = empty(size, 2).uniform_(0, 1)
    test_set = empty(size, 2).uniform_(0, 1)
    train_target = ((center - train_set).norm(dim = 1) <= radius) * 1
    test_target = ((center - test_set).norm(dim = 1) <= radius) * 1
    
    return train_set, train_target.int(), test_set, test_target.int()


### One-hot encoding
def ohe(train_targets, test_targets):
    new_train = empty((train_targets.size()[0],2)).zero_() 
    new_test = empty((train_targets.size()[0],2)).zero_() 
    for i,b in enumerate(train_targets.tolist()):
        new_train[i,b] = 1
    for i,b in enumerate(test_targets.tolist()):
        new_test[i,b] = 1
    return new_train, new_test


def train_model(model, train_input, train_target, test_input, test_target, lr, num_epoch):
    criterion = modules.LossMSE()
    optimizer = SGD(model.param(), lr)
    losses_train = []
    losses_test = []
    n = train_input.size(0)
    for e in range(num_epoch):
        running_loss = empty(n).zero_().float()
        test_loss = empty(n).zero_().float()
        
        for i, input_ in enumerate(train_input) :
            output = model.forward(input_)
            loss = criterion.forward(output, train_target[i])
            optimizer.zero_grad()
            model.backward(criterion.backward())
            optimizer.step()
            running_loss[i] = loss
        losses_train.append(running_loss.mean().tolist())
        
        for i, input_ in enumerate(test_input):
            test_loss[i] = criterion.forward(model.forward(input_), test_target[i])
        losses_test.append(test_loss.mean().tolist())
        
    return losses_train, losses_test


def nb_classification_errors(model, test_input, test_target):
    nb_errors = 0
    
    for i, input_ in enumerate(test_input):
        output = model.forward(input_)
        _, predicted_class = output.max(0)
        if test_target[i, predicted_class] <= 0:
            nb_errors = nb_errors + 1
    return nb_errors


def print_error(name, nb_errors, size_):
    error_rate = (100 * nb_errors) / size_
    print(name + ' error : {:0.2f}% {:d}/{:d}'.format(error_rate,
                                                      nb_errors, size_))