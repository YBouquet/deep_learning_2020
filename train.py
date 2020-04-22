"""
@author: Yann BOUQUET
"""

import torch
from torch import nn
import torch.optim as optim


def decrease_learning_rate(lr, optimizer, e, num_epoch):
    lr = lr * (0.8 ** (e / num_epoch)) # 0.8 best ratio for now
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_model(model, train_input, train_target, mini_batch_size, lr, num_epoch = 25, decrease_lr=False):
    model.train(True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    m_size = train_input.size(0)
    for e in range(num_epoch):
        if decrease_lr:
            decrease_learning_rate(lr, optimizer, e, num_epoch)
        running_loss = 0
        indices = torch.randperm(m_size)
        tr_sf_input = train_input[indices]
        tr_sf_target = train_target[indices]
        for b in range(0, m_size, mini_batch_size):
            optimizer.zero_grad()
            outputs = model(tr_sf_input.narrow(0, b, mini_batch_size)) #(BATCH_SIZE, 2, 14,14)
            loss = criterion(outputs, tr_sf_target.narrow(0, b, mini_batch_size).type_as(outputs))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("epoch {0:3d}, loss = {1:8.5f}".format(e, running_loss))
    model.train(False)
