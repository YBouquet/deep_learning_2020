import torch
from torch import nn
import torch.optim as optim

def train_model(model, train_input, train_target, mini_batch_size, lr, num_epoch = 25):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)


    for e in range(num_epoch):
        running_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            outputs = model(train_input.narrow(0, b, mini_batch_size)) #(BATCH_SIZE, 2, 14,14)
            loss = criterion(outputs, train_target.narrow(0, b, mini_batch_size).type_as(outputs))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("epoch {0:3d}, loss = {1:8.5f}".format(e, running_loss))
