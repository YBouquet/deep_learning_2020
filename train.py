"""
@author: Yann BOUQUET
"""

import torch
from torch import nn
import torch.optim as optim

from io_num_process import one_hot_encoding


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_kfold(train_input, k_fold):
    nrows = train_input.size(0)
    fold_size = int(nrows / k_fold)
    if fold_size * k_fold != nrows:
        raise ValueError(
            'ERROR: k_fold value as to be a divisor of the number of rows in the training set')
    indices = torch.randperm(nrows)
    result = [indices[k * fold_size: (k + 1) * fold_size]
              for k in range(k_fold)]
    return torch.stack(result)


def train_model(model, train_input, train_target, train_figures_target, k_fold, mini_batch_size, lr, num_epoch=25, auxiliary_loss=True):
    criterion = nn.BCEWithLogitsLoss()
    auxiliary_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_ohe_figures = torch.cat((one_hot_encoding(
        train_figures_target[:, 0]), one_hot_encoding(train_figures_target[:, 1])), axis=1)

    for e in range(num_epoch):
        logs = {'loss': [], 'val_loss': []}
        avg_train_loss = []
        avg_val_loss = []
        indices = build_kfold(train_input, k_fold)
        for k in range(k_fold):
            va_indices = indices[k]
            tr_indices = indices[~(torch.arange(
                indices.size(0)) == k)].view(-1)

            tr_sf_input = train_input[tr_indices]
            va_sf_input = train_input[va_indices]

            tr_sf_target = train_target[tr_indices]
            va_sf_target = train_target[va_indices]

            tr_sf_figures = train_ohe_figures[tr_indices]
            va_sf_figures = train_ohe_figures[va_indices]

            m_size = tr_sf_input.size(0)

            model.train()

            running_loss = 0
            for b in range(0, m_size, mini_batch_size):
                # (BATCH_SIZE, 2, 14,14)
                outputs = model(tr_sf_input.narrow(0, b, mini_batch_size))
                try:
                    outputs = torch.stack(outputs)
                    loss = criterion(
                        outputs[-1], tr_sf_target.narrow(0, b, mini_batch_size).type_as(outputs[-1]))
                    if auxiliary_loss:
                        figures_batch = tr_sf_figures.narrow(
                            0, b, minibatch_size)
                        for i in range(2):
                            loss += auxiliary_criterion(
                                outputs[i], figures_batch[:, i].type(torch.LongTensor))
                except TypeError:
                    loss = criterion(outputs, tr_sf_target.narrow(
                        0, b, mini_batch_size).type_as(outputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss
            avg_train_loss.append(running_loss)

            m_size = va_sf_input.size(0)
            running_loss = 0
            model.eval()
            with torch.no_grad():
                for b in range(0, m_size, mini_batch_size):
                    # (BATCH_SIZE, 2, 14,14)
                    outputs = model(va_sf_input.narrow(0, b, mini_batch_size))
                    try:
                        outputs = torch.stack(outputs)
                        loss = criterion(
                            outputs[-1], va_sf_target.narrow(0, b, mini_batch_size).type_as(outputs[-1]))
                        if auxiliary_loss:
                            figures_batch = va_sf_figures.narrow(
                                0, b, minibatch_size)
                            for i in range(2):
                                loss += auxiliary_criterion(
                                    outputs[i], figures_batch[:, i].type(torch.LongTensor))
                    except TypeError:
                        loss = criterion(outputs, va_sf_target.narrow(
                            0, b, mini_batch_size).type_as(outputs))
                    running_loss += loss
            avg_val_loss.append(running_loss)

        logs['loss'].append(torch.tensor(avg_train_loss).mean())
        logs['val_loss'].append(torch.tensor(avg_val_loss).mean())

        temp_loss = torch.tensor(logs['loss'])
        temp_val_loss = torch.tensor(logs['val_loss'])
        print('Epoch %3d / %3d \n\t Training \t\t\t min: %12.5f, max: %12.5f, cur: %12.5f\n\t Validation \t\t\t min: %12.5f, max: %12.5f, cur: %12.5f\n'\
                    % (e+1, num_epoch, temp_loss.min(), temp_loss.max(), temp_loss[-1],temp_val_loss.min(), temp_val_loss.max(), temp_val_loss[-1]))

    model.eval()
