"""
@author: Yann BOUQUET
"""

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from io_num_process import one_hot_encoding


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAINING_PHASE = 'train'
VALIDATION_PHASE = 'validation'

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

    logs = {'loss': [], 'val_loss': []}

    for e in range(num_epoch):
        avg_loss = {TRAINING_PHASE: [], VALIDATION_PHASE: []}

        indices = build_kfold(train_input, k_fold)
        for k in range(k_fold):
            va_indices = indices[k]
            tr_indices = indices[~(torch.arange(indices.size(0)) == k)].view(-1)

            train_dataset = TensorDataset(train_input[tr_indices], train_target[tr_indices], train_ohe_figures[tr_indices])
            validation_dataset = TensorDataset(train_input[va_indices],  train_target[va_indices], train_ohe_figures[va_indices])

            dataloaders = {
                TRAINING_PHASE : DataLoader(train_dataset, batch_size = mini_batch_size, shuffle = False),
                VALIDATION_PHASE : DataLoader(validation_dataset, batch_size = min(mini_batch_size, va_indices.size(0)), shuffle = False)
            }

            for phase in [TRAINING_PHASE, VALIDATION_PHASE]:
                if phase == TRAINING_PHASE:
                    model.train()
                else:
                    model.eval()
                running_loss = 0

                for inputs, targets, figures in dataloaders[phase]:
                    outputs = model(inputs)
                    try:
                        tuples = torch.stack(outputs)
                        loss = criterion(tuples[-1], targets.type_as(tuples[-1]))
                        if auxiliary_loss:
                            for i in range(len(tuples) - 1):
                                loss += auxiliary_criterion(tuples[i], figures[:, i].type(torch.LongTensor))
                    except TypeError:
                        loss = criterion(outputs, targets.type_as(outputs))

                    if phase == TRAINING_PHASE:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss

                avg_loss[phase].append(running_loss)

        logs['loss'].append(torch.tensor(avg_loss[TRAINING_PHASE]).mean())
        logs['val_loss'].append(torch.tensor(avg_loss[VALIDATION_PHASE]).mean())

        temp_loss = torch.tensor(logs['loss'])
        temp_val_loss = torch.tensor(logs['val_loss'])
        print('Epoch %3d / %3d \n\t Training \t\t\t min: %12.5f, max: %12.5f, cur: %12.5f\n\t Validation \t\t\t min: %12.5f, max: %12.5f, cur: %12.5f\n'\
                    % (e+1, num_epoch, temp_loss.min(), temp_loss.max(), temp_loss[-1],temp_val_loss.min(), temp_val_loss.max(), temp_val_loss[-1]))

    model.eval()
