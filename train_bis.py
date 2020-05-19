#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 08:13:09 2020

@author: thomas
"""

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAINING_PHASE = 'train'
VALIDATION_PHASE = 'validation'
PRETRAINING = 'pretraining'
TRAINING = 'training'


def decrease_learning_rate(lr, optimizer, e, num_epoch):
    lr = lr * (0.8 ** (e / num_epoch)) # 0.8 best ratio for now
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def build_kfold(train_input, k_fold):
    nrows = train_input.size(0)
    fold_size = int(nrows / k_fold)
    if fold_size * k_fold != nrows:
        raise ValueError(
            'ERROR: k_fold value as to be a divisor of the number of rows in the training set')
    indices = torch.randperm(nrows)
    result = [indices[k * fold_size : (k + 1) * fold_size] for k in range(k_fold)]
    return torch.stack(result)


def train_model(model, train_input, train_target, train_figures_target, k_fold, mini_batch_size, lr, num_epoch, auxiliary_loss=True, decrease_lr = False, criterion_type='BCE', aux_criterion_type = 'CE'):
    if criterion_type == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif criterion_type ==  'CE':
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == 'MSE': # Number recognition.
        criterion = nn.MSELoss()
    
    if aux_criterion_type == 'CE':
        auxiliary_criterion = nn.CrossEntropyLoss()
    elif aux_criterion_type == 'MSE':
        auxiliary_criterion = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_ohe_figures = train_figures_target

    logs = {'loss': [], 'val_loss': []}
        
    for e in range(num_epoch):
        avg_loss = {TRAINING_PHASE: [], VALIDATION_PHASE: []}

        indices = build_kfold(train_input, k_fold) # size([k_fold, 1000/k_fold])

        if decrease_lr:
            decrease_learning_rate(lr, optimizer, e, num_epoch)

        for k in range(k_fold):

            va_indices = indices[k] # 1000/k_fold indices for validation
            tr_indices = indices[~(torch.arange(indices.size(0)) == k)].view(-1) # (k_fold-1) * 1000 / k_fold indices (the rest)

            if k_fold == 1:
                va_indices, tr_indices = tr_indices, va_indices

            train_dataset = TensorDataset(train_input[tr_indices], train_target[tr_indices], train_ohe_figures[tr_indices])
            validation_dataset = TensorDataset(train_input[va_indices],  train_target[va_indices], train_ohe_figures[va_indices])

            dataloaders = {
                TRAINING_PHASE : DataLoader(train_dataset, batch_size = mini_batch_size, shuffle = False),
                VALIDATION_PHASE : DataLoader(validation_dataset, batch_size = mini_batch_size, shuffle = False)
            }

            for phase in [TRAINING_PHASE, VALIDATION_PHASE]:
                if phase == TRAINING_PHASE:
                    model.train()
                else:
                    model.eval()

                running_loss = []
                
                for inputs, targets, figures in dataloaders[phase]:
                    outputs = model(inputs)
                    try:
                        tuples = torch.stack(outputs)
                    except TypeError:
                        loss = criterion(outputs, targets.type_as(outputs))
                    except RuntimeError:
                        tuples = outputs
                        if criterion_type == 'BCE':
                            loss = criterion(tuples[-1], targets.type_as(torch.FloatTensor()))
                        if criterion_type == 'CE':
                            loss = criterion(tuples[-1], targets.type_as(torch.LongTensor()))
                        elif criterion_type == 'MSE':
                            loss = criterion(tuples[-1], targets)
                        if auxiliary_loss:                                  
                            for i in range(len(tuples) - 1):
                                loss += auxiliary_criterion(tuples[i], figures[:, i].type(torch.LongTensor))
                    if phase == TRAINING_PHASE:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss.append(loss)
                avg_loss[phase].append(torch.tensor(running_loss).mean())

        logs['loss'].append(torch.tensor(avg_loss[TRAINING_PHASE]).mean())
        logs['val_loss'].append(torch.tensor(avg_loss[VALIDATION_PHASE]).mean())

        temp_loss = torch.tensor(logs['loss'])
        temp_val_loss = torch.tensor(logs['val_loss'])


        format = 'Epoch %3d / %3d \n\t Training \t\t\t min: %8.5f, max: %8.5f, cur: %8.5f\n'
        results = (e+1, num_epoch, temp_loss.min(), temp_loss.max(), temp_loss[-1])

        if k_fold > 1:
            format += '\t Validation \t\t\t min: %8.5f, max: %8.5f, cur: %8.5f\n'
            results += (temp_val_loss.min(), temp_val_loss.max(), temp_val_loss[-1])
        
        # if e%10==0:
        #     print(format % results)
    model.eval()