#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:28:32 2020

@author: thomas
"""
### Python modules
from matplotlib import pyplot as plt
import torch

### Project modules
import arguments
import modules as bf
import helpers as h


DICT = {
        'relu' : bf.ReLU,
        'tanh' : bf.Tanh
        }


def validation_sets(ratio):
    train_set, train_target, validation_set, validation_target = h.generate_sets(size = 1000)
    index = int(ratio * len(train_set))
    return train_set[:index], train_target[:index], train_set[index:], train_target[index:]


def main(args):
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    train_subset, train_subtarget, validation_set, validation_target = validation_sets(0.8)
    train_subtarget, validation_target = h.ohe(train_subtarget, validation_target)
    x = range(1, args.n_epochs+1)
    fig, ax = plt.subplots(dpi = 200) # Create a figure and an axes.
    for activation in ['relu','tanh']:
       
        activation_function = DICT[activation]
        m_model = bf.Sequential( bf.Linear(2,25),  activation_function(), bf.Linear(25,25), activation_function(), bf.Linear(25,25),  activation_function(), bf.Linear(25,2))
    
        train_subset_l, validation_set_l = h.train_model(m_model, train_subset, train_subtarget,validation_set, validation_target, lr = 1e-2, num_epoch = args.n_epochs, batch = args.batch_size)
        tr_error = 100 * h.nb_classification_errors(m_model, train_subset, train_subtarget, args.batch_size) / len(train_subset)
        va_error = 100 * h.nb_classification_errors(m_model, validation_set, validation_target, args.batch_size) / len(validation_set)
        print(f"Train accuracy = {100 - tr_error} %, validation accuracy = {100 - va_error} %")

        ax.semilogy(x, train_subset_l, label = activation + f" train subset, final accuracy = {100 - tr_error:0.2f}")
        ax.semilogy(x, validation_set_l, label = activation + f" validation set, final accuracy = {100 - va_error:0.2f}")
    ax.set_xlabel('Epoch')  # Add an x-label to the axes.
    ax.set_ylabel('MSE Loss')  # Add a y-label to the axes.
    ax.set_title('Comparison of the validation, ratio $\dfrac{validation\_set}{train\_set} = 0.2$')  # Add a title to the axes.
    ax.legend()
    filename = 'validation.png'
    fig.savefig(filename)

if __name__ == '__main__':
    main(arguments.get_args())
