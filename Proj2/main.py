#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 08:51:08 2020

@author: thomas
"""
### Python modules
#from matplotlib import pyplot as plt
import torch

### Project modules
import modules as bf
import helpers as h
import torch
import time
import arguments

have_matplotlib = True
try:
    from matplotlib import pyplot as plt
except ImportError:
    have_matplotlib = False

### To do: plot with the ratio


DICT = {
        'relu' : bf.ReLU,
        'tanh' : bf.Tanh
        }

NB_SIMULATIONS = 10
NORMALIZE = True


def main(args):
    torch.set_grad_enabled(False)
    train_errors = torch.zeros(NB_SIMULATIONS)
    test_errors = torch.zeros(NB_SIMULATIONS)


    for nb_simulation in range(NB_SIMULATIONS):
        torch.manual_seed(nb_simulation)
        train_set, train_target,test_set, test_target = h.generate_sets(size = 1000)
        if NORMALIZE:
            train_set, test_set = h.normalize(train_set, test_set)
        train_target, test_target = h.ohe(train_target, test_target)
        activation_function = DICT[args.activation]
        m_model = bf.Sequential( bf.Linear(2,25),  activation_function(), bf.Linear(25,25), activation_function(), bf.Linear(25,25),  activation_function(), bf.Linear(25,2))

        tic = time.perf_counter()
        #logging the losses for creating a graph for the training loss vs the test loss during the process  at the last iteration
        train_l, test_l = h.train_model(m_model, train_set, train_target,test_set, test_target, lr = 1e-2, num_epoch = args.n_epochs, batch = args.batch_size)
        toc = time.perf_counter()

        print(f"{nb_simulation+1}-th simulation trained in {toc - tic:0.2f} seconds.")
        tr_error =h.nb_classification_errors(m_model, train_set, train_target, args.batch_size) / 10
        te_error = h.nb_classification_errors(m_model, test_set, test_target, args.batch_size) / 10
        train_errors[nb_simulation] = tr_error
        test_errors[nb_simulation] = te_error
        print(f"{nb_simulation+1}-th simulation, train errors = {tr_error} %, test accuracy = {te_error} %")
    print('\t', f"Mean train error = {train_errors.mean():0.2f} %, Std train error = {train_errors.std():0.2f}", "\n\t", f"Mean test error = {test_errors.mean():0.2f} %, Std test error = {test_errors.std():0.2f}", '\n')


    if have_matplotlib :
        x = range(args.n_epochs - 1)
        fig, ax = plt.subplots(dpi = 200)  # Create a figure and an axes.
        ax.plot(x, train_l[1:], 'r', label = 'train')
        ax.plot(x, test_l[1:], 'b', label = 'test')
        ax.set_xlabel('Epochs')  # Add an x-label to the axes.
        ax.set_ylabel('MSE Loss')  # Add a y-label to the axes.
        title = 'Training with'+ args.activation +'activation'
        ax.set_title(title)  # Add a title to the axes.
        ax.legend()
        filename = 'training_'+args.activation + '.png'
        fig.savefig(filename)

    
if __name__ == '__main__':
    main(arguments.get_args())
