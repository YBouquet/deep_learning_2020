#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:14:19 2020

@author: Thomas Fry
"""

import torch
import dlc_practical_prologue as prologue



def one_hot_encoding(target):
    """One hot encoding for the targets representing the number categories (from 0 to 9)

    Parameters
    ----------
    target : Tensor
        Figures of the train set (classes from 0 to 9)
    """
    mini_batch_size = len(target)
    results = torch.zeros(mini_batch_size, 10)
    index = 0
    for figure in target:
        results[index][figure] = 1.
        index += 1
    return results


def formatting_input(pairs_nb):
    """Generate the pairs cut the pairs to have a set of images to recognize

    Parameters
    ----------
    pairs_nb : int
        number of pairs we want to generate

    Return
    ----------
        sets of pairs_nb * 2 samples, each representing one image and the digit it represents
    """
    (train_set, train_target_comparison, train_target_fig,
     test_set, test_target_comparison, test_target_fig) = prologue.generate_pair_sets(pairs_nb)

    train_set_figures = torch.cat((train_set[:,0],train_set[:,1])).view(-1, 1, 14, 14)
    train_target_figures = torch.cat((train_target_fig[:,0],train_target_fig[:,1]))

    test_set_figures = torch.cat((test_set[:,0],test_set[:,1])).view(-1, 1, 14, 14)
    test_target_figures = torch.cat((test_target_fig[:,0],test_target_fig[:,1]))

    test_set_first_figures = test_set[:,0].view(-1, 1, 14, 14)
    test_set_second_figures = test_set[:,1].view(-1, 1, 14, 14)

    return(train_set_figures, train_target_figures,
           test_set_figures, test_target_figures,
           test_set_first_figures, test_set_second_figures, test_target_comparison)


def compute_nb_recognition_errors(model, input, target, mini_batch_size):
    """Count the number of errors the model makes during its classification

    Parameters
    ----------
    model : nn.Module
        Instance of the neural network model
    test_input : Tensor
        Test set as input of the model
    target : Tensor
        Targets of the test set
    mini_batch_size: int
        Size of the batch we want to fill in the model at each iteration

    Return
    ------
    nb_errors: int
        number of errors made during the classification
    """
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        for k in range(mini_batch_size):
            if output[k].max(0)[1] != target[b + k].item():
                nb_errors = nb_errors + 1

    return nb_errors


def compute_nb_comparison_errors(model, first_input, second_input, target, mini_batch_size):
    """Count the number of errors the model makes by letting the model recognize
    the digits while the comparison is made with the outputs of the model

    Parameters
    ----------
    model : nn.Module
        Instance of the neural network model
    first_input : Tensor
        Test set with the first images from the initial pairs
    first_input : Tensor
        Test set with the last images from the initial pairs
    target : Tensor
        Targets of the test set (class from 0 to 1)
    mini_batch_size: int
        Size of the batch we want to fill in the model at each iteration


    Return
    ------
    nb_errors: int
        number of errors made during the classification
    """
    nb_errors = 0

    for b in range(0, first_input.size(0), mini_batch_size):
        first_output = model(first_input.narrow(0, b, mini_batch_size))
        second_output = model(second_input.narrow(0, b, mini_batch_size))
        for k in range(mini_batch_size):
            if (first_output[k].max(0)[1] <= second_output[k].max(0)[1]) != target[b + k].item():
                nb_errors = nb_errors + 1

    return nb_errors
