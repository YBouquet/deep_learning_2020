#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:22:16 2020

@author: thomas
"""
import torch
import functions




def train_model(model, train_input, train_target, mini_batch_size, eta = 1e-2):
    criterion = torch.nn.MSELoss()

    for e in range(25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            exp_proba = functions.expected_probabilities(train_target.narrow(0, b, mini_batch_size))
            loss = criterion(output, exp_proba)
            model.zero_grad() # Initialization of the gradient to zero.
            loss.backward() # Compute the gradient.
            sum_loss = sum_loss + loss.item()
            with torch.no_grad(): # Apply gradient descent.
                for p in model.parameters():
                    p -= eta * p.grad
        if e%10 == 0:
            print(e, format(sum_loss, '.3f'))
    print(e, format(sum_loss, '.3f'))


def compute_nb_recognition_errors(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        for k in range(mini_batch_size):
            if output[k].max(0)[1] != target[b + k].item():
                nb_errors = nb_errors + 1

    return nb_errors


def compute_nb_comparison_errors(model, first_input, second_input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, first_input.size(0), mini_batch_size):
        first_output = model(first_input.narrow(0, b, mini_batch_size))
        second_output = model(second_input.narrow(0, b, mini_batch_size))
        for k in range(mini_batch_size):
            if (first_output[k].max(0)[1] <= second_output[k].max(0)[1]) != target[b + k].item():
                nb_errors = nb_errors + 1

    return nb_errors