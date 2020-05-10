#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 09:21:46 2020

@author: thomas
"""
### Python modules
import argparse

parser = argparse.ArgumentParser(description='Arguments file for project 2.')

# Training parameters
parser.add_argument('--n_iter', type = int, default = 10, help = 'Number of epochs')
parser.add_argument('--lr', type = float, default = 1e-2, help = 'Learning rate for the optimizer')
parser.add_argument('--acti_fct_1', type = str, default = 'relu', help = "Select the first activation function of the model")
parser.add_argument('--acti_fct_2', type = str, default = 'relu', help = "Select the second activation function of the model")
parser.add_argument('--acti_fct_3', type = str, default = 'relu', help = "Select the last activation function of the model")
parser.add_argument('--units', type = int, default = 25, help = "Select the number of units for hidden layers")
parser.add_argument('--k_fold', type = int, default = 1, help = 'Indicate how many folds are wanted for cross validation (cross validation deactivated by default)')
args = parser.parse_args()

def get_args():
    return args