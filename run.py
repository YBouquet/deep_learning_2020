#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:40:28 2020

@author: thomas
"""
from number_recognition_architectures import *
import recognition_training
import io_process




PAIRS_NB = 1000
MINI_BATCH_SIZE = 10


#nb_hidden_layers = [100, 200, 300]
nb_hidden_layers = [300]

models = [(Net(nb_hidden),"Net " + str(nb_hidden), 2e-3) for nb_hidden in nb_hidden_layers] + [(Net2(), "Net2", 1e-2), (LeNet5(), "LeNet5", 4e-2)]

(train_set_figures, train_target_figures,
 test_set_figures, test_target_figures,
 test_set_first_figures, test_set_second_figures, test_target_comparison) = io_process.formatting_input(PAIRS_NB)


for (model, name_model, eta_model) in models:
    print(name_model)
    recognition_training.train_model(model, train_set_figures, train_target_figures, MINI_BATCH_SIZE, eta_model)
    
    nb_train_recognition_errors = recognition_training.compute_nb_recognition_errors(model, train_set_figures, train_target_figures, MINI_BATCH_SIZE)
    io_process.print_error(name_model, 'train recognition', nb_train_recognition_errors, train_set_figures.size(0))
    
    nb_test_recognition_errors =recognition_training.compute_nb_recognition_errors(model, test_set_figures, test_target_figures, MINI_BATCH_SIZE)
    io_process.print_error(name_model, 'test recognition', nb_test_recognition_errors, test_set_figures.size(0))
    
    nb_test_comparison_errors = recognition_training.compute_nb_comparison_errors(model, test_set_first_figures, test_set_second_figures, test_target_comparison, MINI_BATCH_SIZE)
    io_process.print_error(name_model, 'test comparison', nb_test_comparison_errors, test_target_comparison.size(0))
    print('\n')