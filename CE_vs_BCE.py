#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:45:15 2020

@author: thomas
"""
import torch
import time
import numpy as np

import dlc_practical_prologue as prologue
import io_bin_process
import io_num_process
import train
import run

CRITERION_TYPES = ['CE']
#CRITERION_TYPES = ['MSE']
AUX_CRITERION_TYPES = ['CE']
#AUX_CRITERION_TYPES = ['MSE']
MODELS = ['2nets', '2nets_ws']
BATCH_SIZE = 5
LR = 5e-3
NB_EPOCHS = 100
NB_SIMULATIONS = 10


for MODEL in MODELS:
    model_tuple = run.GETTERS_DICT[MODEL]
    print(MODEL)
    for CRITERION_TYPE in CRITERION_TYPES:
        for AUX_CRITERION_TYPE in AUX_CRITERION_TYPES:
            train_accuracies = []
            test_accuracies = []
            print(CRITERION_TYPE, AUX_CRITERION_TYPE)
            for nb_simulation in range(NB_SIMULATIONS):
                torch.manual_seed(nb_simulation**4)
                m_model = model_tuple[1]()
                
                tr_input, tr_target, tr_figure_target, te_input, te_target,_ = prologue.generate_pair_sets(run.PAIRS_NB)
                if model_tuple[0] == 'Binary':
                    tr_input, tr_target, tr_figure_target = io_bin_process.data_augmentation(tr_input, tr_target, tr_figure_target, run.PAIRS_NB, run.AUGMENTATION_FOLDS)
                    if run.DATA_DOUBLING:
                        tr_input, tr_target, tr_figure_target = io_bin_process.data_doubling(tr_input, tr_target, tr_figure_target)
                    if CRITERION_TYPE == 'BCE':
                        tr_target, te_target = io_bin_process.targets_reshape(tr_target, te_target)
                    tic = time.perf_counter()
                    train.train_model(m_model, tr_input, tr_target, tr_figure_target, 1, BATCH_SIZE, LR, NB_EPOCHS, auxiliary_loss = False, criterion_type = CRITERION_TYPE, aux_criterion_type = AUX_CRITERION_TYPE)
                    toc = time.perf_counter()
                    
                elif model_tuple[0] == 'Number':
                    (tr_input, tr_figure_target, test_set_figures, test_target_figures, test_set_first_figures, test_set_second_figures, test_target_comparison) = io_num_process.formatting_input(run.PAIRS_NB)
                    tic = time.perf_counter()
                    train.train_model(m_model, tr_input, tr_figure_target, tr_figure_target, 1, BATCH_SIZE, LR, NB_EPOCHS, auxiliary_loss = False, criterion_type = CRITERION_TYPE, decrease_lr=False)
                    toc = time.perf_counter()
                
                
                if model_tuple[0] == 'Binary':
                    nb_errors_train = io_bin_process.nb_classification_errors(m_model, tr_input, tr_target, BATCH_SIZE, criterion_type=CRITERION_TYPE)
                    nb_errors_test = io_bin_process.nb_classification_errors(m_model, te_input, te_target, BATCH_SIZE, criterion_type=CRITERION_TYPE)
                elif model_tuple[0] == 'Number':
                    nb_errors_train = io_num_process.compute_nb_recognition_errors(m_model, tr_input, tr_figure_target, BATCH_SIZE) # for recognition!
                    nb_errors_test = io_num_process.compute_nb_comparison_errors(m_model, test_set_first_figures, test_set_second_figures, test_target_comparison, BATCH_SIZE)
        
                print(f"{nb_simulation+1}-th simulation, train accuracy = {100 * (1 - nb_errors_train / tr_input.size(0)):0.2f}%, test accuracy = {100 * (1 - nb_errors_test / te_input.size(0)):0.2f}%")
                train_accuracies.append(100 * (1 - nb_errors_train / tr_input.size(0)))
                test_accuracies.append(100 * (1 - nb_errors_test / te_input.size(0)))
                
                del(m_model)
                
                #print(f"{nb_simulation+1}-th simulation trained in {toc - tic:0.2f} seconds.")
            print('\t', f"Mean train accuracy = {np.mean(train_accuracies):0.2f}, Std train accuracy = {np.std(train_accuracies):0.2f}", "\n\t", f"Mean test accuracy = {np.mean(test_accuracies):0.2f}, Std test accuracy = {np.std(test_accuracies):0.2f}", '\n')