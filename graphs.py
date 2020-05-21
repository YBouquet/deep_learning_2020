import torch
import time
import numpy as np

import dlc_practical_prologue as prologue
import io_bin_process
import io_num_process
import train
import run

import numpy as np
import matplotlib.pyplot as plt

from train import TRAINING, TRAINING_PHASE, VALIDATION_PHASE

MODEL_OVERFIT = '2nets_ws'
MODEL_DO = '2nets_ws_do'
DICT_ = run.GETTERS_DICT
BATCH_SIZE = 5
LR_PRETRAIN = 0.001668
WD_PRETRAIN = 0.000001
LRS = {MODEL_OVERFIT: (7e-5, 0.5), MODEL_DO: (6e-4, 0.6)}
WAL = 1.
WD_TRAIN = 0.
NB_EPOCHS = 100
NB_SIMULATIONS = 10

K_FOLD = 5
#models = [(Net(nb_hidden),"Net " + str(nb_hidden), 2e-3) for nb_hidden in nb_hidden_layers] + [(Net2(), "Net2", 1e-2), (LeNet5(), "LeNet5", 4e-2)]

def main(args):
    train_accuracies = []
    test_accuracies = []
    for model_name in [MODEL_OVERFIT, MODEL_DO]:
        model_tuple = DICT_[model_name]
        torch.manual_seed(args.seed)
        m_model = model_tuple[1]()

        tr_input, tr_target, tr_figure_target, te_input, te_target,_ = prologue.generate_pair_sets(run.PAIRS_NB)
        if model_tuple[0] == 'Binary':
            tr_input, tr_target, tr_figure_target = io_bin_process.data_augmentation(tr_input, tr_target, tr_figure_target, run.PAIRS_NB, run.AUGMENTATION_FOLDS)
            if run.DATA_DOUBLING:
                tr_input, tr_target, tr_figure_target = io_bin_process.data_doubling(tr_input, tr_target, tr_figure_target)

        elif model_tuple[0] == 'Number':
            (tr_input, tr_figure_target, test_set_figures, test_target_figures, test_set_first_figures, test_set_second_figures, test_target_comparison) = io_num_process.formatting_input(run.PAIRS_NB)
            tr_target = io_num_process.one_hot_encoding(tr_figure_target)

        tic = time.perf_counter()
        temp = train.pretrain_train_model(m_model, tr_input, tr_target, tr_figure_target,'adam', K_FOLD, BATCH_SIZE, NB_EPOCHS, lr_train = LRS[model_name][0], beta = LRS[model_name][1], weight_decay_train = WD_TRAIN, weight_auxiliary_loss = WAL, num_epoch_pretrain = 0, lr_pretrain = LR_PRETRAIN, weight_decay_pretrain = WD_PRETRAIN)
        toc = time.perf_counter()


        x = np.arange(NB_EPOCHS)
        fig, ax = plt.subplots()  # Create a figure and an axes.
        ax.plot(x, temp[TRAINING][TRAINING_PHASE], label='Train')  # Plot some data on the axes.
        ax.plot(x, temp[TRAINING][VALIDATION_PHASE], label='Validation')  # Plot more data on the axes...
        ax.set_xlabel('Epochs')  # Add an x-label to the axes.
        ax.set_ylabel('Cross Entropy Loss')  # Add a y-label to the axes.
        title = 'Training 2Nets_ws with auxiliary loss'
        if model_name == MODEL_DO :
            title +=', dropout '
        title += ' and k fold cross validation'
        ax.set_title(title)  # Add a title to the axes.
        ax.legend()
        filename = 'training_' + model_name +'.png'
        fig.savefig('test.png')

        del(m_model)

if __name__ == '__main__':
    main(prologue.get_args())
