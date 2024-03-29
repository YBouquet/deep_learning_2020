import torch
import numpy as np

import dlc_practical_prologue as prologue
import io_bin_process
import io_num_process
import train
from main import GETTERS_DICT, DATA_DOUBLING

import matplotlib.pyplot as plt

from train import TRAINING, TRAINING_PHASE, VALIDATION_PHASE

MODEL_OVERFIT = '2nets_ws'
MODEL_DO = '2nets_ws_do'

BATCH_SIZE = 100
LRS = {MODEL_OVERFIT: (6e-3, 0.8), MODEL_DO: (1e-3, 0.9)}
WAL = 1.
WD_TRAIN = 0.
NB_EPOCHS = 50
NB_SIMULATIONS = 10

PAIRS_NB = 1000
AUGMENTATION_FOLDS = 0
DATA_DOUBLING = False

K_FOLD = 5
#models = [(Net(nb_hidden),"Net " + str(nb_hidden), 2e-3) for nb_hidden in nb_hidden_layers] + [(Net2(), "Net2", 1e-2), (LeNet5(), "LeNet5", 4e-2)]

def main(args):
    fig, ax = plt.subplots(dpi = 200)  # Create a figure and an axes.
    for model_name in [MODEL_OVERFIT, MODEL_DO]:
        model_tuple = GETTERS_DICT[model_name]
        torch.manual_seed(args.seed)
        m_model = model_tuple[1]()

        tr_input, tr_target, tr_figure_target, te_input, te_target,_ = prologue.generate_pair_sets(PAIRS_NB)
        if model_tuple[0] == 'Binary':
            tr_input, tr_target, tr_figure_target = io_bin_process.data_augmentation(tr_input, tr_target, tr_figure_target, PAIRS_NB, AUGMENTATION_FOLDS)
            if DATA_DOUBLING:
                tr_input, tr_target, tr_figure_target = io_bin_process.data_doubling(tr_input, tr_target, tr_figure_target)

        elif model_tuple[0] == 'Number':
            (tr_input, tr_figure_target, test_set_figures, test_target_figures, test_set_first_figures, test_set_second_figures, test_target_comparison) = io_num_process.formatting_input(PAIRS_NB)
            tr_target = io_num_process.one_hot_encoding(tr_figure_target)

        temp = train.pretrain_train_model(m_model, tr_input, tr_target, tr_figure_target, torch.nn.CrossEntropyLoss, 'adam', K_FOLD, BATCH_SIZE, NB_EPOCHS, lr_train = LRS[model_name][0], beta = LRS[model_name][1], weight_decay_train = WD_TRAIN, weight_auxiliary_loss = WAL)
        x = np.arange(1, NB_EPOCHS+1)
        ax.plot(x, temp[TRAINING][TRAINING_PHASE], label="Train" + int(model_name == MODEL_DO) * " with dropout" + f", final loss = {temp[TRAINING][TRAINING_PHASE][-1]:0.2f}")  # Plot some data on the axes.
        ax.plot(x, temp[TRAINING][VALIDATION_PHASE], label="Validation" + int(model_name == MODEL_DO) * " with dropout" + f", final loss = {temp[TRAINING][VALIDATION_PHASE][-1]:0.2f}")  # Plot more data on the axes...
        ax.set_xlabel('Epoch') # Add an x-label to the axes.
        ax.set_ylabel('Cross Entropy Loss') # Add a y-label to the axes.
        title = 'Training 2nets_ws with auxiliary loss\n and 5 fold cross validation'
        ax.set_title(title) # Add a title to the axes.
        ax.legend()
        del(m_model)
    filename = 'comparison.png'
    fig.savefig(filename)

if __name__ == '__main__':
    main(prologue.get_args())
