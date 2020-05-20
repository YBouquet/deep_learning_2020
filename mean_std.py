import torch
import time
import numpy as np

import dlc_practical_prologue as prologue
import io_bin_process
import io_num_process
import train
import run


MODEL = '2nets_ws'
model_tuple = run.GETTERS_DICT[MODEL]
BATCH_SIZE = 5
LR_PRETRAIN = 0.001668
WD_PRETRAIN = 0.000001
LR = 6e-4
WAL = 1.
WD_TRAIN = 1e-5
NB_EPOCHS = 50
NB_SIMULATIONS = 10


#models = [(Net(nb_hidden),"Net " + str(nb_hidden), 2e-3) for nb_hidden in nb_hidden_layers] + [(Net2(), "Net2", 1e-2), (LeNet5(), "LeNet5", 4e-2)]

def main():
    train_accuracies = []
    test_accuracies = []
    for nb_simulation in range(NB_SIMULATIONS):
        torch.manual_seed(nb_simulation**4)
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
        temp = train.pretrain_train_model(m_model, tr_input, tr_target, tr_figure_target, 1, BATCH_SIZE, NB_EPOCHS, lr_train = LR, weight_decay_train = WD_TRAIN, weight_auxiliary_loss = WAL, num_epoch_pretrain = 0, lr_pretrain = LR_PRETRAIN, weight_decay_pretrain = WD_PRETRAIN)
        toc = time.perf_counter()


        if model_tuple[0] == 'Binary':
            nb_errors_train = io_bin_process.nb_classification_errors(m_model, tr_input, tr_target, BATCH_SIZE)
            nb_errors_test = io_bin_process.nb_classification_errors(m_model, te_input, te_target, BATCH_SIZE)
        elif model_tuple[0] == 'Number':
            nb_errors_train = io_num_process.compute_nb_recognition_errors(m_model, tr_input, tr_figure_target, BATCH_SIZE) # for recognition!
            nb_errors_test = io_num_process.compute_nb_comparison_errors(m_model, test_set_first_figures, test_set_second_figures, test_target_comparison, BATCH_SIZE)

        print(f"{nb_simulation+1}-th simulation, train accuracy = {100 * (1 - nb_errors_train / tr_input.size(0)):0.2f}%, test accuracy = {100 * (1 - nb_errors_test / te_input.size(0)):0.2f}%")
        train_accuracies.append(100 * (1 - nb_errors_train / tr_input.size(0)))
        test_accuracies.append(100 * (1 - nb_errors_test / te_input.size(0)))

        del(m_model)

        #print(f"{nb_simulation+1}-th simulation trained in {toc - tic:0.2f} seconds.")
    print('\t', f"Mean train accuracy = {np.mean(train_accuracies):0.2f}, Std train accuracy = {np.std(train_accuracies):0.2f}", "\n\t", f"Mean test accuracy = {np.mean(test_accuracies):0.2f}, Std test accuracy = {np.std(test_accuracies):0.2f}", '\n')

if __name__ == '__main__':
    main()
