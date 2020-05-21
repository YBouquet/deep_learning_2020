import torch
import time
import numpy as np

import dlc_practical_prologue as prologue
import io_bin_process
import io_num_process
import train

from bin_models import get_2nets, get_2nets_ws, get_2nets_ws_do,get_2nets_ws_bn, get_2nets_ws_required
from number_recognition_architectures import get_net, get_net2, get_lenet5



GETTERS_DICT =  {
                    'net': ('Number', get_net), #model given in the practicals
                    'net2': ('Number', get_net2), #model given in the practicals
                    'lenet5': ('Number', get_lenet5), #LeCun's model with correction due to different input format

                    'required' : ('Binary', get_required),

                    '2nets': ('Binary', get_2nets),
                    '2nets_ws': ('Binary', get_2nets_ws),

                    '2nets_ws_do': ('Binary', get_2nets_ws_do),
                    '2nets_ws_bn': ('Binary', get_2nets_ws_bn)
                }

PAIRS_NB = 1000
AUGMENTATION_FOLDS = 0
DATA_DOUBLING = False

NB_SIMULATIONS = 10

#models = [(Net(nb_hidden),"Net " + str(nb_hidden), 2e-3) for nb_hidden in nb_hidden_layers] + [(Net2(), "Net2", 1e-2), (LeNet5(), "LeNet5", 4e-2)]

def main(args):
    train_accuracies = []
    test_accuracies = []
    try:
        model_tuple = GETTERS_DICT[args.model.lower()]
    except KeyError:
        print('ERROR : model unknown')
        return
    for nb_simulation in range(NB_SIMULATIONS):
        torch.manual_seed(nb_simulation ** 4)
        m_model = model_tuple[1]()

        tr_input, tr_target, tr_figure_target, te_input, te_target,_ = prologue.generate_pair_sets(PAIRS_NB)
        if model_tuple[0] == 'Binary':
            tr_input, tr_target, tr_figure_target = io_bin_process.data_augmentation(tr_input, tr_target, tr_figure_target, PAIRS_NB, AUGMENTATION_FOLDS)
            if DATA_DOUBLING:
                tr_input, tr_target, tr_figure_target = io_bin_process.data_doubling(tr_input, tr_target, tr_figure_target)
        elif model_tuple[0] == 'Number':
            (tr_input, tr_figure_target, test_set_figures, test_target_figures, test_set_first_figures, test_set_second_figures, test_target_comparison) = io_num_process.formatting_input(run.PAIRS_NB)
            tr_target = io_num_process.one_hot_encoding(tr_figure_target)

        tic = time.perf_counter()
        temp = train.pretrain_train_model(m_model, tr_input, tr_target, tr_figure_target, args.optimizer, max(1, args.k_fold), args.batch_size, args.n_epochs, lr_train = args.learning_rate, beta = args.adam_beta1, weight_decay_train = args.weight_decay, weight_auxiliary_loss = args.weight_auxiliary_loss, num_epoch_pretrain = 0, lr_pretrain = 0., weight_decay_pretrain = 0.)
        toc = time.perf_counter()

        print(f"{nb_simulation+1}-th simulation trained in {toc - tic:0.2f} seconds.")

        if model_tuple[0] == 'Binary':
            nb_errors_train = io_bin_process.nb_classification_errors(m_model, tr_input, tr_target, args.batch_size)
            nb_errors_test = io_bin_process.nb_classification_errors(m_model, te_input, te_target, args.batch_size)
        elif model_tuple[0] == 'Number':
            nb_errors_train = io_num_process.compute_nb_recognition_errors(m_model, tr_input, tr_figure_target, args.batch_size) # for recognition!
            nb_errors_test = io_num_process.compute_nb_comparison_errors(m_model, test_set_first_figures, test_set_second_figures, test_target_comparison, args.batch_size)

        print(f"{nb_simulation+1}-th simulation, train accuracy = {100 * (1 - nb_errors_train / tr_input.size(0)):0.2f}%, test accuracy = {100 * (1 - nb_errors_test / te_input.size(0)):0.2f}%")
        train_accuracies.append(100 * (1 - nb_errors_train / tr_input.size(0)))
        test_accuracies.append(100 * (1 - nb_errors_test / te_input.size(0)))

        del(m_model)


    print('\t', f"Mean train accuracy = {np.mean(train_accuracies):0.2f}, Std train accuracy = {np.std(train_accuracies):0.2f}", "\n\t", f"Mean test accuracy = {np.mean(test_accuracies):0.2f}, Std test accuracy = {np.std(test_accuracies):0.2f}", '\n')

if __name__ == '__main__':
    main(prologue.get_args())
