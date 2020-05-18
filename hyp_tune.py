import torch
import math

import base64

import sys
from io import StringIO

import dlc_practical_prologue as prologue


from bin_models import get_2channels, get_2nets, get_2nets_ws
from bin_v2_models import get_2_one_channel, get_one_image, get_2_LeNet5
from number_recognition_architectures import get_net, get_net2, get_lenet5

from train import grid_search
import io_bin_process
import io_num_process

from saver import save_csv

GETTERS_DICT =  {
                    '2nets': ('Binary', get_2nets, (2,14,14)),
                    '2nets_ws': ('Binary', get_2nets_ws, (2,14,14)),
                }

PAIRS_NB = 1000
AUGMENTATION_FOLDS = 0
DATA_DOUBLING = False

#models = [(Net(nb_hidden),"Net " + str(nb_hidden), 2e-3) for nb_hidden in nb_hidden_layers] + [(Net2(), "Net2", 1e-2), (LeNet5(), "LeNet5", 4e-2)]


def main(args):
    if args.seed >= 0:
        torch.manual_seed(args.seed)
    try:
        model_tuple = GETTERS_DICT[args.model.lower()]
    except KeyError:
        print('ERROR : model unknown')
        return

    if model_tuple[0] == 'Binary':
        tr_input, tr_target, tr_figure_target, te_input, te_target,_ = prologue.generate_pair_sets(PAIRS_NB)
        tr_input, tr_target, tr_figure_target = io_bin_process.data_augmentation(tr_input, tr_target, tr_figure_target, PAIRS_NB, AUGMENTATION_FOLDS)
        if DATA_DOUBLING:
            tr_input, tr_target, tr_figure_target = io_bin_process.data_doubling(tr_input, tr_target, tr_figure_target)
        #tr_target, te_target = io_bin_process.targets_reshape(tr_target, te_target)
    else:
        (tr_input, train_target,
        test_set_figures, test_target_figures,
        test_set_first_figures, test_set_second_figures, test_target_comparison) = io_num_process.formatting_input(PAIRS_NB)
        tr_target = io_num_process.one_hot_encoding(train_target)
    m_model = model_tuple[1]()

    lrt_array = torch.logspace(-5, -1, steps = 10)
    wdt_array = torch.cat((torch.logspace(-7,-1, steps = 9),torch.Tensor([0])))
    num_epoch_pretrain = 50
    lrp_array = lrt_array.clone()
    wdp_array = wdt_array.clone()
    wal_array = torch.cat((torch.linspace(0.2,0.99, steps = 9), torch.Tensor([1.])))

    print("---------- START GRID SEARCH ---------------")
    try :
        train_results, pretrain_results, independant_bests, hyperparameters = grid_search(m_model, 'grid_search.pth', tr_input, tr_target, tr_figure_target, max(1,args.k_fold),  args.batch_size, args.n_iter,\
                                   lrt_array = lrt_array, wdt_array = wdt_array, num_epoch_pretrain = num_epoch_pretrain, lrp_array = lrp_array, wdp_array = wdp_array, wal_array = wal_array)
        print(train_results)
        print(pretrain_results)
        print(independant_bests)
        with open('hyperparameters_'+args.model.lower()+'.txt', 'w+') as f:
            f.write(args.model.lower() + '\n')
            f.write('\n%f\n%f\n%f\n%f\n%f\n' % tuple(hyperparameters))
    except KeyboardInterrupt:
        del(m_model)
        return
    print("----------- END GRID SEARCH ----------------")
if __name__ == '__main__':
    # miscelaneous parameters
    #parser.add_argument('--seed', type=int, default=1, help = 'Random seed (default 1, < 0 is no seeding)')
    #parser.add_argument('--make_gif', type=bool, default=True)
    #parser.add_argument('--device', type=str, default='cpu')
    main(prologue.get_args())
