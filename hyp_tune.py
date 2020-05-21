import torch
import math

import base64

import sys
from io import StringIO

import dlc_practical_prologue as prologue


from bin_models import get_2nets, get_2nets_ws, get_2nets_ws_do
from bin_v2_models import get_2_one_channel, get_one_image, get_2_LeNet5, get_2channels
from number_recognition_architectures import get_net, get_net2, get_lenet5

from train import grid_search
import io_bin_process
import io_num_process

from saver import save_csv

GETTERS_DICT =  {
                    '2nets': ('Binary', get_2nets),
                    '2nets_ws': ('Binary', get_2nets_ws),
                    '2nets_ws_do': ('Binary', get_2nets_ws_do),

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

    tr_input, tr_target, tr_figure_target, te_input, te_target,_ = prologue.generate_pair_sets(PAIRS_NB)
    tr_input, tr_target, tr_figure_target = io_bin_process.data_augmentation(tr_input, tr_target, tr_figure_target, PAIRS_NB, AUGMENTATION_FOLDS)
    if DATA_DOUBLING:
        tr_input, tr_target, tr_figure_target = io_bin_process.data_doubling(tr_input, tr_target, tr_figure_target)
    #tr_target, te_target = io_bin_process.targets_reshape(tr_target, te_target)
    m_model = model_tuple[1]()

    lrt_array = torch.logspace(-6, -1, steps = 10)
    beta_array = torch.linspace(0.5, 0.9, steps = 5)
    wdt_array = torch.cat((torch.logspace(-7,-1, steps = 9),torch.Tensor([0])))
    num_epoch_pretrain = 0
    lrp_array = [0.] #lrt_array.clone()
    wdp_array = [0.] #wdt_array.clone()
    wal_array = torch.cat((torch.linspace(0.2,0.99, steps = 9), torch.Tensor([1.])))

    print("---------- START GRID SEARCH ---------------")
    try :
        train_results, pretrain_results, independant_bests, hyperparameters = grid_search(m_model, 'grid_search.pth', tr_input, tr_target, tr_figure_target, 'adam', max(1,args.k_fold),  args.batch_size, args.n_epochs,\
                                   lrt_array = lrt_array, beta_array = beta_array,  wdt_array = wdt_array, num_epoch_pretrain = num_epoch_pretrain, lrp_array = lrp_array, wdp_array = wdp_array, wal_array = wal_array)
        print(train_results)
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
