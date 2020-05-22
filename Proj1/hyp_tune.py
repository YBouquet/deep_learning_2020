import torch

import dlc_practical_prologue as prologue


from models import get_2nets, get_2nets_ws, get_2nets_ws_do

from train import grid_search
import io_bin_process
import io_num_process

GETTERS_DICT =  {
                    '2nets_ws_do': ('Binary', get_2nets_ws_do)
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

    lrt_array = torch.logspace(-4, -3, steps = 3)
    beta_array = [0.9]#torch.linspace(0.5, 0.9, steps = 5)
    wdt_array = torch.logspace(-6,-4, steps = 3)
    num_epoch_pretrain = 0
    #lrp_array = [] #lrt_array.clone()
    #wdp_array = [] #wdt_array.clone()
    wal_array = [1.]

    print("---------- START SEARCH ---------------")
    try :
        train_results, pretrain_results, independant_bests, hyperparameters = grid_search(m_model, 'grid_search.pth', tr_input, tr_target, tr_figure_target, torch.nn.CrossEntropyLoss, 'adam', max(1,args.k_fold),  args.batch_size, args.n_epochs,\
                                   lrt_array = lrt_array, beta_array = beta_array,  wdt_array = wdt_array)

        with open('hyperparameters_'+args.model.lower()+'.txt', 'w+') as f:
            f.write(args.model.lower() + '\n')
            f.write('\nlearning_rate = %f\nbeta = %f\nweight_decay = %f \n' % tuple(hyperparameters[:3]))

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
