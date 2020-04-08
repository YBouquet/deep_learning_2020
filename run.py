import argparse

import dlc_practical_prologue as prologue

import torch
import math

from bin_models import get_2channels
from bin_training import train_model
from io_bin_process import targets_reshape, error_rate_classification
from saver import save_csv

GETTERS_DICT =   {'2Channels': ('Binary', get_2channels)}

def main(args):
    if args.seed >= 0:
        torch.manual_seed(args.seed)
    tr_input, tr_bin_target, tr_num_target, te_input, te_bin_target, te_num_target = prologue.generate_pair_sets(1000)
    tr_bin_target, te_bin_target = targets_reshape(tr_bin_target, te_bin_target)
    m_model = GETTERS_DICT[args.model][1]()
    print("---------- START TRAINING ---------------")
    train_model(m_model, tr_input, tr_bin_target, args.batch_size, args.lr, num_epoch = args.n_iter)
    print("----------- END TRAINING ----------------")
    nb_errors_train = error_rate_classification(m_model, tr_input, tr_bin_target, args.batch_size)
    print('train error '+ args.model +' {:0.2f}% {:d}/{:d}'.format((100 * nb_errors_train) / tr_input.size(0),
                                                      nb_errors_train, tr_input.size(0)))
    nb_errors_test = error_rate_classification(m_model, te_input, te_bin_target, args.batch_size)
    error_rate = (100 * nb_errors_test) / te_input.size(0)
    print('test error '+ args.model +' {:0.2f}% {:d}/{:d}'.format(error_rate,
                                                      nb_errors_test, te_input.size(0)))
    if args.save :
        infos = {}
        infos['target'] = GETTERS_DICT[args.model][0]
        infos['model'] =  args.model
        infos['optimizer'] = 'SGD'
        infos['epochs'] = args.n_iter
        infos['minibatch_size'] = args.batch_size
        infos['accuracy'] = 100. - error_rate
        infos['f1_score'] = math.nan
        infos[ 'roc'] = math.nan
        infos['comments'] = args.comments
        save_csv('test_report.csv', infos)

if __name__ == '__main__':
    # miscelaneous parameters
    #parser.add_argument('--seed', type=int, default=1, help = 'Random seed (default 1, < 0 is no seeding)')
    #parser.add_argument('--make_gif', type=bool, default=True)
    #parser.add_argument('--device', type=str, default='cpu')
    main(prologue.get_args())
