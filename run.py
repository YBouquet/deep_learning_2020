"""
Created on Mon Apr  6 10:40:28 2020

@authors: Thomas Fry, Yann BOUQUET
"""

import argparse
import torch
import math

import base64

have_sum = True
import sys
from io import StringIO

try:
    from torchsummary import summary
except ImportError:
    have_sum = False

import dlc_practical_prologue as prologue

from bin_models import get_2channels, get_2nets
from number_recognition_architectures import get_net, get_net2, get_lenet5

from train import train_model
import io_bin_process
import io_num_process

from saver import save_csv

GETTERS_DICT =  {
                    '2Channels': ('Binary', get_2channels, (2,14,14)),
                    '2Nets': ('Binary', get_2nets, (2,14,14)),
                    'Net': ('Number', get_net, (1,14,14)),
                    'Net2': ('Number', get_net2, (1,14,14)),
                    'LeNet5': ('Number', get_lenet5, (1,14,14))
                }

PAIRS_NB = 1000
AUGMENTATION_FOLDS = 9

#models = [(Net(nb_hidden),"Net " + str(nb_hidden), 2e-3) for nb_hidden in nb_hidden_layers] + [(Net2(), "Net2", 1e-2), (LeNet5(), "LeNet5", 4e-2)]

def print_error(name, error_type, nb_errors, size_):
    error_rate = (100 * nb_errors) / size_
    print(error_type + ' error '+ name +': {:0.2f}% {:d}/{:d}'.format(error_rate,
                                                      nb_errors, size_))
    return 100. - error_rate

def main(args):
    if args.seed >= 0:
        torch.manual_seed(args.seed)
    try:
        model_tuple = GETTERS_DICT[args.model]
        if model_tuple[0] == 'Binary':
            tr_input, tr_target, tr_figure_target, te_input, te_target,_ = prologue.generate_pair_sets(PAIRS_NB)
            tr_input, tr_target, tr_figure_target = io_bin_process.data_augmentation(tr_input, tr_target, tr_figure_target, PAIRS_NB, AUGMENTATION_FOLDS)
            tr_target, te_target = io_bin_process.targets_reshape(tr_target, te_target)
        else:
            (tr_input, train_target,
            test_set_figures, test_target_figures,
            test_set_first_figures, test_set_second_figures, test_target_comparison) = io_num_process.formatting_input(PAIRS_NB)
            tr_target = io_num_process.one_hot_encoding(train_target)

        m_model = model_tuple[1]()
        print("---------- START TRAINING ---------------")
        train_model(m_model, tr_input, tr_target, args.batch_size, args.lr, num_epoch = args.n_iter)
        print("----------- END TRAINING ----------------")

        if model_tuple[0] == 'Binary':
            nb_errors_train = io_bin_process.nb_classification_errors(m_model, tr_input, tr_target, args.batch_size)
            print_error(args.model, 'train', nb_errors_train, tr_input.size(0))
            nb_errors_test = io_bin_process.nb_classification_errors(m_model, te_input, te_target, args.batch_size)
            accuracy = print_error(args.model, 'test', nb_errors_test, te_input.size(0))
        else:
            nb_train_recognition_errors = io_num_process.compute_nb_recognition_errors(m_model, tr_input, train_target, args.batch_size)
            print_error(args.model, 'train recognition', nb_train_recognition_errors, tr_input.size(0))
            nb_test_recognition_errors = io_num_process.compute_nb_recognition_errors(m_model, test_set_figures, test_target_figures, args.batch_size)
            print_error(args.model, 'test recognition', nb_test_recognition_errors, test_set_figures.size(0))
            nb_test_comparison_errors = io_num_process.compute_nb_comparison_errors(m_model, test_set_first_figures, test_set_second_figures, test_target_comparison, args.batch_size)
            accuracy = print_error(args.model, 'test comparison', nb_test_comparison_errors, test_target_comparison.size(0))
        del(m_model)

        if args.save and have_sum :
            dummy = model_tuple[1]()

            stdout = sys.stdout
            s = StringIO()
            sys.stdout = s
            if torch.cuda.is_available():
                dummy.cuda()
                summary(dummy, model_tuple[2])
            else:
                print("no summary")
            sys.stdout = stdout
            s.seek(0)
            infos = {}
            infos['target'] = GETTERS_DICT[args.model][0]
            infos['model'] =  args.model
            infos['summary'] = base64.b64encode(s.read().encode('utf-8',errors = 'strict'))
            infos['optimizer'] = 'SGD'
            infos['learning_rate'] = args.lr
            infos['epochs'] = args.n_iter
            infos['minibatch_size'] = args.batch_size
            infos['accuracy'] = accuracy
            infos['f1_score'] = math.nan
            infos[ 'roc'] = math.nan
            if len(args.comments) == 0:
                print("Don't put an empty string as a comment!")
            else :
                infos['b64_comments'] = base64.b64encode(args.comments.encode('utf-8',errors = 'strict'))
            save_csv('test_report.csv', infos)
    except KeyError:
        print('ERROR : model unknown')
if __name__ == '__main__':
    # miscelaneous parameters
    #parser.add_argument('--seed', type=int, default=1, help = 'Random seed (default 1, < 0 is no seeding)')
    #parser.add_argument('--make_gif', type=bool, default=True)
    #parser.add_argument('--device', type=str, default='cpu')
    main(prologue.get_args())
