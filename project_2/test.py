#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 08:51:08 2020

@author: thomas
"""
### Python modules
#from matplotlib import pyplot as plt
import torch

### Project modules
import modules
import helper
import arguments

### To do: plot with the ratio 


DICT = { 
        'relu' : modules.ReLU,
        'tanh' : modules.Tanh
        }


# Implémenter ratio validation, plot graph
# tanh not working
def main(args):
    torch.set_grad_enabled(False)
    torch.manual_seed(1)
    train_set, train_target,test_set, test_target = helper.generate_sets(size = 1000)
    train_target, test_target = helper.ohe(train_target, test_target)
    
    activation_function_1 = DICT[args.acti_fct_1]
    activation_function_2 = DICT[args.acti_fct_2]
    activation_function_3 = DICT[args.acti_fct_3]
    m_model = modules.Sequential(modules.Linear(2,args.units), 
                                 activation_function_1(),
                                 modules.Linear(args.units,args.units),
                                 activation_function_2(),
                                 modules.Linear(args.units,args.units), 
                                 activation_function_3(),
                                 modules.Linear(args.units,2))
    
    train_l, test_l = helper.train_model(m_model, train_set, train_target,test_set, test_target, lr = args.lr, num_epoch = args.n_iter)
    
    #plt.plot(range(NUM_EPOCHS), train_l, 'r', range(NUM_EPOCHS), test_l, 'b')
    
    nb_errors = helper.nb_classification_errors(m_model, test_set, test_target)
    helper.print_error(args.acti_fct_1+'_'+args.acti_fct_2+'_'+args.acti_fct_3, nb_errors, 1000)


if __name__ == '__main__':
    main(arguments.get_args())