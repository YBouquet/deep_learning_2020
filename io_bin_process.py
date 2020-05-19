"""
@author: Yann BOUQUET
"""

import torch


def targets_reshape(train_targets, test_targets, one_hot_encoding = True):
    if one_hot_encoding :
        new_train = torch.zeros((train_targets.size()[0],2))
        new_test = torch.zeros((train_targets.size()[0],2))
        for i,b in enumerate(list(train_targets)):
            new_train[i,b] = 1
        for i,b in enumerate(list(test_targets)):
            new_test[i,b] = 1
        return new_train, new_test
    return train_targets.view(-1,1), test_targets.view(-1,1)


def nb_classification_errors(model, test_input, target, mini_batch_size, criterion_type = 'BCE'):
    nb_errors = 0

    for b in range(0, test_input.size(0), mini_batch_size):
        output = model(test_input.narrow(0, b, mini_batch_size))
        if isinstance(output, tuple):
            if len(output) == 3 :
                _, _, output = output
            else:
                raise Exception('ERROR', 'the output of the model isn\'t recognized')
        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if criterion_type == 'CE' or criterion_type == 'MSE':
                if target[b + k] != predicted_classes[k]:
                    nb_errors = nb_errors + 1
            elif criterion_type == 'BCE':
                if target[[b + k],predicted_classes[k]] <= 0:
                    nb_errors = nb_errors + 1

    return nb_errors


def data_augmentation(tr_input, tr_target, tr_figure_target, pairs_nb, nb_augmentation):
    data_augmentation_input = torch.zeros(nb_augmentation*pairs_nb,2,14,14)
    data_augmentation_figure_target = torch.zeros(nb_augmentation*pairs_nb, 2, dtype=torch.long)
    rand_indices = torch.randperm(2*nb_augmentation*pairs_nb)%pairs_nb

    data_augmentation_input[:,0] = tr_input[rand_indices[:nb_augmentation*pairs_nb],0]
    data_augmentation_input[:,1] = tr_input[rand_indices[nb_augmentation*pairs_nb:],1]
    data_augmentation_target = (tr_figure_target[rand_indices[:nb_augmentation*pairs_nb],0] <=
                                tr_figure_target[rand_indices[nb_augmentation*pairs_nb:],1])*1
    data_augmentation_figure_target[:,0] = tr_figure_target[rand_indices[:nb_augmentation*pairs_nb],0]
    data_augmentation_figure_target[:,1] = tr_figure_target[rand_indices[nb_augmentation*pairs_nb:],1]

    data_augmentation_input = torch.cat((tr_input, data_augmentation_input))
    data_augmentation_target = torch.cat((tr_target, data_augmentation_target))
    data_augmentation_figure_target = torch.cat((tr_figure_target, data_augmentation_figure_target))

    return (data_augmentation_input, data_augmentation_target, data_augmentation_figure_target)


def data_doubling(tr_input, tr_target, tr_figure_target):
    pairs_nb = tr_input.size(0)
    data_augmentation_input = torch.zeros(2*pairs_nb,2,14,14)
    data_augmentation_figure_target = torch.zeros(2*pairs_nb, 2, dtype=torch.long)

    data_augmentation_input[:pairs_nb,0] = tr_input[:,0]
    data_augmentation_input[pairs_nb:,0] = tr_input[:,1]
    data_augmentation_input[:pairs_nb,1] = tr_input[:,1]
    data_augmentation_input[pairs_nb:,1] = tr_input[:,0]

    data_augmentation_target = torch.cat((tr_target,1-tr_target))

    data_augmentation_figure_target[:,0] = torch.cat((tr_figure_target[:,0],tr_figure_target[:,1]))
    data_augmentation_figure_target[:,1] = torch.cat((tr_figure_target[:,1],tr_figure_target[:,0]))

    return (data_augmentation_input, data_augmentation_target, data_augmentation_figure_target)
