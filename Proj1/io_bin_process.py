"""
@author: Yann BOUQUET
"""

import torch



def targets_reshape(train_targets, test_targets, one_hot_encoding = True):
    """One hot encoding for the targets representing the binary classes

    An option is about return a vector Nx1 of the targets

    Parameters
    ----------
    train_targets : Tensor,
        Targets of the train set
    test_targets : Tensor,
        Targets of the test set
    one_hot_encoding : bool, optional
        Trigger the one hot encoding of the targets
    """
    if one_hot_encoding :
        new_train = torch.zeros((train_targets.size()[0],2))
        new_test = torch.zeros((train_targets.size()[0],2))
        for i,b in enumerate(list(train_targets)):
            new_train[i,b] = 1
        for i,b in enumerate(list(test_targets)):
            new_test[i,b] = 1
        return new_train, new_test
    return train_targets.view(-1,1), test_targets.view(-1,1)

def nb_classification_pretrain(model, test_input, target, mini_batch_size):
    """Count the number of errors the model makes during its classification

    Parameters
    ----------
    model : nn.Module
        Instance of the neural network model
    test_input : Tensor
        Test set as input of the model
    target : Tensor
        Targets of the test set
    mini_batch_size: int
        Size of the batch we want to fill in the model at each iteration

    Raises
    ------
    Exception
        if the size of the tuple returned by the tuple isn't recognized

    Return
    ------
    nb_errors: int
        number of errors made during the classification
    """
    nb_errors = 0

    for b in range(0, test_input.size(0), mini_batch_size):
        output_1, output_2, _ = model(test_input.narrow(0, b, mini_batch_size))
        _, predicted_classes_1 = output_1.max(1)
        _, predicted_classes_2 = output_2.max(1)
        for k in range(mini_batch_size):
            if target[b + k, 0] != predicted_classes_1[k]:
                nb_errors = nb_errors + 1
            if target[b + k, 1] != predicted_classes_2[k]:
                nb_errors = nb_errors + 1

    return nb_errors
def nb_classification_errors(model, test_input, target, mini_batch_size):
    """Count the number of errors the model makes during its classification

    Parameters
    ----------
    model : nn.Module
        Instance of the neural network model
    test_input : Tensor
        Test set as input of the model
    target : Tensor
        Targets of the test set
    mini_batch_size: int
        Size of the batch we want to fill in the model at each iteration

    Raises
    ------
    Exception
        if the size of the tuple returned by the tuple isn't recognized

    Return
    ------
    nb_errors: int
        number of errors made during the classification
    """
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
            if target[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1

    return nb_errors


def data_augmentation(tr_input, tr_target, tr_figure_target, pairs_nb, nb_augmentation):
    """This function augment the data by cutting the pairs of images and reassemble randomly
    new pairs to increase the size of the training set.

    Parameters
    ----------
    tr_input : Tensor
        Targets of the train set
    tr_target : Tensor
        Targets of the test set
    tr_figure_target : Tensor
        trigger the one hot encoding of the targets
    pairs_nb : int
        Number of pairs that we initially have
    nb_augmentation : int
        Increasing ratio

    Return
    ----------
    data_augmentation_input, data_augmentation_target, data_augmentation_figure_target : tuple<Tensor>
        augmented data with pairs_nb*(1 + nb_augmentation) new pairs
    """
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
    """Return the initial pairs with their mirrors
    example : from [(1,2)] we get [](1,2),(2,1)]

    Parameters
    ----------
    tr_input : Tensor
        Train set
    tr_target : Tensor
        Targets of the train set (class from 0 to 1)
    tr_figure_target : Tensor
        Figures of the train set (class from 0 to 9)
    """
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
