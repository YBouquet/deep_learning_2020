"""
@author: Yann BOUQUET
"""

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from io_num_process import one_hot_encoding


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAINING_PHASE = 'Train'
VALIDATION_PHASE = 'Validation'

PRETRAINING = 'Pretraining'
TRAINING = 'Training'

PATH = 'temp.pth' #for the k_fold process we reinitialize the parameters of the model to retrain it with different validation sets


def build_kfold(train_input, k_fold):
    nrows = train_input.size(0)
    fold_size = int(nrows / k_fold)
    if fold_size * k_fold != nrows:
        raise ValueError(
            'ERROR: k_fold value as to be a divisor of the number of rows in the training set')
    indices = torch.arange(nrows)
    result = [indices[k * fold_size : (k + 1) * fold_size] for k in range(k_fold)]
    return torch.stack(result)


def decrease_learning_rate(lr, optimizer, e, num_epoch):
    """decreasing the learning rate over the epochs
    Use this function to see some enhancement for the Vanilla SGD
    This function shouldn't be called for Adam Algorithm

    Parameters
    ----------
    lr : float
        Current learning rate
    optimizer : torch.optim instance
        Optimizer whose learning rate we want to decrease
    e : int
        Current epoch
    num_epoch: int,
        Total number of epochs during the training process
    """
    lr = lr * (0.8 ** (e / num_epoch)) # 0.8 best ratio for now
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

SGD = 'sgd'
ADAM = 'adam'

'''
This train process was implement to handle training process, validation with k_fold crossvalidation,
auxiliary losses, and pretraining
'''
# The result on the pretraining wasn't sufficient, so we decided to let the code be as it is
# However we don't provide any test for it
def pretrain_train_model(model, train_input, train_target, train_figures_target, pr_input, pr_figures_target, criterion_class, optimizer_algo, k_fold, mini_batch_size, num_epoch_train, lr_train = 1e-3, beta = 0.9, weight_decay_train = 0, num_epoch_pretrain = 0, lr_pretrain = 1e-3, beta_pretrain = 0.9, weight_decay_pretrain = 0, weight_auxiliary_loss = 1., shuffle = False, decrease_lr = False):
    """
    Parameters
    ----------
    model : nn.Module
        Instance of the neural network model
    train_input : str
        Train set with samples
    train_target : int, optional
        Train targets for comparison
    train_figures_target : str
        Train targets for figure recognition
    criterion_class : Class
        Torch.nn class of the criterion
    optimizer_algo : str
        Name of the optimizer ('sgd' or 'adam')
    k_fold : int
        Number of folds for cross_validation (1 means no cross validation)
    mini_batch_size : int
        Size of a batch (input of the model)
    num_epoch_train : int
        Number of epochs for the training process
    lr_train : float, optional
        Learning rate of the optimizer for the trainig process
    beta : float, optional
        first decaying order parameter for Adam algorithm
    weight_decay_train : float, optional
        L2 regularization parameter for the training process
    num_epoch_pretrain : int, optional
        Number of epochs for the pretraining process (0 means no pretraining)
    lr_pretrain : float, optional
        Learning rate of the optimizer for the pretrainig process
    weight_decay_pretrain : float, optional
        L2 regularization for the pretraining process
    weight_auxiliary_loss : float, optional
        Weight given to the auxiliary losses for the sum of the losses
    shuffle : bool, optional
        True implies the shuffling of the data at each
    decrease_lr : bool, optional
        Choose if we decrease the learning rate with time (according to the associated function)

    Return
    ----------
    logs : dict
        Dictionary regrouping the losses at each steps of the pretraining, training, validation processes
    """
    criterion = criterion_class()
    auxiliary_criterion = nn.CrossEntropyLoss()

    m_type = torch.FloatTensor
    if isinstance(criterion, nn.CrossEntropyLoss):
        m_type = torch.LongTensor

    logs = {PRETRAINING : {TRAINING_PHASE: [], VALIDATION_PHASE: []}, TRAINING: {TRAINING_PHASE: [], VALIDATION_PHASE: []}}

    torch.save(model.state_dict(), PATH) #save the initial model parameter in order to reinitialize the model for each k_fold combination

    indices = build_kfold(train_input, k_fold) # seperate the training set into k_fold parts for cross_validation

    pr_indices = build_kfold(pr_input, k_fold)
    #at each iteration we will consider different trainset and validation set, we will at each epoch
    #train the network on the selected trainset and verify its performance on the selected validation set
    #the results would be stored the get the mean of this epoch through all variants of train sets and validation sets
    for k in range(k_fold): #k_fold = 5
        # at each k_fold combination we retrain the model with its initial parameters in order to reinitialize it for the new train and validation sets
        model.load_state_dict(torch.load(PATH))

        va_indices = indices[k] # 1000/k_fold indices for validation
        tr_indices = indices[~(torch.arange(indices.size(0)) == k)].view(-1) # (k_fold-1) * 1000 / k_fold indices (the rest)

        pr_va_indices = pr_indices[k]
        pr_tr_indices = pr_indices[~(torch.arange(pr_indices.size(0)) == k)].view(-1)

        if k_fold == 1: #if there is no k_fold cross validation
            va_indices, tr_indices = tr_indices, va_indices
            pr_va_indices, pr_tr_indices = pr_tr_indices, pr_va_indices

        train_dataset = TensorDataset(train_input[tr_indices], train_target[tr_indices], train_figures_target[tr_indices])
        validation_dataset = TensorDataset(train_input[va_indices],  train_target[va_indices], train_figures_target[va_indices])

        pr_train_dataset = TensorDataset(pr_input[pr_tr_indices], train_target[pr_tr_indices], pr_figures_target[pr_tr_indices])
        pr_validation_dataset = TensorDataset(pr_input[pr_va_indices],  train_target[pr_va_indices], pr_figures_target[pr_va_indices])

        dataloaders = {
            TRAINING_PHASE : DataLoader(train_dataset, batch_size = mini_batch_size, shuffle = shuffle),
            VALIDATION_PHASE : DataLoader(validation_dataset, batch_size = mini_batch_size, shuffle = shuffle and k_fold > 1) #if the validation set is empty, Dataloader can't shuffle
        }

        pr_dataloaders = {
            TRAINING_PHASE : DataLoader(pr_train_dataset, batch_size = mini_batch_size, shuffle = shuffle),
            VALIDATION_PHASE : DataLoader(pr_validation_dataset, batch_size = mini_batch_size, shuffle = shuffle and k_fold > 1) #if the validation set is empty, Dataloader can't shuffle
        }


        #setting the different parameters for training and pretraining
        sets = {PRETRAINING : pr_dataloaders, TRAINING : dataloaders}
        epochs = {PRETRAINING : num_epoch_pretrain, TRAINING : num_epoch_train}
        lrs = {PRETRAINING: lr_pretrain, TRAINING: lr_train}
        betas = {PRETRAINING : beta_pretrain , TRAINING : beta}
        aux_weights = {PRETRAINING: 1., TRAINING: weight_auxiliary_loss}
        decays = {PRETRAINING: weight_decay_pretrain, TRAINING : weight_decay_train}

        #starting with pretraining the model and starting the training afterward
        #the iteration over pretraining is empty if num_epoch_pretrain = 0
        for step in [PRETRAINING, TRAINING]:
            if optimizer_algo == ADAM:
                optimizer = optim.Adam(model.parameters(), betas = (beta, 0.999), lr=lrs[step], weight_decay = decays[step])
            else :
                optimizer = optim.SGD(model.parameters(), lr = lrs[step], weight_decay = decays[step])

            for e in range(epochs[step]): #no iteration if epochs[step] = 0
                avg_loss = {TRAINING_PHASE: [], VALIDATION_PHASE: []}

                if decrease_lr: #we use a simple version of an adaptative algorithm by reducing the learning rate for the sgd, it can help to reach great performance
                    decrease_learning_rate(lr, optimizer, e, num_epoch)

                #we then train the model and test it on the validation set
                #if the validation set is empty, the for loop on the validation DataLoader will directly break
                for phase in [TRAINING_PHASE, VALIDATION_PHASE]:
                    if phase == TRAINING_PHASE: #we set the model in train mode during training phase
                        model.train()
                    else:
                        model.eval() #we set eval mode otherwise

                    running_loss = []

                    for inputs, targets, figures in sets[step][phase]: #training or testing the model over every bacth
                        outputs = model(inputs)
                        if not(isinstance(outputs, tuple)): #check if the outputs return many tensors (for auxiliary loss) or a single tensor
                            loss = criterion(outputs, targets.type(m_type))
                        else:
                            tuples = outputs # num_1, num_2, comp
                            loss = 0
                            if step == TRAINING: #if we are on pretraining we want to optimize the model only over the first part of the model so we exclude the loss previously computed
                                loss = criterion(tuples[-1], targets.type(m_type)) #compute the loss of the results for binary classification
                            for i in range(len(tuples) - 1): #we add all auxiliary loss to the initial loss (can be zero if in pretraining)
                                loss += aux_weights[step] * auxiliary_criterion(tuples[i], figures[:, i].type(torch.LongTensor))
                        if phase == TRAINING_PHASE: #we train the model here based on the computed loss
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        running_loss.append(loss)

                    if running_loss:
                        '''
                        avg_loss[phase].append(torch.tensor(running_loss).mean())
                        logs[step][phase].append(torch.tensor(avg_loss[phase]).mean())
                        '''
                        logs[step][phase].append(torch.tensor(running_loss).mean())

                temp_loss = torch.tensor(logs[step][TRAINING_PHASE])
                if e % 10 == 0:
                    format = 'Epoch %3d / %3d \n\t %s \t\t\t min: %8.5f, max: %8.5f, cur: %8.5f\n'
                    results = (e+1, epochs[step], step, temp_loss.min(), temp_loss.max(), temp_loss[-1])

                    if k_fold > 1:
                        temp_val_loss = torch.tensor(logs[step][VALIDATION_PHASE])
                        format += '\t Validation \t\t\t min: %8.5f, max: %8.5f, cur: %8.5f\n'
                        results += (temp_val_loss.min(), temp_val_loss.max(), temp_val_loss[-1])

                    print(format % results)

    #we compute the mean value of each epoch for each validation and train set through the all process
    for k,v in logs.items():
        for k_v, v_v in v.items():
            if v_v:
                logs[k][k_v] = torch.tensor(v_v).view(k_fold,-1).mean(0)

    format = 'Mean Final Loss \n\t %s :\t %8.5f\n'
    results = [TRAINING]
    if len(logs[TRAINING][TRAINING_PHASE]):
        results.append(logs[TRAINING][TRAINING_PHASE][-1])

        if k_fold > 1 :
            format += '\t Validation :\t %8.5f\n'
            results.append(logs[TRAINING][VALIDATION_PHASE][-1])
    else:
        results = [PRETRAINING, logs[PRETRAINING][TRAINING_PHASE][-1]]
        if k_fold > 1 :
            format += '\t Validation :\t %8.5f\n'
            results.append(logs[PRETRAINING ][VALIDATION_PHASE][-1])
    print(format % tuple(results))
    model.eval()


    return logs


def grid_search(model, filename, train_input, train_target, train_figures_target, criterion_class, optimizer_name, k_fold, mini_batch_size, n_epochs, lrt_array = [], beta_array = [],  wdt_array = [],  weight_auxiliary_loss  = 1. , seed = 0):
    """
    Parameters
    ----------
    model : nn.Module
        Instance of the neural network model
    filename : str
        File where we want to store the initial parameters of the model
    train_input : str
        Train set with samples
    train_target : int, optional
        Train targets for comparison
    train_figures_target : str
        Train targets for figure recognition
    criterion_class : Class
        Torch.nn class of the criterion
    optimizer_name : str
        Name of the optimizer ('sgd' or 'adam')
    k_fold : int
        Number of folds for cross_validation (1 means no cross validation)
    mini_batch_size : int
        Size of a batch (input of the model)
    num_epoch_train : int
        Number of epochs for the training process
    lrt_array : list<float>, optional
        List of learning rates we want to test for the trainig process
    beta_array : list<float>, optional
        List of first decaying order parameters to test for the trainig process
    wdt_array : list<float>, optional
        List of L2 regularization parameters o test for the trainig process
    weight_auxiliary_loss : float, optional
        Weight given to the auxiliary losses for the sum of the losses
    seed : int, optional
        Seed used for feeding the randomizator of torch
    Return
    ----------
    train_final_results, mins, values : dict, dict, list<float>, list<float>
        Dictionary regrouping the losses at each steps of the pretraining, training, validation processes for the best combination of parameters
        List of the final losses reached for each best parameter
        List of the values of the best parameters
    """

    torch.save(model.state_dict(), filename)

    train_final_results = {}
    mins = []
    values = []

    try:
        print('\nTRAINING OPTIMIZATION STARTED\n')
        min_ = float('inf')
        value = 0.
        for learning_rate_train in lrt_array:
            model.load_state_dict(torch.load(filename))
            torch.manual_seed(seed)
            temp = pretrain_train_model(model, train_input, train_target, train_figures_target, criterion_class, optimizer_name, k_fold, mini_batch_size, n_epochs, lr_train = learning_rate_train, weight_auxiliary_loss = weight_auxiliary_loss )
            result = temp[TRAINING][TRAINING_PHASE][-1]
            if k_fold > 1:
                result = temp[TRAINING][VALIDATION_PHASE][-1]

            if result < min_:
                min_ = result
                value = learning_rate_train
        best_learning_rate_train = value
        values.append(value)
        mins.append(min_)

        min_ = float('inf')
        value = 0.
        if optimizer_name == 'adam':
            min_ = float('inf')
            value = 0.
            for beta in beta_array:
                model.load_state_dict(torch.load(filename))
                torch.manual_seed(seed)
                temp = pretrain_train_model(model, train_input, train_target, train_figures_target, criterion_class, optimizer_name, k_fold, mini_batch_size, n_epochs, lr_train = best_learning_rate_train, beta = beta , weight_auxiliary_loss = weight_auxiliary_loss )
                result = temp[TRAINING][TRAINING_PHASE][-1]
                if k_fold > 1:
                    result = temp[TRAINING][VALIDATION_PHASE][-1]

                if result < min_:
                    min_ = result
                    value = beta
            best_beta = value
            values.append(value)
            mins.append(min_)

        min_ = float('inf')
        value = 0.
        for weight_decay_train in wdt_array:
            model.load_state_dict(torch.load(filename))
            torch.manual_seed(seed)
            temp = pretrain_train_model(model, train_input, train_target, train_figures_target, criterion_class,optimizer_name, k_fold, mini_batch_size, n_epochs, lr_train = best_learning_rate_train, weight_decay_train = weight_decay_train, weight_auxiliary_loss = weight_auxiliary_loss )
            result = temp[TRAINING][TRAINING_PHASE][-1]
            if k_fold > 1:
                result = temp[TRAINING][VALIDATION_PHASE][-1]

            if result < min_:
                min_ = result
                value = weight_decay_train
        best_weight_decay_train = value
        values.append(value)
        mins.append(min_)

        model.load_state_dict(torch.load(filename))
        torch.manual_seed(seed)
        train_final_results = pretrain_train_model(model, train_input, train_target, train_figures_target, criterion_class, optimizer_name, k_fold, mini_batch_size,n_epochs, lr_train = best_learning_rate_train, weight_decay_train = best_weight_decay_train, weight_auxiliary_loss = weight_auxiliary_loss)
        torch.save(model.state_dict(), filename)
        print('\nEND OF TRAINING OPTIMIZATION\n')
    except KeyboardInterrupt:
        return train_final_results, mins, values
    return train_final_results, mins, values
