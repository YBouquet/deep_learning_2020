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

PATH = 'temp.pth'

def decrease_learning_rate(lr, optimizer, e, num_epoch):
    lr = lr * (0.8 ** (e / num_epoch)) # 0.8 best ratio for now
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def build_kfold(train_input, k_fold):
    nrows = train_input.size(0)
    fold_size = int(nrows / k_fold)
    if fold_size * k_fold != nrows:
        raise ValueError(
            'ERROR: k_fold value as to be a divisor of the number of rows in the training set')
    indices = torch.Tensor(range(nrows))
    result = [indices[k * fold_size : (k + 1) * fold_size] for k in range(k_fold)]
    return torch.stack(result)

SGD = 'sgd'
ADAM = 'adam'
def pretrain_train_model(model, train_input, train_target, train_figures_target, optimizer_algo, k_fold, mini_batch_size, num_epoch_train, lr_train = 1e-3, beta = 0.9, weight_decay_train = 0, num_epoch_pretrain = 0, lr_pretrain = 1e-3, weight_decay_pretrain = 0, weight_auxiliary_loss = 1., shuffle = False):
    criterion =nn.CrossEntropyLoss()
    auxiliary_criterion = nn.CrossEntropyLoss()

    logs = {PRETRAINING : {TRAINING_PHASE: [], VALIDATION_PHASE: []}, TRAINING: {TRAINING_PHASE: [], VALIDATION_PHASE: []}}

    torch.save(model.state_dict(), PATH)

    global_train_loss = []
    global_valid_loss = []

    indices = build_kfold(train_input, k_fold)

    for k in range(k_fold):
        model.load_state_dict(torch.load(PATH))

        va_indices = indices[k] # 1000/k_fold indices for validation
        tr_indices = indices[~(torch.arange(indices.size(0)) == k)].view(-1) # (k_fold-1) * 1000 / k_fold indices (the rest)

        if k_fold == 1:
            va_indices, tr_indices = tr_indices, va_indices

        train_dataset = TensorDataset(train_input[tr_indices], train_target[tr_indices], train_figures_target[tr_indices])
        validation_dataset = TensorDataset(train_input[va_indices],  train_target[va_indices], train_figures_target[va_indices])

        dataloaders = {
            TRAINING_PHASE : DataLoader(train_dataset, batch_size = mini_batch_size, shuffle = shuffle),
            VALIDATION_PHASE : DataLoader(validation_dataset, batch_size = mini_batch_size, shuffle = shuffle)
        }

        epochs = {PRETRAINING : num_epoch_pretrain, TRAINING : num_epoch_train}
        lrs = {PRETRAINING: lr_pretrain, TRAINING: lr_train}
        aux_weights = {PRETRAINING: 1., TRAINING: weight_auxiliary_loss}
        decays = {PRETRAINING: weight_decay_pretrain, TRAINING : weight_decay_train}

        for step in [PRETRAINING, TRAINING]:
            if optimizer_algo == ADAM:
                optimizer = optim.Adam(model.parameters(), betas = (beta, 0.999), lr=lrs[step], weight_decay = decays[step])
            else :
                optimizer = optim.SGD(model.parameters(), lr = lrs[step], weight_decay = decays[step])
            for e in range(epochs[step]):
                avg_loss = {TRAINING_PHASE: [], VALIDATION_PHASE: []}

                # size([k_fold, 1000/k_fold])

                for phase in [TRAINING_PHASE, VALIDATION_PHASE]:
                    if phase == TRAINING_PHASE:
                        model.train()
                    else:
                        model.eval()

                    running_loss = []

                    for inputs, targets, figures in dataloaders[phase]:
                        outputs = model(inputs)
                        if not(isinstance(outputs, tuple)):
                            loss = criterion(outputs, targets.type(torch.LongTensor))
                        else:
                            tuples = outputs
                            loss = criterion(tuples[-1], targets.type(torch.LongTensor))
                            if step == PRETRAINING:
                                loss = 0
                            for i in range(len(tuples) - 1):
                                loss += aux_weights[step] * auxiliary_criterion(tuples[i], figures[:, i].type(torch.LongTensor))
                        if phase == TRAINING_PHASE:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        running_loss.append(loss)

                    if running_loss:
                        avg_loss[phase].append(torch.tensor(running_loss).mean())
                        logs[step][phase].append(torch.tensor(avg_loss[phase]).mean())

                temp_loss = torch.tensor(logs[step][TRAINING_PHASE])
                if e % 10 == 0:
                    format = 'Epoch %3d / %3d \n\t %s \t\t\t min: %8.5f, max: %8.5f, cur: %8.5f\n'
                    results = (e+1, epochs[step], step, temp_loss.min(), temp_loss.max(), temp_loss[-1])

                    if k_fold > 1:
                        temp_val_loss = torch.tensor(logs[step][VALIDATION_PHASE])
                        format += '\t Validation \t\t\t min: %8.5f, max: %8.5f, cur: %8.5f\n'
                        results += (temp_val_loss.min(), temp_val_loss.max(), temp_val_loss[-1])

                    print(format % results)


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


def grid_search(model, filename, train_input, train_target, train_figures_target, optimizer_name, k_fold, mini_batch_size, n_epochs, lrt_array = [], beta_array = [],  wdt_array = [], num_epoch_pretrain = 50, lrp_array = [], wdp_array = [], wal_array = [], seed = 0):
    torch.save(model.state_dict(), filename)
    '''
    size = [len(lrt_array),len(wdt_array),len(nep_array),len(lrp_array),len(wdp_array),len(wal_array),2]

    results = torch.empty(tuple(size)).fill_(float('inf'))
    '''
    train_final_results = {}
    pretrain_final_results = {}
    mins = []
    values = []

    '''
    print("PRETRAINING OPTIMIZATION STARTED")
    min_ = float('inf')
    value = 0.
    try :
        for learning_rate_pretrain in lrp_array:

            model.load_state_dict(torch.load(filename))
            torch.manual_seed(seed)
            temp = pretrain_train_model(model, train_input, train_target, train_figures_target, k_fold, mini_batch_size, 0, num_epoch_pretrain = num_epoch_pretrain, lr_pretrain = learning_rate_pretrain)
            result = temp[PRETRAINING][TRAINING_PHASE][-1]
            if k_fold > 1:
                result = temp[PRETRAINING][VALIDATION_PHASE][-1]

            if result < min_:
                min_ = result
                value = learning_rate_pretrain

        best_learning_rate_pretrain = value
        values.append(value)
        mins.append(min_)

        min_ = float('inf')
        value = 0.
        for weight_decay_pretrain in wdp_array:

            model.load_state_dict(torch.load(filename))
            torch.manual_seed(seed)
            temp = pretrain_train_model(model, train_input, train_target, train_figures_target, k_fold, mini_batch_size, 0, num_epoch_pretrain = num_epoch_pretrain, lr_pretrain = best_learning_rate_pretrain, weight_decay_pretrain = weight_decay_pretrain)

            result = temp[PRETRAINING][TRAINING_PHASE][-1]
            if k_fold > 1:
                result = temp[PRETRAINING][VALIDATION_PHASE][-1]

            if result < min_:
                min_ = result
                value = weight_decay_pretrain
        best_weight_decay_pretrain = value
        values.append(value)
        mins.append(min_)

        print('\nEND OF PRETRAINING OPTIMIZATION\n')
        model.load_state_dict(torch.load(filename))
        torch.manual_seed(seed)
        pretrain_final_results = pretrain_train_model(model, train_input, train_target, train_figures_target, k_fold, mini_batch_size, 0, num_epoch_pretrain = num_epoch_pretrain, lr_pretrain = best_learning_rate_pretrain, weight_decay_pretrain = best_weight_decay_pretrain)
        torch.save(model.state_dict(), filename)

        print('\nPRETRAINED MODEL SUCCESSFULLY REGISTERED\n')
        '''
    try:
        print('\nTRAINING OPTIMIZATION STARTED\n')
        min_ = float('inf')
        value = 0.
        for learning_rate_train in lrt_array:
            model.load_state_dict(torch.load(filename))
            torch.manual_seed(seed)
            temp = pretrain_train_model(model, train_input, train_target, train_figures_target, optimizer_name, k_fold, mini_batch_size, n_epochs, lr_train = learning_rate_train)
            result = temp[TRAINING][TRAINING_PHASE][-1]
            if k_fold > 1:
                result = temp[TRAINING][VALIDATION_PHASE][-1]

            if result < min_:
                min_ = result
                value = learning_rate_train
        best_learning_rate_train = value
        values.append(value)
        mins.append(min_)

        if optimizer_name == 'adam':
            min_ = float('inf')
            value = 0.
            for beta in beta_array:
                model.load_state_dict(torch.load(filename))
                torch.manual_seed(seed)
                temp = pretrain_train_model(model, train_input, train_target, train_figures_target, optimizer_name, k_fold, mini_batch_size, n_epochs, lr_train = best_learning_rate_train, beta = beta )
                result = temp[TRAINING][TRAINING_PHASE][-1]
                if k_fold > 1:
                    result = temp[TRAINING][VALIDATION_PHASE][-1]

                if result < min_:
                    min_ = result
                    value = weight_auxiliary_loss
            best_beta = beta
            values.append(value)
            mins.append(min_)

        min_ = float('inf')
        value = 0.
        for weight_auxiliary_loss in wal_array:
            model.load_state_dict(torch.load(filename))
            torch.manual_seed(seed)
            temp = pretrain_train_model(model, train_input, train_target, train_figures_target, optimizer_name, k_fold, mini_batch_size, n_epochs, lr_train = best_learning_rate_train, weight_auxiliary_loss = weight_auxiliary_loss )
            result = temp[TRAINING][TRAINING_PHASE][-1]
            if k_fold > 1:
                result = temp[TRAINING][VALIDATION_PHASE][-1]

            if result < min_:
                min_ = result
                value = weight_auxiliary_loss
        best_weight_auxiliary_loss = weight_auxiliary_loss
        values.append(value)
        mins.append(min_)

        min_ = float('inf')
        value = 0.
        for weight_decay_train in wdt_array:
            model.load_state_dict(torch.load(filename))
            torch.manual_seed(seed)
            temp = pretrain_train_model(model, train_input, train_target, train_figures_target, optimizer_name, k_fold, mini_batch_size, n_epochs, lr_train = best_learning_rate_train, weight_decay_train = weight_decay_train, weight_auxiliary_loss = best_weight_auxiliary_loss)
            result = temp[TRAINING][TRAINING_PHASE][-1]
            if k_fold > 1:
                result = temp[TRAINING][VALIDATION_PHASE][-1]

            if result < min_:
                min_ = result
                value = weight_decay_train
        best_weight_decay_train = weight_decay_train
        values.append(value)
        mins.append(min_)

        model.load_state_dict(torch.load(filename))
        torch.manual_seed(seed)
        train_final_results = pretrain_train_model(model, train_input, train_target, train_figures_target, optimizer_name, k_fold, mini_batch_size,n_epochs, lr_train = best_learning_rate_train, weight_decay_train = best_weight_decay_train, weight_auxiliary_loss = best_weight_auxiliary_loss)
        torch.save(model.state_dict(), filename)
        print('\nEND OF TRAINING OPTIMIZATION\n')
    except KeyboardInterrupt:
        return train_final_results, pretrain_final_results, mins, values
    return train_final_results, pretrain_final_results, mins, values
