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

def error_rate_classification(model, test_input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, test_input.size(0), mini_batch_size):
        output = model(test_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if target[b + k, predicted_classes[k]] <= 0:
                nb_errors = nb_errors + 1

    return nb_errors
