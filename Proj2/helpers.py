from torch import empty
import torch
import statistics
import math

from modules import LossMSE

class SGD():
    def __init__(self, params, lr = 1e-2):
        self.lr = lr
        self.params = params
    def step(self):
        for p, grad in self.params:
            p.add_(-self.lr * grad)
    def zero_grad(self):
        for p, grad in self.params:
            grad.zero_()

def generate_sets(size = 1000):
    """Generating a set of (x,y) coordinates in [0,1]^2 and the associated classes :
    0 if outside the disk of radius 1/sqrt(2pi) centered on (0.5,0.5)
    1 if inside

    We consider that the disk is closed

    Parameters
    ----------
    size : int, optional
        number of points we want to generate

    Return
    ----------
    train_set, train_target.int(), test_set, test_target.int() : Tensor, Tensor, Tensor, Tensor
        'size' samples for the training set
        'size' targets for the training set
        'size' samples for the test set
        'size' targets for the test set
    """
    center = 0.5
    radius = 1 / math.sqrt(2*math.pi)
    train_set  = empty(size, 2).uniform_(0, 1)
    test_set = empty(size, 2).uniform_(0, 1)
    train_target = ((center - train_set).norm(dim = 1) <= radius) * 1
    test_target = ((center - test_set).norm(dim = 1) <= radius) * 1
    return train_set, train_target.int(), test_set, test_target.int()

def nb_classification_errors(model, test_input, test_target, batch):
    """Count the number of errors the model makes during its classification

    Parameters
    ----------
    model : nn.Module
        Instance of the neural network model
    test_input : Tensor
        Test set as input of the model
    target : Tensor
        Targets of the test set
    batch: int
        Size of the batch we want to fill in the model at each iteration

    Return
    ------
    nb_errors: int
        number of errors made during the classification
    """
    nb_errors = 0
    for b in range(0, test_input.size(0), batch):
        output = model.forward(test_input.narrow(0,b, batch))
        _, predicted_class = output.max(1)
        for i in range(batch):
            if test_target[b + i, predicted_class[i]] <= 0 :
                nb_errors = nb_errors + 1
    return nb_errors

def ohe(train_targets, test_targets):
    """One hot encoding for the targets representing the binary classes

    Parameters
    ----------
    train_targets : Tensor,
        Targets of the train set
    test_targets : Tensor,
        Targets of the test set

    Return
    ----------
    new_train, new_test : Tensor, Tensor
        one hot encoded targets
    """
    new_train = empty((train_targets.size()[0],2)).zero_()
    new_test = empty((train_targets.size()[0],2)).zero_()
    for i,b in enumerate(train_targets.tolist()):
        new_train[i,b] = 1
    for i,b in enumerate(test_targets.tolist()):
        new_test[i,b] = 1
    return new_train, new_test


def normalize(train_input, test_input):
    """Data normalization

    Parameters
    ----------
    train_input : Tensor,
        Samples of the train set
    test_input : Tensor,
        Samples of the test set
    Return
    ----------
    Tensor, Tensor
        normalized samples
    """
    return((train_input - train_input.mean(0))/train_input.std(0), (test_input - train_input.mean(0))/train_input.std(0))


def train_model(model, train_input, train_target, test_input, test_target, lr = 1e-2, num_epoch = 25, batch = 100):
    """
    Parameters
    ----------
    model : nn.Module
        Instance of the neural network model
    train_input : Tensor
        Train set with samples
    train_target : Tensor
        Train targets for comparison
    test_input : Tensor
        Train targets for figure recognition
    test_target : Tensor
        Train targets for figure recognition
    lr : Class
        Torch.nn class of the criterion
    num_epoch : str
        Name of the optimizer ('sgd' or 'adam')
    batch : int
        Number of folds for cross_validation (1 means no cross validation)

    Return
    ----------
    losses_train, losses_test : list<float>, list<float>
        mean losses on the training set at each epoch
        mean losses on the test set at each epoch
    """
    criterion = LossMSE()
    optimizer = SGD(model.param(), lr=lr)
    losses_train = []
    losses_test = []
    n = train_input.size(0)
    for e in range(num_epoch):
        running_loss = []
        test_loss = []

        for b in range(0, train_input.size(0), batch):
            output = model.forward(train_input.narrow(0, b, batch))
            loss = criterion.forward(output, train_target.narrow(0, b, batch))
            optimizer.zero_grad()
            model.backward(criterion.backward())
            optimizer.step()
            running_loss.append(loss)
        losses_train.append(torch.Tensor(running_loss).mean().tolist())

        for b in range(0, test_input.size(0), batch):
            test_loss.append(criterion.forward(model.forward(test_input.narrow(0, b, batch)), test_target.narrow(0, b, batch)))
        losses_test.append(torch.Tensor(test_loss).mean().tolist())

    return losses_train, losses_test
