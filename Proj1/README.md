# Deep Learning Mini-Projects
-------
## Requirements
The code was run in a VM built on a Linux Debian 9.3

* Python version 3.7.2
* Pytorch version 1.4.0
* Matplotlib version 3.1.1 (for plots)
* Numpy version 1.17.4 (for plots)

To install Pytorch with conda, run:

`conda install pytorch==1.4.0 -c pytorch`

------------------------
## Project 1

You can find the code for the project 1 in the ./Proj1 folder

### Files

#### Models
The models detailed in the development of the report are located in [models.py]{}
* `Two_nets` class implements the baseline architecture of our work and returns also the necessary outputs for auxiliary loss computation (2nets);
* `Two_nets_ws` adds weight sharing to the baseline (2nets_ws);
* `Two_nets_ws_do` adds dropout to 2nets_ws (2nets_ws_do);
* `Two_nets_ws_bn` adds batch normalization to 2_nets_do;
* `Required` is the convnet asked in the description of the project (~70000 parameters, 25 epochs in ~2.5 sec, ~15% of errors);

[fr_models.py]{./Proj1/models.py} regroups all models that do figure recognition:
* `LeNet5` is inspired of Yann LeCun's work;
* `Net` and `Net2` are copied from the 4th practical session.

[appendix.py]{./Proj1/models.py} regroups all models created during researches for our project

#### Training
* `train.py` contains the training and the grid search (for hyperparameters optimization) algorithms;
* `main.py` run all 10 rounds of training process for a selected model to compute the mean and the standard deviation of its performance according to the randomization of the training set, the test set and its parameters. We ensure a different seed fixed at each round to ensure reproducibility of our results;

All arguments proposed for main.py:
* `model`: default = 2nets;

'net': Net
'net2': Net2
'lenet5': LeNet5
'required': Required
'2nets': Two_Nets
'2nets_ws': Two_nets_ws
'2nets_ws_do': Two_nets_ws_do
'2nets_ws_bn': Two_nets_ws_bn

* `n_epochs` : default = 50, the number of epochs for the training process;
* `batch_size`: default = 5, size of a batch during an optimization step of the training process;
* `learning_rate`: default = 1e-3, learning rate of the optimizer;
* `optimizer`: default = sgd, optimizer for the training process; sgd for Stochastic Gradient Descent, adam for Adam algorithm;
* `adam_beta1`: default = 0.9, first order decaying parameter for Adam Algorithm;
* `weight_decay`: default = 0., L2 regularization parameter;
* `weight_auxiliary_loss`: default = 1., weight applied to every auxiliary loss;
* `shuffling`: default = False, True for shuffling the training set at every epoch.

#### Quick Start
To generate the training rounds and the final performance of the `Required` architecture:

`python required.py`

To generate the training rounds and the final performance of all architecture given in the development (without appendix) of our report:

`python performances.py`

To run both `required.py` and `performances.py`, run :

`python test.py`

To generate the graph analyzing the performance of two models through cross validation:

`python graphs.py`
--------------------
