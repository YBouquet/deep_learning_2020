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
## Project 2

You can find the code for the project 1 in the ./Proj2 folder

### Files

* `modules.py` regroups every module of the framework (Module, ReLU, Tanh, Linear, Sequential, LossMSE)
* `helpers.py` regroups the SGD module, the functions that generate and preprocess the data, and the train function
* `main.py` is the code that creates the model and train it

All arguments proposed for main.py:
* `n_epochs` : default = 5000, the number of epochs for the training process;
* `batch_size`: default = 100, size of a batch during an optimization step of the training process;
* `lr`: default = 1e-2, learning rate of the optimizer;
* `activation`: default = 'relu', activation function used in the model; relu or tanh;
* `units`: default = 25, number of units in the three hidden layers;

### Quick Start
To generate the performance of the two version of architecture (first with relu and second with tanh):

`python test.py`

To generate the graph of the data distribution:

`python plot_data.py`

To generate the graph of the data distribution with the results of the models:

`python plot_results.py`

To generate the plot of the performance of the architectures through a training and a validation set:

`python plot_validation.py`

plot validation take the same arguments as main that you can tune. Another one can also be entered:
* `ratio`: default = 0.8, ratio for the validation set (1 means no validation, 0.8 means 80% for training, 20% for validation)
