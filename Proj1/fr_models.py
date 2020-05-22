#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F


'''
THE STATE OF THE ART : MODELS FOR
'''

# Constants
KERNEL_SIZE = 5
PADDING = KERNEL_SIZE//2
POOLING_KERNEL_SIZE = 2
IN_CHANNELS = 1
OUT_CHANNELS_1 = 12
OUT_CHANNELS_2 = 32
NB_FEATURES = 336

# Neural networks from the classes

OUT_CHANNELS_NET = 32
NB_HIDDEN = 200

'''
based on the 4th week practical
'''
class Net(torch.nn.Module):
    def __init__(self, nb_hidden):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(IN_CHANNELS, OUT_CHANNELS_NET,
                                     kernel_size=KERNEL_SIZE, padding=PADDING)
        self.conv2 = torch.nn.Conv2d(OUT_CHANNELS_2, 2*OUT_CHANNELS_2,
                                     kernel_size=KERNEL_SIZE, padding=0)
        self.fc1 = torch.nn.Linear(2*OUT_CHANNELS_2*5*5, nb_hidden)
        self.fc2 = torch.nn.Linear(nb_hidden, nb_hidden//2)
        self.fc3 = torch.nn.Linear(nb_hidden//2, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 1x14x14 -> 32x14x14
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2)) # 32x14x14 -> 64x10x10 -> 64x5x5
        x = x.view(-1, 2*OUT_CHANNELS_2*5*5)
        x = F.relu(self.fc1(x)) # 64x5x5 -> nb_hidden
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
based on the 4th week practical
'''
class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # First convolution.
        self.conv1 = torch.nn.Conv2d(IN_CHANNELS, OUT_CHANNELS_NET,
                                     kernel_size=KERNEL_SIZE, padding=PADDING)
        # Second convolution.
        self.conv2 = torch.nn.Conv2d(OUT_CHANNELS_NET, OUT_CHANNELS_NET,
                                     kernel_size=KERNEL_SIZE, padding=0)
        # Third convolution.
        self.conv3 = torch.nn.Conv2d(OUT_CHANNELS_NET, 2*OUT_CHANNELS_NET,
                                     kernel_size=3, padding=3//2)
        self.fc1 = torch.nn.Linear(2*OUT_CHANNELS_NET*5*5, NB_HIDDEN)
        self.fc2 = torch.nn.Linear(NB_HIDDEN, NB_HIDDEN//2)
        self.fc3 = torch.nn.Linear(NB_HIDDEN//2, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 1x14x14 -> 32x14x14
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2)) # 32x14x14 -> 32x10x10 -> 32x5x5
        x = F.relu(self.conv3(x)) # 32x5x5 -> 64x5x5
        x = x.view(-1, 2*OUT_CHANNELS_NET*5*5)
        x = F.relu(self.fc1(x)) # 64x5x5 -> 200
        x = F.relu(self.fc2(x)) # 200 -> 100
        x = self.fc3(x) # 100 -> 10
        return x

##################################################################################################################
##################################################################################################################
##################################################################################################################

# LeNet5 neural network for number recognition
'''
based on Yann LeCun's LeNet5 
'''
class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # First convolution (In LeNet-5, 32x32 images are given as input. Here the images are 14x14.)
        # Padding of 2 for kernel size of 5 to avoid any crop. Output size: mini_batch_size x 1 x 14 x 14.
        self.conv1 = torch.nn.Conv2d(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS_1,
                                     kernel_size=KERNEL_SIZE, padding=PADDING, bias=True, padding_mode='zeros')
        # Second convolution. Output size: mini_batch_size x 1 x 10 x 10.
        self.conv2 = torch.nn.Conv2d(in_channels=OUT_CHANNELS_1, out_channels=OUT_CHANNELS_2,
                                     kernel_size=KERNEL_SIZE, padding=0, bias=True, padding_mode='zeros')
        # Max-pooling. Output size: mini_batch_size x 1 x 5 x 5.
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=POOLING_KERNEL_SIZE)
        # Fully connected layers.
        self.fc1 = torch.nn.Linear(OUT_CHANNELS_2*5*5, NB_FEATURES) # convert matrix with 32*5*5 (= 800) features to a matrix of 336 features (columns)
        self.fc2 = torch.nn.Linear(NB_FEATURES, NB_FEATURES//2)
        self.fc3 = torch.nn.Linear(NB_FEATURES//2, NB_FEATURES//4)
        self.fc4 = torch.nn.Linear(NB_FEATURES//4, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # convolve, then perform ReLU non-linearity
        x = F.relu(self.conv2(x)) # convolve, then perform ReLU non-linearity
        x = self.max_pool_2(x) # max-pooling with 2x2 grid
        x = x.view(-1, OUT_CHANNELS_2*5*5) # first flatten 'max_pool_2_out' to contain 32*5*5 columns
        x = F.relu(self.fc1(x)) # FC-1, then perform ReLU non-linearity
        x = F.relu(self.fc2(x)) # FC-2, then perform ReLU non-linearity
        x = F.relu(self.fc3(x)) # FC-2, then perform ReLU non-linearity
        x = self.fc4(x) # FC-4

        return x

def get_lenet5():
    return LeNet5()

def get_net2():
    return Net2()

def get_net():
    return Net(300)
