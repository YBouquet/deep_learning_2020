#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:05:24 2020

@author: Thomas Fry
"""


import torch
from torch import nn


# Constants
KERNEL_SIZE = 5
PADDING = KERNEL_SIZE//2
POOLING_KERNEL_SIZE = 2


def convolution_block(in_channels, out_channels_1, out_channels_2, kernel_size=KERNEL_SIZE, padding = PADDING):
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels_1, kernel_size=kernel_size,
                    padding = padding, bias=True, padding_mode = 'zeros'),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=kernel_size),
        nn.ReLU())
    return block

def fully_connected_network(in_features, hd_features, out_features):
    block = nn.Sequential(
        nn.Linear(in_features, hd_features),
        nn.ReLU(),
        nn.Linear(hd_features, hd_features//2),
        nn.ReLU(),
        nn.Linear(hd_features//2, hd_features//4),
        nn.ReLU(),
        nn.Linear(hd_features//4, out_features))
    return block


CONV_PARAMETERS = {
    'in_channels' : 1,
    'out_channels_1' : 12,
    'out_channels_2' : 32
}

FCN_PARAMETERS = {
    'in_features_a' : CONV_PARAMETERS['out_channels_2']*3*3,
    'in_features_b' : CONV_PARAMETERS['out_channels_2']*5*5,
    'in_features_c' : CONV_PARAMETERS['out_channels_2']*2*5,
    'hd_features' : 512,
    'out_features_a' : 2,
    'out_features_b' : 10
}


##################################################################################################################
##################################################################################################################
##################################################################################################################

class Two_Channels(nn.Module):

    @staticmethod
    def convolution_block(in_channels,mid_channels, out_channels, vector_size, kernel_size=KERNEL_SIZE):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size = kernel_size,
                        padding = 2, padding_mode = 'zeros'),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size = kernel_size),
            nn.MaxPool2d(kernel_size=POOLING_KERNEL),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_channels, out_channels = vector_size, kernel_size = kernel_size),
            nn.ReLU()
        )
        return block

    @staticmethod
    def fully_connected_network(in_features, hd_features, out_features):
        block = nn.Sequential(
            nn.Linear(in_features, hd_features),
            nn.ReLU(),
            nn.Linear(hd_features, out_features)
        )
        return block

    '''
    def get_hyperparameters():
        return self.hyper_params
    '''
    def __init__(self, cn_parameters, vector_size, fn_parameters):
        super(Two_Channels, self).__init__()
        #self.hyper_params = (CN_PARAMETERS_2C, VECTOR_SIZE, FN_PARAMETERS_2C)
        self.conv_ = self.convolution_block(cn_parameters['in_channels'], cn_parameters['mid_channels'], cn_parameters['out_channels'], vector_size)
        self.fcn_ = self.fully_connected_network(vector_size, fn_parameters['hd_features'], fn_parameters['out_features'])


    def forward(self, x):
        m_vector = self.conv_(x)
        return self.fcn_(m_vector.view(-1,  VECTOR_SIZE))

VECTOR_SIZE = 120

CN_PARAMETERS_2C = {
    'in_channels' : 2,
    'mid_channels' : 12,
    'out_channels' : 32
}

FN_PARAMETERS_2C = {
    'hd_features' : 150,
    'out_features' : 2
}

def get_2channels():
    return Two_Channels(CN_PARAMETERS_2C, VECTOR_SIZE, FN_PARAMETERS_2C)


##################################################################################################################
##################################################################################################################
##################################################################################################################


class Two_One_Channel(nn.Module):
    def __init__(self, conv_parameters, fcn_parameters):
        super(Two_One_Channel, self).__init__()
        self.conv_1 = convolution_block(conv_parameters['in_channels'], conv_parameters['out_channels_1'], conv_parameters['out_channels_2'])
        self.conv_2 = convolution_block(2*conv_parameters['out_channels_2'], conv_parameters['out_channels_1'], conv_parameters['out_channels_2'])
        self.max_pool = nn.MaxPool2d(kernel_size=POOLING_KERNEL_SIZE)
        self.fcn_ = fully_connected_network(fcn_parameters['in_features_a'], fcn_parameters['hd_features'], fcn_parameters['out_features_a'])

    def forward(self, x):
        (x_first, x_second) = (x[:,0].view(-1, 1, 14, 14), x[:,1].view(-1, 1, 14, 14))
        (x_first, x_second) = (self.conv_1(x_first), self.conv_1(x_second)) # 1 x 14 x 14 -> 12 x 14 x 14 -> 32 x 10 x 10
        x = torch.cat((x_first,x_second), 1)
        x = self.conv_2(x) # 64 x 10 x 10 -> 12 x 10 x 10 -> 32 x 6 x 6
        x = self.max_pool(x) # 32 x 6 x 6 -> 32 x 3 x 3
        x = x.view(-1, 32*3*3)
        x = self.fcn_(x)

        return x

def get_2_one_channel():
    return Two_One_Channel(CONV_PARAMETERS, FCN_PARAMETERS)


##################################################################################################################
##################################################################################################################
##################################################################################################################


class One_Image(nn.Module):
    def __init__(self, conv_parameters, fcn_parameters):
        super(One_Image, self).__init__()
        self.conv_ = self.convolution_block(conv_parameters['in_channels'], conv_parameters['out_channels_1'], conv_parameters['out_channels_2'])
        self.fcn_ = fully_connected_network(fcn_parameters['in_features_c'], fcn_parameters['hd_features'], fcn_parameters['out_features_a'])

    @staticmethod
    def convolution_block(in_channels, out_channels_1, out_channels_2, kernel_size=KERNEL_SIZE, padding = PADDING):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels_1, kernel_size=kernel_size,
                        padding = (4,2), bias=True, padding_mode = 'zeros'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=POOLING_KERNEL_SIZE),
            nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=kernel_size,
                         padding = (0,0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=POOLING_KERNEL_SIZE))
        return block

    def forward(self, x):
        merged_image = torch.zeros(x.size(0),1,14,28)
        merged_image[:,0,:,:14] = x[:,0]
        merged_image[:,0,:,14:] = x[:,1]
        x = merged_image
        x = self.conv_(x) # 1 x 14 x 28 -> 12 x 16 x 28 -> 12 x 8 x 14 -> 32 x 4 x 10 -> 32 x 2 x 5
        x = x.view(-1, 32*2*5)
        x = self.fcn_(x)

        return x

def get_one_image():
    return One_Image(CONV_PARAMETERS, FCN_PARAMETERS)


##################################################################################################################
##################################################################################################################
##################################################################################################################


class Two_LeNet5(nn.Module):
    def __init__(self, conv_parameters, fcn_parameters):
        super(Two_LeNet5, self).__init__()
        self.conv_1 = convolution_block(conv_parameters['in_channels'], conv_parameters['out_channels_1'], conv_parameters['out_channels_2'])
        self.conv_2 = convolution_block(conv_parameters['in_channels'], conv_parameters['out_channels_1'], conv_parameters['out_channels_2'])
        self.max_pool = nn.MaxPool2d(kernel_size=POOLING_KERNEL_SIZE)
        self.fcn_1 = fully_connected_network(fcn_parameters['in_features_b'], fcn_parameters['hd_features'], fcn_parameters['out_features_b'])
        self.fcn_2 = fully_connected_network(fcn_parameters['in_features_b'], fcn_parameters['hd_features'], fcn_parameters['out_features_b'])
        self.fcn_final = nn.Linear(20, 2)

    def forward(self, x):
        (x_first, x_second) = (x[:,0].view(-1, 1, 14, 14), x[:,1].view(-1, 1, 14, 14))
        (x_first, x_second) = (self.max_pool(self.conv_1(x_first)), self.max_pool(self.conv_2(x_second)))
        (x_first, x_second) = (self.fcn_1(x_first.view(-1, 32*5*5)), self.fcn_2(x_second.view(-1, 32*5*5)))
        x = torch.cat((x_first,x_second), 1)
        x = self.fcn_final(x)

        return x

def get_2_LeNet5():
    return Two_LeNet5(CONV_PARAMETERS, FCN_PARAMETERS)
