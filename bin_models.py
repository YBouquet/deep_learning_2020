"""
@author: Yann BOUQUET
"""


import torch
from torch import nn
#from torch.nn import functional as F


POOLING_KERNEL = 2
CONV_KERNEL = 5

class Two_Channels(nn.Module):

    @staticmethod
    def convolution_block(in_channels,mid_channels, out_channels, vector_size, kernel_size=CONV_KERNEL):
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
