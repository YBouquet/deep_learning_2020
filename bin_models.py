"""
@author: Yann BOUQUET
"""

import torch
from torch import nn
from torch.nn import functional as f
#from torch.nn import functional as F


POOLING_KERNEL = 2
CONV_KERNEL = 5

class m_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0, padding_mode = 'zeros'):
        super(m_conv, self).__init__()
        self.conv_ = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,  padding = padding, padding_mode = padding_mode)
    def forward(self, x):
        return f.relu(self.conv_(x))

CN_U_PARAMETERS = {
    'in_channels' : 1,
    'out_channels': 6,
    'padding' : 2
}

CN_S_PARAMETERS = {
    'in_channels' : CN_U_PARAMETERS['out_channels'],
    'out_channels' : 16,
    'padding' : 0
}

FN_U_PARAMETERS = {
    'in_features' : CN_S_PARAMETERS['out_channels'] * 5 * 5,
    'hd_features_1' : 120 ,
    'hd_features_2': 84,
    'out_features' : 10
}

FN_COMP_PARAMETERS = {
    'in_features' : 20,
    'hd_features' : 8,
    'out_features' : 2
}

class Two_nets(nn.Module):
    @staticmethod
    def conv_unit(in_channels, out_channels, padding = 0, padding_mode = 'zeros', kernel_size = CONV_KERNEL):
        return m_conv(in_channels, out_channels, kernel_size, padding = padding, padding_mode = padding_mode)

    @staticmethod
    def s_linear(in_features, hd_features, out_features):
        block = nn.Sequential(
            nn.Linear(in_features, hd_features),
            nn.ReLU(),
            nn.Linear(hd_features, out_features)

        )
        return block

    @staticmethod
    def u_linear(in_features, hd_features_1, hd_features_2, out_features):
        block = nn.Sequential(
            nn.Linear(in_features, hd_features_1),
            nn.ReLU(),
            nn.Linear(hd_features_1, hd_features_2),
            nn.ReLU(),
            nn.Linear(hd_features_2, out_features)
        )
        return block


    def __init__(self, cn_u_parameters, fn_u_parameters, cn_s_parameters, fn_comp_parameters):
        super(Two_nets, self).__init__()
        self.unshared_conv_1 =   self.conv_unit(
                                                cn_u_parameters['in_channels'],
                                                cn_u_parameters['out_channels'],
                                                padding = cn_u_parameters['padding']
                                            )
        self.unshared_conv_2 =   self.conv_unit(
                                                cn_u_parameters['in_channels'],
                                                cn_u_parameters['out_channels'],
                                                padding = cn_u_parameters['padding']
                                            )
        self.shared_conv =       nn.Sequential(
                                                     nn.Conv2d(
                                                            cn_s_parameters['in_channels'],
                                                            cn_s_parameters['out_channels'],
                                                            kernel_size = CONV_KERNEL,
                                                            ),
                                                     nn.MaxPool2d(kernel_size = POOLING_KERNEL),
                                                     nn.ReLU(),
                                                )
        self.unshared_linear_1 = self.u_linear(
                                                fn_u_parameters['in_features'],
                                                fn_u_parameters['hd_features_1'],
                                                fn_u_parameters['hd_features_2'],
                                                fn_u_parameters['out_features']
                                            )
        self.unshared_linear_2 = self.u_linear(
                                                fn_u_parameters['in_features'],
                                                fn_u_parameters['hd_features_1'],
                                                fn_u_parameters['hd_features_2'],
                                                fn_u_parameters['out_features']
                                            )
        self.shared_linear =     self.s_linear(
                                                fn_comp_parameters['in_features'],
                                                fn_comp_parameters['hd_features'],
                                                fn_comp_parameters['out_features']
                                            )
    def forward(self, x):
        unshared_1 = self.unshared_conv_1(x[:,0].view(-1,1,14,14))
        unshared_2 = self.unshared_conv_2(x[:,1].view(-1,1,14,14))
        shared_1 = self.shared_conv(unshared_1)
        shared_2 = self.shared_conv(unshared_2)
        num_1 = self.unshared_linear_1(shared_1.view(-1, 16*5*5))
        num_2 = self.unshared_linear_2(shared_2.view(-1, 16*5*5))
        comp = self.shared_linear(torch.cat((num_1, num_2), axis = 1))
        return comp

def get_2nets():
    return Two_nets(CN_U_PARAMETERS, FN_U_PARAMETERS, CN_S_PARAMETERS, FN_COMP_PARAMETERS)

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
        
