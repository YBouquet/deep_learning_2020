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
    'hd_features' : 500,
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
        num_1 = f.relu(self.unshared_linear_1(shared_1.view(-1, 16*5*5)))
        num_2 = f.relu(self.unshared_linear_2(shared_2.view(-1, 16*5*5)))
        comp = self.shared_linear(torch.cat((num_1, num_2), axis = 1))
        return num_1, num_2, comp

def get_2nets():
    return Two_nets(CN_U_PARAMETERS, FN_U_PARAMETERS, CN_S_PARAMETERS, FN_COMP_PARAMETERS)


class Two_nets_ws(nn.Module):
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
        super(Two_nets_ws, self).__init__()
        self.shared_conv_1 =   self.conv_unit(
                                                cn_u_parameters['in_channels'],
                                                cn_u_parameters['out_channels'],
                                                padding = cn_u_parameters['padding']
                                            )
        self.shared_conv_2 =       nn.Sequential(
                                                     nn.Conv2d(
                                                            cn_s_parameters['in_channels'],
                                                            cn_s_parameters['out_channels'],
                                                            kernel_size = CONV_KERNEL,
                                                            ),
                                                     nn.MaxPool2d(kernel_size = POOLING_KERNEL),
                                                     nn.ReLU(),
                                                )
        self.shared_linear_1 = self.u_linear(
                                                fn_u_parameters['in_features'],
                                                fn_u_parameters['hd_features_1'],
                                                fn_u_parameters['hd_features_2'],
                                                fn_u_parameters['out_features']
                                            )
        self.shared_linear_2 =     self.s_linear(
                                                fn_comp_parameters['in_features'],
                                                fn_comp_parameters['hd_features'],
                                                fn_comp_parameters['out_features']
                                            )
    def forward(self, x):
        unshared_1 = self.shared_conv_1(x[:,0].view(-1,1,14,14))
        unshared_2 = self.shared_conv_1(x[:,1].view(-1,1,14,14))
        shared_1 = self.shared_conv_2(unshared_1)
        shared_2 = self.shared_conv_2(unshared_2)
        num_1 = f.relu(self.shared_linear_1(shared_1.view(-1, 16*5*5)))
        num_2 = f.relu(self.shared_linear_1(shared_2.view(-1, 16*5*5)))
        comp = self.shared_linear_2(torch.cat((num_1, num_2), axis = 1))
        return num_1, num_2, comp


class Two_nets_ws_do(nn.Module):
    @staticmethod
    def conv_unit(in_channels, out_channels, padding = 0, padding_mode = 'zeros', kernel_size = CONV_KERNEL):
        return m_conv(in_channels, out_channels, kernel_size, padding = padding, padding_mode = padding_mode)

    @staticmethod
    def s_linear(in_features, hd_features, out_features):
        block = nn.Sequential(
            nn.Linear(in_features, hd_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hd_features, out_features)
        )
        return block

    @staticmethod
    def u_linear(in_features, hd_features_1, hd_features_2, out_features):
        block = nn.Sequential(
            nn.Linear(in_features, hd_features_1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hd_features_1, hd_features_2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hd_features_2, out_features),
        )
        return block


    def __init__(self, cn_u_parameters, fn_u_parameters, cn_s_parameters, fn_comp_parameters):
        super(Two_nets_ws_do, self).__init__()
        self.shared_conv_1 =   self.conv_unit(
                                                cn_u_parameters['in_channels'],
                                                cn_u_parameters['out_channels'],
                                                padding = cn_u_parameters['padding']
                                            )
        self.shared_conv_2 =       nn.Sequential(
                                                     nn.Conv2d(
                                                            cn_s_parameters['in_channels'],
                                                            cn_s_parameters['out_channels'],
                                                            kernel_size = CONV_KERNEL,
                                                            ),
                                                     nn.MaxPool2d(kernel_size = POOLING_KERNEL),
                                                     nn.ReLU(),
                                                )
        self.shared_linear_1 = self.u_linear(
                                                fn_u_parameters['in_features'],
                                                fn_u_parameters['hd_features_1'],
                                                fn_u_parameters['hd_features_2'],
                                                fn_u_parameters['out_features']
                                            )
        self.shared_linear_2 =     self.s_linear(
                                                fn_comp_parameters['in_features'],
                                                fn_comp_parameters['hd_features'],
                                                fn_comp_parameters['out_features']
                                            )
    def forward(self, x):
        unshared_1 = self.shared_conv_1(x[:,0].view(-1,1,14,14))
        unshared_2 = self.shared_conv_1(x[:,1].view(-1,1,14,14))
        shared_1 = self.shared_conv_2(unshared_1)
        shared_2 = self.shared_conv_2(unshared_2)
        num_1 = f.relu(self.shared_linear_1(shared_1.view(-1, 16*5*5)))
        num_2 = f.relu(self.shared_linear_1(shared_2.view(-1, 16*5*5)))
        comp = self.shared_linear_2(torch.cat((num_1, num_2), axis = 1))
        return num_1, num_2, comp

class Two_nets_ws_bn(nn.Module):
    @staticmethod
    def conv_unit(in_channels, out_channels, padding = 0, padding_mode = 'zeros', kernel_size = CONV_KERNEL):
        return m_conv(in_channels, out_channels, kernel_size, padding = padding, padding_mode = padding_mode)

    @staticmethod
    def s_linear(in_features, hd_features, out_features):
        block = nn.Sequential(
            nn.Linear(in_features, hd_features),
            nn.BatchNorm1d(hd_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hd_features, out_features)
        )
        return block

    @staticmethod
    def u_linear(in_features, hd_features_1, hd_features_2, out_features):
        block = nn.Sequential(
            nn.Linear(in_features, hd_features_1),
            nn.BatchNorm1d(hd_features_1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hd_features_1, hd_features_2),
            nn.BatchNorm1d(hd_features_2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hd_features_2, out_features),
        )
        return block


    def __init__(self, cn_u_parameters, fn_u_parameters, cn_s_parameters, fn_comp_parameters):
        super(Two_nets_ws_bn, self).__init__()
        self.shared_conv_1 =   self.conv_unit(
                                                cn_u_parameters['in_channels'],
                                                cn_u_parameters['out_channels'],
                                                padding = cn_u_parameters['padding']
                                            )
        self.shared_conv_2 =       nn.Sequential(
                                                     nn.Conv2d(
                                                            cn_s_parameters['in_channels'],
                                                            cn_s_parameters['out_channels'],
                                                            kernel_size = CONV_KERNEL,
                                                            ),
                                                     nn.MaxPool2d(kernel_size = POOLING_KERNEL),
                                                     nn.ReLU(),
                                                )
        self.shared_linear_1 = self.u_linear(
                                                fn_u_parameters['in_features'],
                                                fn_u_parameters['hd_features_1'],
                                                fn_u_parameters['hd_features_2'],
                                                fn_u_parameters['out_features']
                                            )
        self.shared_linear_2 =     self.s_linear(
                                                fn_comp_parameters['in_features'],
                                                fn_comp_parameters['hd_features'],
                                                fn_comp_parameters['out_features']
                                            )

    def forward(self, x):
        unshared_1 = self.shared_conv_1(x[:,0].view(-1,1,14,14))
        unshared_2 = self.shared_conv_1(x[:,1].view(-1,1,14,14))
        shared_1 = self.shared_conv_2(unshared_1)
        shared_2 = self.shared_conv_2(unshared_2)
        num_1 = f.relu(self.shared_linear_1(shared_1.view(-1, 16*5*5)))
        num_2 = f.relu(self.shared_linear_1(shared_2.view(-1, 16*5*5)))
        comp = self.shared_linear_2(torch.cat((num_1, num_2), axis = 1))
        return num_1, num_2, comp

class Required(nn.Module):

    @staticmethod
    def s_linear(in_features, hd_features, out_features):
        block = nn.Sequential(
            nn.Linear(in_features, hd_features),
            nn.ReLU(),
            nn.Linear(hd_features, hd_features),
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
        super(Required, self).__init__()
        self.conv = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 5,  padding = 0, padding_mode = 'zeros')
        self.shared_linear_1 = self.u_linear(
                                                3*10*10,
                                                fn_u_parameters['hd_features_1'],
                                                fn_u_parameters['hd_features_2'],
                                                fn_u_parameters['out_features']
                                            )
        self.shared_linear_2 =     self.s_linear(
                                                fn_comp_parameters['in_features'],
                                                50,
                                                fn_comp_parameters['out_features']
                                            )
    def forward(self, x):
        x_1 = f.relu(self.conv(x[:,0].view(-1,1,14,14)))
        x_2 = f.relu(self.conv(x[:,1].view(-1,1,14,14)))
        num_1 = f.relu(self.shared_linear_1(x_1.view(-1,3*10*10)))
        num_2 = f.relu(self.shared_linear_1(x_2.view(-1,3*10*10)))
        comp = self.shared_linear_2(torch.cat((num_1, num_2), axis = 1))
        return comp

def get_2nets_ws():
    return Two_nets_ws(CN_U_PARAMETERS, FN_U_PARAMETERS, CN_S_PARAMETERS, FN_COMP_PARAMETERS)

def get_required():
    return Required(CN_U_PARAMETERS, FN_U_PARAMETERS, CN_S_PARAMETERS, FN_COMP_PARAMETERS)

def get_2nets_ws_do():
    return Two_nets_ws_do(CN_U_PARAMETERS, FN_U_PARAMETERS, CN_S_PARAMETERS, FN_COMP_PARAMETERS)

def get_2nets_ws_bn():
    return Two_nets_ws_bn(CN_U_PARAMETERS, FN_U_PARAMETERS, CN_S_PARAMETERS, FN_COMP_PARAMETERS)
