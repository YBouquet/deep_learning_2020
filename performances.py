import os

GETTERS_DICT =  {
                    #'net':                                  {'model' : 'net', 'criterion' : 'mse','optimizer' : 'sgd', 'rate' : '5e-3'}, #model given in the practicals
                    #'net2':                                 {'model' : 'net2','criterion' : 'mse','optimizer' : 'sgd', 'rate' : '5e-3'}, #model given in the practicals
                    #'lenet5':                               {'model' : 'lenet5','criterion' : 'mse','optimizer' : 'sgd', 'rate' : '5e-3'}, #LeCun's model with correction due to different input format
                    #'2nets_sgd':                            {'model' : '2nets','criterion' : 'ce','optimizer' : 'sgd', 'rate' : '5e-3', 'weight_auxiliary_loss': '0.', 'weight_decay' : '0.'},
                    #'2nets_ws_sgd':                         {'model' : '2nets_ws','criterion' : 'ce','optimizer' : 'sgd', 'rate' : '5e-3', 'weight_auxiliary_loss': '0.', 'weight_decay' : '0.'},
                    #'2nets_ws_aux_sgd':                     {'model' : '2nets_ws','criterion' : 'ce','optimizer' : 'sgd', 'rate' : '5e-3', 'weight_auxiliary_loss': '1.', 'weight_decay' : '0.'},
                    #'2nets_ws_aux_adam':                    {'model' : '2nets_ws','criterion' : 'ce','optimizer' : 'adam', 'rate' : '7e-5', 'beta': "0.5", 'weight_auxiliary_loss': '1.', 'weight_decay' : '0.'},
                    #'2nets_ws_aux_do_adam':                 {'model' : '2nets_ws_do','criterion' : 'ce','optimizer' : 'adam', 'rate' : '6e-4', 'beta': "0.6", 'weight_auxiliary_loss': '1.','weight_decay' : '0.'},
                    #'2nets_ws_aux_do_bn_adam':              {'model' : '2nets_ws_bn','criterion' : 'ce','optimizer' : 'adam', 'rate' : '6e-4', 'beta': "0.6", 'weight_auxiliary_loss': '1.','weight_decay' : '0.'},
                    #'2netws_ws_aux_do_wd_adam':             {'model' : '2nets_ws_do','criterion' : 'ce','optimizer' : 'adam', 'rate' : '6e-4', 'beta': "0.9", 'weight_auxiliary_loss': '1.','weight_decay' : '1e-5'},
                    #'2netws_ws_aux_do_wd_adam_shuffle':     {'model' : '2nets_ws_do','criterion' : 'ce','optimizer' : 'adam', 'rate' : '6e-4', 'beta': "0.9", 'weight_auxiliary_loss': '1.','weight_decay' : '1e-5', 'shuffle' : 'True'},
                    'required':                             {'model' : '2nets_ws','criterion' : 'ce','optimizer' : 'adam', 'rate' : '4e-3', 'beta': "0.6", 'weight_auxiliary_loss': '0.', 'weight_decay' : '0.'}
                }

NUM_EPOCHS = '50'
BATCH_SIZE = '5'

if __name__ == '__main__':
    for architecture, hyperparameters in GETTERS_DICT.items():
        print('============ ARCHITECTURE : '+architecture+' ====================')
        command = 'python main.py --model ' + hyperparameters['model'] + \
            ' --n_epochs '+ NUM_EPOCHS + \
            ' --batch_size '+ BATCH_SIZE + \
            ' --criterion ' + hyperparameters['criterion'] + \
            ' --optimizer ' + hyperparameters['optimizer'] + \
            ' --learning_rate '+ hyperparameters['rate']

        if 'weight_auxiliary_loss' in hyperparameters.keys():
            command += ' --weight_auxiliary_loss ' + hyperparameters['weight_auxiliary_loss']

        if hyperparameters['optimizer'] == 'adam':
            command += ' --adam_beta1 ' + hyperparameters['beta'] + ' --weight_decay ' + hyperparameters['weight_decay']

        if 'shuffle' in hyperparameters.keys():
            command += ' --shuffling ' + hyperparameters['shuffle']

        os.system(command)
