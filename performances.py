import os

GETTERS_DICT =  {
                    'net':  {'model' : 'net','optimizer' : 'sgd', 'rate' : '5e-3'}, #model given in the practicals
                    'net2':  {'model' : 'net2','optimizer' : 'sgd', 'rate' : '5e-3'}, #model given in the practicals
                    'lenet5': {'model' : 'lenet5','optimizer' : 'sgd', 'rate' : '5e-3'}, #LeCun's model with correction due to different input format
                    '2nets': {'model' : '2nets','optimizer' : 'adam', 'rate' : '7e-5', 'beta': "0.5", 'weight_auxiliary_loss': '0.', 'wegith_decay' : '0.'},
                    '2nets_ws': {'model' : '2nets_ws','optimizer' : 'adam', 'rate' : '7e-5', 'beta': "0.5", 'weight_auxiliary_loss': '0.','wegith_decay' : '0.'},
                    '2nets_ws_aux': {'model' '2nets_ws': ,'optimizer' : 'adam', 'rate' : '7e-5', 'beta': "0.5", 'weight_auxiliary_loss': '1.', 'wegith_decay' : '0.'},
                    '2nets_ws_aux_do':  {'model' : '2nets_ws_do','optimizer' : 'adam', 'rate' : '6e-4', 'beta': "0.6", 'weight_auxiliary_loss': '1.','wegith_decay' : '0.'},
                    '2nets_ws_aux_do_bn':  {'model' : '2nets_ws_bn','optimizer' : 'adam', 'rate' : '6e-4', 'beta': "0.6", 'weight_auxiliary_loss': '1.','wegith_decay' : '0.'},
                    '2netws_ws_aux_do_wd': {'model' : '2nets_ws_do','optimizer' : 'adam', 'rate' : '6e-4', 'beta': "0.6", 'weight_auxiliary_loss': '1.','wegith_decay' : '1e-5'}
                }

NUM_EPOCHS = 50
BATCH_SIZE = 5

if __name__ == '__main__':
    for architecture, hyperparameters in GETTERS_DICT.item():
        print('============ ARCHITECTURE : '+architecture+' ====================')
        if hyperparameters['optimizer'] == 'sgd':
            os.system('python main.py --model ' + hyperparameters['model'] + \
                ' --n_epochs '+ NUM_EPOCHS + \
                ' --batch_size '+ BATCH_SIZE + \
                ' --optimizer ' + hyperparameters['optimizer'] + \
                ' --learning_rate '+ hyperparameters['rate'])

        else:
            os.system('python main.py --model ' + hyperparameters['model'] + \
                ' --n_epochs '+ NUM_EPOCHS + \
                ' --batch_size '+ BATCH_SIZE + \
                ' --optimizer ' + hyperparameters['optimizer'] + \
                ' --learning_rate '+ hyperparameters['rate'] + \
                ' --adam_beta1 ' + hyperparameters['beta'] + \
                ' --weight_auxiliary_loss' + hyperparameters['weight_auxiliary_loss'] + \
                ' --weight_decay ' + hyperparameters['weight_decay'])