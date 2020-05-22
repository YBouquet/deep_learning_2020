#!/usr/bin/env python3
#!/usr/bin/env python2

import os
'''
This code generate the script that compute the performance of the convnet with 50000 parameters in 25 epochs in less than 2 seconds for each round
and a mean test accuracy of 85%
'''

MODEL = 'required'
LEARNING_RATE  = '4e-3'
BETA1 = '0.6'
BATCH_SIZE = '250'
NUM_EPOCHS = '25'
SEED = '0'
WAL = '0.'
OPTIMIZER = "adam"
SHUFFLING = "False"

if __name__ == '__main__':
    os.system('python main.py --model ' + MODEL + \
        ' --optimizer ' + OPTIMIZER + \
        ' --n_epochs '+ NUM_EPOCHS + \
        ' --batch_size '+ BATCH_SIZE + \
        ' --learning_rate '+ LEARNING_RATE + \
        ' --adam_beta1 ' + BETA1 + \
        ' --weight_auxiliary_loss ' + WAL + \
        ' --shuffling ' + SHUFFLING)
