#!/usr/bin/env python3
#!/usr/bin/env python2

import os

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
