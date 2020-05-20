import os

MODEL = '2nets_ws'
LEARNING_RATE  = '7e-5'
BETA1 = '0.5'
BATCH_SIZE = '5'
NUM_EPOCHS = '25'
SEED = '0'
WAL = 1.
if __name__ == '__main__':
    os.system('python main.py --model ' + MODEL + \
        ' --n_epochs '+ NUM_EPOCHS + \
        ' --batch_size '+ BATCH_SIZE + \
        ' --learning_rate '+ LEARNING_RATE + \
        ' --adam_beta1 ' + BETA1 + \
        ' --weight_auxiliary_loss' + WAL)
