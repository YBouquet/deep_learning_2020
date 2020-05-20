import os

MODEL = '2nets_ws_required'
LEARNING_RATE  = '6e-4'
BETA1 = '0.9'
BATCH_SIZE = '250'
NUM_EPOCHS = '25'
SEED = '0'
WAL = '0.'
OPTIMIZER = "adam"
if __name__ == '__main__':
    os.system('python main.py --model ' + MODEL + \
        ' --optimizer ' + OPTIMIZER + \
        ' --n_epochs '+ NUM_EPOCHS + \
        ' --batch_size '+ BATCH_SIZE + \
        ' --learning_rate '+ LEARNING_RATE + \
        ' --adam_beta1 ' + BETA1 + \
        ' --weight_auxiliary_loss ' + WAL)
