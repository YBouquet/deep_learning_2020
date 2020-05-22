#!/usr/bin/env python3
#!/usr/bin/env python2

import os

ACTIVATIONS = ['tanh', 'relu']

if __name__ == '__main__':
    for activation in ACTIVATIONS:
        print("=========================== TRAINING WITH "+activation+" ACTIVATION ===========================")
        os.system('python main.py --activation ' + activation + ' --n_epochs ' + '7000')
