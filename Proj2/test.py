import os

ACTIATIONS = ['tanh', 'relu']

if __name__ == '__main__':
    for activation in ACTIVATIONS:
        os.system('python main.py --activation ' + activation)
