#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:28:12 2020

@author: thomas
"""
### Python modules
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import torch
import math

### Project modules
import arguments
import modules as bf
import helpers as h


DICT = {
        'relu' : bf.ReLU,
        'tanh' : bf.Tanh
        }


# tracer cercle
# axes orthonormaux, centrés, normalisation ? v

def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection


def main(args):
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    train_set, train_target, test_set, test_target = h.generate_sets(size = 1000)
    train_target, test_target = h.ohe(train_target, test_target)
    
    for activation in ['relu','tanh']:
        fig, ax = plt.subplots(dpi = 200)  # Create a figure and an axes.
        activation_function = DICT[activation]
        model = bf.Sequential( bf.Linear(2,25),  activation_function(), bf.Linear(25,25), activation_function(), bf.Linear(25,25),  activation_function(), bf.Linear(25,2))
        train_l, test_l = h.train_model(model, train_set, train_target,test_set, test_target, lr = 1e-2, num_epoch = args.n_epochs, batch = args.batch_size)
        
        tr_error = h.nb_classification_errors(model, train_set, train_target, args.batch_size) / 10
        te_error = h.nb_classification_errors(model, test_set, test_target, args.batch_size) / 10
        print(f"Train accuracy = {100 - tr_error} %, test accuracy = {100 - te_error} %")
            
        blues, reds, oranges, purples = 0, 0, 0, 0
        for nb in range(len(train_set)):
            point = test_set[nb]
            if test_target[nb][0] == 0: # in
                output = model.forward(test_set.narrow(0,nb,1))
                _, predicted_class = output.max(1)
                if test_target[nb, predicted_class[0]] <= 0 :
                    color = 'red'
                    reds += 1
                else:
                    color = 'blue'
                    blues += 1
            else: # out
                output = model.forward(test_set.narrow(0,nb,1))
                _, predicted_class = output.max(1)
                if test_target[nb, predicted_class[0]] <= 0 :
                    color = 'orange'
                    oranges += 1
                else:
                    color = 'purple'
                    purples += 1
            ax.plot(point[0], point[1], color = color, marker=".")
    
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('Absciss')  # Add an x-label to the axes.
        ax.set_ylabel('Ordinate')  # Add a y-label to the axes.
        ax.set_title('Results with '+ activation + ' activation,\n' + f"test accuracy = {100 - te_error:0.2f}%")
        
        #circle = plt.Circle((0.5,0.5),radius = 1/math.sqrt(2*math.pi))
        circles(0.5, 0.5, 1/math.sqrt(2*math.pi), 'k', alpha=0.5)
        
        blue = mlines.Line2D([], [], color='blue', marker='.', label=str(blues) + ' In points well classified', linestyle = 'None')
        red = mlines.Line2D([], [], color='red', marker='.', label=str(reds) + ' In points misclassified', linestyle = 'None')
        purple = mlines.Line2D([], [], color='purple', marker='.', label=str(purples) + ' Out points well classified', linestyle = 'None')
        orange = mlines.Line2D([], [], color='orange', marker='.', label=str(oranges) + ' Out points misclassified', linestyle = 'None')
        black = mlines.Line2D([], [], color='black', alpha=0.5, marker='o', label='Disk centered at $(0.5, 0.5)$\n of radius $\dfrac{1}{\sqrt{2 \pi}}$', linestyle = 'None')
        lgd = plt.legend(handles=[blue, red, purple, orange, black], loc='center left', bbox_to_anchor=(1, 0.5))
        
        #ax.legend()
        plt.show()
        filename = activation + '_results.png'
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    main(arguments.get_args())
