#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:04:58 2020

@author: thomas
"""
### Python modules
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import torch
import math

### Project modules
import helpers as h
from plot_results import circles




torch.manual_seed(0)
_, train_target, test_set, test_target = h.generate_sets(size = 1000)
_, test_target = h.ohe(train_target, test_target)


fig, ax = plt.subplots(dpi = 200)  

    
blues, purples = 0, 0
for nb in range(len(test_set)):
    point = test_set[nb]
    if test_target[nb][0] == 0: # in
        color = 'blue'
        blues += 1
    else: # out
        color = 'purple'
        purples += 1
    ax.plot(point[0], point[1], color = color, marker=".")

ax.set_aspect('equal', 'box')
ax.set_xlabel('Absciss')  # Add an x-label to the axes.
ax.set_ylabel('Ordinate')  # Add a y-label to the axes.

ax.set_title('Data location\nclasses ratio = $\dfrac{521}{1000}$ = ' + f"{blues/(blues + purples):0.2f}" + r"$\approx$ 0.5 = 2$\pi$($\dfrac{1}{\sqrt{2 \pi}}$)²")

circles(0.5, 0.5, 1/math.sqrt(2*math.pi), 'k', alpha=0.5)

blue = mlines.Line2D([], [], color='blue', marker='.', label=str(blues) + ' In points', linestyle = 'None')
purple = mlines.Line2D([], [], color='purple', marker='.', label=str(purples) + ' Out points', linestyle = 'None')
black = mlines.Line2D([], [], color='black', alpha=0.5, marker='o', label='Disk centered at $(0.5, 0.5)$\n of radius $\dfrac{1}{\sqrt{2 \pi}}$', linestyle = 'None')
lgd = plt.legend(handles=[blue, purple, black], loc='center left', bbox_to_anchor=(1, 0.5))

filename = 'data.png'
fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
