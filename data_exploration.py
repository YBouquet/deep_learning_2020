#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 19:51:33 2020

@author: thomas
"""
import numpy as np
import dlc_practical_prologue as prologue

counts = []
for nb in range(100):
    tr_input, tr_target, tr_figure_target, te_input, te_target,_ = prologue.generate_pair_sets(1000)
    counts.append(len([t for t in te_target if t==1]))
print(np.mean(counts), np.std(counts))