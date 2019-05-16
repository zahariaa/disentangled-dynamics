#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:48:00 2019

@author: benjamin
"""


import numpy as np
import subprocess

normalized_beta_values = np.logspace(np.log(.001), np.log(5), 6, base=np.e)
for ii, nb in enumerate(normalized_beta_values):
    b = '--beta=%0.4f' % nb    
    print(b)
    subprocess.call(['python', 'main.py', b, '--beta_is_normalized=True', '--max_iter=500000'])