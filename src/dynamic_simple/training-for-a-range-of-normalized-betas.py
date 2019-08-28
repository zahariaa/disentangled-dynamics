#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:48:00 2019

@author: benjamin
"""


import numpy as np
import subprocess

gamma_values = np.logspace(-4, 4, 5, base=10)
normalized_beta_values = np.logspace(np.log(.001), np.log(5), 6, base=np.e)

for _, ng in enumerate(gamma_values):

    for _, nb in enumerate(normalized_beta_values):
        b = '--beta=%0.4f' % nb
        g = '--gamma=%0.4f' % ng
        print(b)
        print(g)
        subprocess.call(['python', 'main.py', b, g, '--beta_is_normalized=True', '--max_iter=50000', '--n_latent=10'])
