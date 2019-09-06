#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:48:00 2019

@author: benjamin
"""


import numpy as np
import subprocess

gamma_values = np.logspace(-4, 4, 5, base=10)
gamma_values = np.insert(gamma_values,0,0)

for _, ng in enumerate(gamma_values):

    g = '--gamma=%s' % ng
    print(g)
    subprocess.call(['python', 'main.py', g, '--max_iter=50000', '--n_latent=10'])
