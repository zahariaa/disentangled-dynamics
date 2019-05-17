#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:46:23 2019

@author: benjamin
"""

import numpy as np
import subprocess

proportions = np.linspace(.2,.8,4)

for ii,prop in enumerate(proportions):
    prop_command = '--proportion_train_partition=%0.2f' % prop    
    print(prop_command)
    subprocess.call(['python', 'main.py', '--model=decoderBVAE_like_wElu_SigmoidOutput', prop_command, '--max_iter=200000'])
