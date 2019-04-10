#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:01:19 2019

@author: benjamin
"""

import argparse
import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore")


from solver import Solver

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)

    net.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='supervised e')

    parser.add_argument('--model', default='decoderBVAE_like_wElu', type=str, help='which model to train (encoderBVAE_like, decoderBVAE_like, decoderBVAE_like_wElu))')

    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--shuffle', default=True, type=str2bool, help='shuffle training data')
    parser.add_argument('--max_iter', default=100000, type=int, help='number of training iterations')

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--n_latent', default=4, type=int, help='dimension of the latent code')
    parser.add_argument('--img_channels', default=1, type=int, help='number of image channels')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='dsprites_circle', type=str, help='dataset name')
    parser.add_argument('--image_size', default=32, type=int, help='image size. now only (32,32) is supported')
    parser.add_argument('--num_workers', default=6, type=int, help='dataloader num_workers')

    """
    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

    parser.add_argument('--gather_step', default=1000, type=int, help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=10000, type=int, help='number of iterations after which loss data is printed and visdom is updated')
    """
    parser.add_argument('--save_step', default=10000, type=int, help='number of iterations after which a checkpoint is saved')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default=None, type=str, help='load previous checkpoint. insert checkpoint filename')

    args = parser.parse_args()

    main(args)