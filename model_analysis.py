# -*- coding:utf-8 -*-
"""
This is a analysis tools for visualizing and quantitative analysis caffe model.

Notice:

   1. There is a 'zeros_cnt' for counting how many kernels are empty ( L1-norm < 1e-7 )

"""
caffe_root = '/home/sad/caffe/'     # Change this to your project's caffe path 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import collections
from collections import OrderedDict
from matplotlib import pyplot as plt
import kmapper as km
import sklearn
import scipy.io as sio
import argparse

__author__ = 'Lamply'
__email__ = 'Lamplykyz@gmail.com'


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file')
    parser.add_argument('--display', help='display curve', action="store_true")

    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()

    caffe.set_mode_cpu()
    net0 = caffe.Net(args.model, args.weights, caffe.TEST)

    keys0 = net0.params.keys()
    # print(keys0)
    Hs = collections.OrderedDict()
    i = 0

    L1_name = []
    L1_conv = collections.OrderedDict()
    L1_zeros = collections.OrderedDict()

    H_scalew = collections.OrderedDict()
    i_scale = 0

    for key0 in keys0:
        # print(key0)
        zeros_cnt = 0
        L1_tmp = []
        param0 = net0.params[key0][0].data
        # print(param0.shape)

        # Just analysis convolution-like 4 dims param
        # Conv w matrix
        if param0.ndim == 4:
            # L1 Norm
            for num in range(param0.shape[0]):
                l1_p = np.sum(np.abs(param0[num]))
                L1_tmp.append(l1_p)
                if l1_p < 1e-7:
                    zeros_cnt += 1
                print(l1_p)
            L1_name.append(key0)
            L1_conv[key0] = L1_tmp
            L1_zeros[key0] = zeros_cnt
            print('\n')

        # Scale w matrix
        if 'scale' in key0:
            H_scalew[i_scale] = param0
            i_scale += 1

    ## print mean, std, L1 zeros num
    l1_mean = [np.mean(H) for i, H in L1_conv.iteritems()]
    l1_std = [np.std(H) for i, H in L1_conv.iteritems()]
    l1_zeros = [H for i, H in L1_zeros.iteritems()]
    cnt = 0
    print('%-20s %-15s %-15s %-20s %-15s\n' % ('layer_name:', 'mean:', 'std:', 'zeros_cnt:', 'shape:'))
    for i, H in L1_conv.iteritems():
        print('%-20s %-15f %-20f %-15d %-15s' % (i, l1_mean[cnt], l1_std[cnt], l1_zeros[cnt], net0.params[i][0].data.shape))
        cnt += 1

    print('zero count: %d' % np.sum(l1_zeros))

    if args.display:
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.plot(l1_mean, 'ob-')
        plt.title('means')
        plt.subplot(132)
        plt.plot(l1_std, 'or-')
        plt.title('stds')
        plt.subplot(133)
        plt.plot(l1_zeros, 'og-')
        plt.title('zeros')
        plt.show()




