#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script produce Precision-Recall cruve MATLAB matrix.

Notice:
    1. label 0 as foreground and label 1 as background

    2. It seems much faster if process in MATLAB
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
import cv2
import scipy.io
from time import time
# from image_fix_size import fix_resize


__author__ = 'Lamply'
__email__ = 'lamplykyz@gmail.com'
__data__ = '18th Apr, 2018'

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_dir', type=str, default=None, help='output directory in which the segmented images '
                                                                  'should be stored')
    parser.add_argument('--dataset', type=str, default='dev', help='dev or hard')
 
    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()

    if args.dataset == 'hard':
        input_files = open('/home/sad/ENet/dataset/video_hard.txt')
    elif args.dataset == 'dev':
        input_files = open('/home/sad/ENet/dataset/protrait_test_list.txt')
    elif args.dataset == 'hard_square':
        input_files = open('/home/sad/ENet/dataset/video_hard_square.txt')
    elif args.dataset == 'dev_square':
        input_files = open('/home/sad/ENet/dataset/protrait_test_list_square.txt')
    else:
        print("error: error dataset selected.")
        pass

    files_list = input_files.readlines()

    thresholdC = np.linspace(1, 0, 100)
    precisionC = np.array([], dtype=np.float32)
    recallC = np.array([], dtype=np.float32)
    for threshold in thresholdC:
        precision = np.array([], dtype=np.float32)
        recall = np.array([], dtype=np.float32)

        for input in files_list:
            input = input.strip()
            input = input.split(' ')

            input_path_ext = input[0].split(".")[-1]
            input_image_name = input[0].split("/")[-1:][0].replace('.' + input_path_ext, '')

            resize_pred = np.load(args.seg_dir + input_image_name + '.npy')
            if resize_pred is None:
                print("Cannot load segment result.")
                continue

            prediction = 1.0*(resize_pred > threshold)

            # Test score
            label_name = input[1]
            idx = label_name.find(input_image_name)
            if idx is not -1:
                test_label_ = cv2.imread(label_name, 0).astype(np.float32)
                if test_label_ is None:
                    print("empty test label.")
                    break
                test_label = 1.0 - test_label_
                recall = np.append(recall, np.sum(test_label * prediction) / (np.sum(test_label)+(1e-7)))
                precision = np.append(precision, np.sum(test_label * prediction) / (np.sum(prediction)+(1e-7)))

            else:
                print('error: not corresponding labels.')
                break

        precisionC = np.append(precisionC, np.mean(precision))
        recallC = np.append(recallC, np.mean(recall))

    input_files.close()

    scipy.io.savemat('PR_curve.mat', {'precision': precisionC, 'recall': recallC, 'threshold': thresholdC})




