#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script merge the semantic segmentation result.

Notice:

   1. Support multiprocessing
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys

caffe_root = '/home/sad/ENet/caffe-enet/'   # Change this to your project's caffe path 
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2
from time import time
import multiprocessing


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True, help='input image path file for merging')
    parser.add_argument('--labels', type=str, required=True, help='label result path for merging')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory in which the segmented images '
                                                                  'should be stored')
    return parser


def merge_label_image(label_path):
    label = cv2.imread(args.labels + label_path, 1).astype(np.float32)
    if label is None:
        return

    input_path_ext = label_path.split(".")[-1]
    input_image_name = label_path.split("/")[-1:][0].replace('.' + input_path_ext, '')

    img = cv2.imread(args.images + input_image_name + '.jpg', 1).astype(np.float32)
    if img is None:
        return

    resize_label = cv2.resize(label, (img.shape[1], img.shape[0]))

    alpha = 0.5
    foreground_tmp = np.sum(resize_label, 2)
    foreground = np.where(foreground_tmp != 0)
    merge = img
    merge[foreground] = img[foreground] * (1 - alpha) + alpha * resize_label[foreground]

    if args.out_dir is not None:
        out_path_im = args.out_dir + input_image_name + '.png'
        cv2.imwrite(out_path_im, merge)


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()

    pool = multiprocessing.Pool(8)

    if os.path.exists(args.images) and os.path.exists(args.labels):
        lb_files = os.listdir(args.labels)
        pool.map(merge_label_image, lb_files)
        pool.close()
        pool.join()






