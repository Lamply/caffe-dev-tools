#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script save the semantic segmentation result.

Notice:
    1. class limit to [0, 255]
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
caffe_root = '/home/sad/ENet/caffe-enet/'  # Change this to the absolute directory to ENet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2
from time import time
from image_fix_size import fix_resize


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input lists file for inference')
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file')
    parser.add_argument('--colours', type=str, required=True, help='label colours')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory in which the segmented images '
                                                                   'should be stored')
    parser.add_argument('--use_mean', help='implement with mean subtraction', action="store_true")
    parser.add_argument('--use_scale', help='implement with value scale', action="store_true")
 
    return parser


def predict_label(image_path):
    path = image_path.split(' ')
    if len(path) == 2:
        image_path = path[0]
    elif len(path) != 1:
        print("Input not valid.")
        return

    img = cv2.imread(image_path, 1)
    if img is None:
        return

    img = img.astype(np.float32)

    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['score_final'].data.shape

    input_path_ext = image_path.split(".")[-1]
    input_image_name = image_path.split("/")[-1:][0].replace('.' + input_path_ext, '')

    resize_img = cv2.resize(img, (input_shape[3], input_shape[2]))
    input_img = resize_img

    if args.use_mean:
        input_img = input_img - [103.939, 116.779, 123.68]

    if args.use_scale:
        input_img = input_img * 0.017

    input_image = input_img.transpose((2, 0, 1))
    input_image = np.asarray([input_image])

    net.blobs['data'].data[...] = input_image

    start = time()
    net.forward()
    end = time()
    print("time: " + str((end - start) * 1000) + 'ms')

    pred = net.blobs['score_final'].data[0].argmax(axis=0).astype(np.uint8)

    prediction = cv2.resize(pred, (img.shape[1], img.shape[0]))
    prediction = np.resize(prediction, (3, prediction.shape[0], prediction.shape[1]))
    prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

    visualize = np.zeros(prediction.shape, dtype=np.uint8)
    # label_colours_bgr = label_colours[:, :, ::-1]
    cv2.LUT(prediction, label_colours, visualize)  # This function require label_colours_bgr has 3-channel [1,256,3]

    if args.out_dir is not None:
        out_path_im = args.out_dir + input_image_name + '.png'

        cv2.imwrite(out_path_im, visualize)


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(args.model, args.weights, caffe.TEST)

    if args.use_mean:
        print('use mean substraction')

    if args.use_scale:
        print('use scale')

    label_colours = cv2.imread(args.colours).astype(np.uint8)

    input_files = open(args.input)

    for input_file in input_files.readlines():
        input_file = input_file.strip()
        predict_label(image_path=input_file)

    input_files.close()






