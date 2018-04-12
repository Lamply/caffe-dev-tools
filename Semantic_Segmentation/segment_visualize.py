#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script visualize the semantic segmentation.
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
caffe_root = '/home/sad/ENet/caffe-enet/'     # Change this to your project's caffe path 
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2
from time import time
import multiprocessing


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

    img = cv2.imread(image_path, 1).astype(np.float32)
    if img is None:
        return

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

    prediction = net.blobs['score_final'].data[0].argmax(axis=0)

    prediction = np.squeeze(prediction)
    prediction = np.resize(prediction, (3, input_shape[2], input_shape[3]))
    prediction = prediction.transpose(1, 2, 0).astype(np.uint8)
    # prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    visualize = np.zeros(prediction.shape, dtype=np.uint8)
    label_colours_bgr = label_colours[:, :, ::-1]
    cv2.LUT(prediction, label_colours_bgr, visualize)  # This function require label_colours_bgr has 3-channel [1,256,3]

    if args.out_dir is not None:
        # input_path_ext = input.split(".")[-1]
        # input_image_name = input.split("/")[-1:][0].replace('.' + input_path_ext, '')
        out_path_im = args.out_dir + input_image_name + '.png'
        out_path_gt = args.out_dir + input_image_name + '.png'

        cv2.imwrite(out_path_im, visualize)
        # cv2.imwrite(out_path_gt, prediction) #  label images, where each pixel has an ID that represents the class


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






