#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script visualize the semantic segmentation of ENet.
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

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--iters', type=int, required=True, help='training iters')
    parser.add_argument('--bias', type=int, default=0, help='multi-test start bias iters')
    parser.add_argument('--interval', type=int, required=True, help='training interval')
    parser.add_argument('--weights_dir', type=str, default=None, help='weights directory in which the trained weights '
                                                                      'should be stored')
    parser.add_argument('--prefix', type=str, required=True, help='training weights prefix')
    parser.add_argument('--use_mean', help='implement with mean subtraction', action="store_true")
    parser.add_argument('--use_scale', help='implement with value scale', action="store_true")
    parser.add_argument('--dataset', type=str, default='dev', help='dev or hard')

    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(args.model, caffe.TEST)

    interval = args.interval
    mIoU = np.array([], np.float32)
    stdIoU = np.array([], np.float32)
    iter_cnt = 0

    for witer in range(args.bias+interval, args.iters+interval, interval):
        wname = args.weights_dir + args.prefix + '_iter_' + str(witer) + '.caffemodel'

        net.copy_from(wname)

        input_shape = net.blobs['data'].data.shape
        output_shape = net.blobs['label'].data.shape

        if args.dataset == 'hard':
            input_files = open('/home/sad/ENet/dataset/video_hard_images.txt')
            labels_files = open('/home/sad/ENet/dataset/video_hard_labels.txt')
        elif args.dataset == 'dev':
            input_files = open('/home/sad/ENet/dataset/test_images.txt')
            labels_files = open('/home/sad/ENet/dataset/test_label.txt')
        elif args.dataset == 'hard_square':
            input_files = open('/home/sad/ENet/dataset/video_hard_images_square.txt')
            labels_files = open('/home/sad/ENet/dataset/video_hard_labels_square.txt')
        elif args.dataset == 'dev_square':
            input_files = open('/home/sad/ENet/dataset/test_images_square.txt')
            labels_files = open('/home/sad/ENet/dataset/test_label_square.txt')
        else:
            print("error: error dataset selected.")
            pass

        labels = labels_files.readlines()

        cnt = 0

        if args.use_mean:
            print('use mean substraction')

        if args.use_scale:
            print('use scale')

        try:
            IoU = np.array([], dtype=np.float32)
            for input in input_files.readlines():
                input = input.strip()
                img = cv2.imread(input, 1).astype(np.float32)
                if img is None:
                    continue

                input_path_ext = input.split(".")[-1]
                input_image_name = input.split("/")[-1:][0].replace('.' + input_path_ext, '')

                resize_img = cv2.resize(img, (input_shape[3], input_shape[2]))
                input_img = resize_img

                if args.use_mean:
                    input_img = input_img - [103.939, 116.779, 123.68]

                if args.use_scale:
                    input_img = input_img * 0.017

                input_image = input_img.transpose((2, 0, 1))
                input_image = np.asarray([input_image])

                net.blobs['data'].data[...] = input_image

                net.forward()
                # out = net.forward_all(**{net.inputs[0]: input_image})

                pred = net.blobs['label'].data

                pred = np.squeeze(pred)[0]
                # print(pred.shape)

                prediction = 1.0 * (pred > 0.5)

                # IoU test
                label_name = labels[cnt].strip()
                idx = label_name.find(input_image_name)
                if idx is not -1:
                    test_label_ = cv2.imread(label_name, 0).astype(np.float32)
                    if test_label_ is None:
                        print("empty test label.")
                        break
                    # test_label = np.asarray([test_label_], dtype=np.float32)
                    # test_label = test_label_/255.0
                    test_label_ = cv2.resize(test_label_, (input_shape[3], input_shape[2]), interpolation=cv2.INTER_NEAREST)
                    test_label = 1.0 - test_label_
                    IoU = np.append(IoU, np.sum(test_label * prediction) / (np.sum(test_label) + np.sum((1.0 - test_label) * prediction)))
                    # print('%s IoU: %f' % (input_image_name, IoU))

                    # recall = np.sum(np.dot(test_label, prediction)) / np.sum(test_label)
                    # precision = np.sum(np.dot(test_label, prediction)) / np.sum(prediction)

                else:
                    print('error: not corresponding labels.')
                    break

                cnt += 1

            mIoU = np.append(mIoU, IoU.mean())
            stdIoU = np.append(stdIoU, IoU.std())

        except Exception, error:
            print error.message

        labels_files.close()
        input_files.close()

        print('%-20s %-20s %-20s' % ('iters:', 'mIoU:', 'stdIoU:'))
        print('%-20d %-20f %-20f' % (witer, mIoU[iter_cnt], stdIoU[iter_cnt]))
        iter_cnt += 1

    i = 0
    print('%-20s %-20s %-20s' % ('iters:', 'mIoU:', 'stdIoU:'))
    for idx in range(args.bias+interval, args.iters+interval, interval):
        print('%-20d %-20f %-20f' % (idx, mIoU[i], stdIoU[i]))
        i += 1





