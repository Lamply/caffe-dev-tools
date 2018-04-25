#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script is just use for 2-classes semantic segmentation.

Notice:
    1. label 0 as foreground and label 1 as background

    2. 0.5 as threshold for prediction
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
# from image_fix_size import fix_resize


__author__ = 'Lamply'
__email__ = 'lamplykyz@gmail.com'
__data__ = '2nd Feb, 2018'


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file')
    # parser.add_argument('--colours', type=str, required=True, help='label colours')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory in which the segmented images '
                                                                  'should be stored')
    parser.add_argument('--use_mean', help='implement with mean subtraction', action="store_true")
    parser.add_argument('--use_scale', help='implement with value scale', action="store_true")
    parser.add_argument('--dataset', type=str, default='dev', help='dev or hard')
    parser.add_argument('--prior', type=str, default=None, help='use prior as input')
 
    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(args.model, args.weights, caffe.TEST)

    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['label'].data.shape

    # label_colours = cv2.imread(args.colours, 1).astype(np.uint8)

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

    cnt = 0
    mIoU = 0

    if args.use_mean:
        print('use mean substraction')

    if args.use_scale:
        print('use scale')

    if args.prior is not None:
        prior = np.load(args.prior)
        if prior is None:
            print("prior load error")
        prior = cv2.resize(prior, (input_shape[3], input_shape[2]))
        input_prior = prior.reshape((1, 1, prior.shape[0], prior.shape[1]))

    IoU = np.array([], dtype=np.float32)
    try:
        for input in input_files.readlines():
            input = input.strip()
            input = input.split(' ')
            img = cv2.imread(input[0], 1).astype(np.float32)
            if img is None:
                continue

            input_path_ext = input[0].split(".")[-1]
            input_image_name = input[0].split("/")[-1:][0].replace('.' + input_path_ext, '')

            img_height = img.shape[0]
            img_width = img.shape[1]
            # if (input_shape[3] * img_height / img_width) != input_shape[2]:
            #     print("no")
            #     continue

            resize_img = cv2.resize(img, (input_shape[3], input_shape[2]))
            input_img = resize_img

            if args.use_mean:
                input_img = input_img - [103.939, 116.779, 123.68]

            if args.use_scale:
                input_img = input_img * 0.017

            input_image = input_img.transpose((2, 0, 1))
            input_image = np.asarray([input_image])

            net.blobs['data'].data[...] = input_image

            if args.prior is not None:
                net.blobs['prior'].data[...] = input_prior

            start = time()
            net.forward()
            # out = net.forward_all(**{net.inputs[0]: input_image})
            end = time()
            print("time: " + str((end-start)*1000) + 'ms')

            pred = net.blobs['label'].data

            pred = np.squeeze(pred)[0]       # [128, 96] or [256, 192] etc.

            resize_pred = pred.reshape(pred.shape[0], pred.shape[1], 1)
            resize_pred = cv2.resize(resize_pred, (img.shape[1], img.shape[0]))
            # cv2.imshow('what', (resize_pred * 255).astype(np.uint8))
            # cv2.waitKey(0)

            prediction = 1.0*(resize_pred > 0.5)

            # IoU test
            label_name = input[1]
            idx = label_name.find(input_image_name)
            if idx is not -1:
                test_label_ = cv2.imread(label_name, 0).astype(np.float32)
                if test_label_ is None:
                    print("empty test label.")
                    break
                # test_label = np.asarray([test_label_], dtype=np.float32)
                # test_label = test_label_/255.0
                test_label = 1.0 - test_label_
                # cv2.imshow('what', test_label_)
                # cv2.waitKey(0)
                IoU = np.append(IoU, np.sum(test_label * prediction) / (np.sum(test_label) + np.sum((1.0 - test_label) * prediction)))
                print('%s IoU: %f' % (input_image_name, IoU[-1]))
                # recall = np.sum(np.dot(test_label, prediction)) / np.sum(test_label)
                # precision = np.sum(np.dot(test_label, prediction)) / np.sum(prediction)

            else:
                print('error: not corresponding labels.')
                break

            cnt += 1

            
            if args.out_dir is not None:
                # Colourful display
                color_segment = np.resize(resize_pred, (3, resize_pred.shape[0], resize_pred.shape[1]))
                visualize = color_segment * np.array([[[255]], [[204]], [[102]]])
                visualize = visualize.transpose(1, 2, 0).astype(np.uint8)

                # Merge result
                alpha = 0.5
                foreground = np.where(resize_pred != 0)
                merge = img
                merge[foreground] = img[foreground] * (1 - alpha) + alpha * visualize[foreground]

                # Origin image segment display
                # seg_img = pred.reshape(pred.shape[0], pred.shape[1], 1) * resize_img
                # seg_img = cv2.resize(seg_img, (img.shape[1], img.shape[0]))

                out_path_im = args.out_dir + input_image_name + '.' + input_path_ext
                out_path_gt = args.out_dir + input_image_name + '.' + input_path_ext

                cv2.imwrite(out_path_im, merge)
                # cv2.imwrite(out_path_gt, prediction) #  label images, where each pixel has an ID that represents the class

        print('mIoU: %f' % IoU.mean())
        print('stdIoU: %f' % IoU.std())

    except Exception, error:
        print error.message

    input_files.close()






