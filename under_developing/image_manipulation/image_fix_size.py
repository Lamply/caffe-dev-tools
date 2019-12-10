# -*- coding:utf-8 -*-
"""
This is a resize function that fix on size or scale.

Notice:

   1. Recover from fix_resize() is needed. (NO IMPLEMENT)

"""
import numpy as np
import cv2


def scale_resize(src_img, scale=1.0, align_length=0, align_flag='height', interp=cv2.INTER_LINEAR):
    """
    Scale resize input image.
    :param src_img: Input image, cv2 numpy array, in [H, W, C] or [H, W] format
    :param scale: Scale ratio, default 1.0
    :param align_length: Align length, auto infer scale param, default 0
    :param align_flag: 'height' or 'width' to align, default 'height'
    :param interp: interpolation method, default bilinear
    :return: Resized image
    """
    if scale == 1.0 and align_length == 0:
        return src_img
    elif align_length > 0:
        if align_flag == 'height':
            resize_img = cv2.resize(src_img, (int(src_img.shape[1] * align_length / src_img.shape[0]), align_length),
                                    interpolation=interp)
        elif align_flag == 'width':
            resize_img = cv2.resize(src_img, (align_length, int(src_img.shape[0] * align_length / src_img.shape[1])),
                                    interpolation=interp)
        else:
            print('scale_resize(): unkown align_flag %s' % align_flag)
            return
    elif scale != 1.0:
        resize_img = cv2.resize(src_img, (int(src_img.shape[1] * scale), int(src_img.shape[0] * scale)), interpolation=interp)
    else:
        print('scale_resize(): error input align_length or scale')
        return src_img

    return resize_img


def fix_resize(src_img, dst_h, dst_w, method='bilinear', flag='pad', pad_value=0):
    """
    Fix source image size to a fixed size.
    :param src_img: Input image, cv2 numpy array, in [H, W, C] or [H, W] format
    :param dst_h: Fixed height
    :param dst_w: Fixed width
    :param method: Interpolation method: 'bilinear' or 'nearest', default 'bilinear'
    :param flag: Resize flag: 'pad' or 'crop', default 'pad'
                              pad: Resize according image's long size and pad 0 to center fix short size
                              crop: Resize according image's short size and crop center to fix long size
    :param pad_value: In 'pad' flag, padding with pad_value
    :return: Fixed-size image and align flag (0: H,  1: W)
    """
    if len(src_img.shape) != 3 and len(src_img.shape) != 2:
        print("fix_to_image_size(): error input image array %d." % len(src_img.shape))
        return

    if src_img.shape[0] < 2 or src_img.shape[1] < 2 or dst_h < 2 or dst_w < 2:
        print('fix_to_image_size(): input shape too small.')
        return

    if method == 'bilinear':
        interp = cv2.INTER_LINEAR
    elif method == 'nearest':
        interp = cv2.INTER_NEAREST
    else:
        print("fix_to_image_size(): error input method.")
        return

    src_h = src_img.shape[0]
    src_w = src_img.shape[1]
    src_ratio = np.float32(src_h)/np.float32(src_w)
    dst_ratio = np.float32(dst_h)/np.float32(dst_w)

    if src_ratio > dst_ratio:
        if flag == 'crop':
            align_flag = 1
            resize_img = scale_resize(src_img, align_length=dst_w, align_flag='width', interp=interp)
            crop_start = int((resize_img.shape[0] - dst_h) / 2)
            if len(resize_img.shape) == 3:
                fixed_img = resize_img[crop_start:crop_start + dst_h, :, :]
            else:
                fixed_img = resize_img[crop_start:crop_start + dst_h, :]

        elif flag == 'pad':
            align_flag = 0
            resize_img = scale_resize(src_img, align_length=dst_h, align_flag='height', interp=interp)
            pad_num = int((dst_w - resize_img.shape[1]) / 2)
            fixed_img = cv2.copyMakeBorder(resize_img, 0, 0, pad_num, dst_w - resize_img.shape[1] - pad_num,
                                           cv2.BORDER_CONSTANT, value=pad_value)
        else:
            print("fix_to_image_size(): error input flag.")
            return
    else:
        if flag == 'crop':
            align_flag = 0
            resize_img = scale_resize(src_img, align_length=dst_h, align_flag='height', interp=interp)
            crop_start = int((resize_img.shape[1] - dst_w) / 2)
            if len(resize_img.shape) == 3:
                fixed_img = resize_img[:, crop_start:crop_start + dst_w, :]
            else:
                fixed_img = resize_img[:, crop_start:crop_start + dst_w]

        elif flag == 'pad':
            align_flag = 1
            resize_img = scale_resize(src_img, align_length=dst_w, align_flag='width', interp=interp)
            pad_num = int((dst_h - resize_img.shape[0]) / 2)
            fixed_img = cv2.copyMakeBorder(resize_img, pad_num, dst_h - resize_img.shape[0] - pad_num, 0, 0,
                                           cv2.BORDER_CONSTANT, value=pad_value)
        else:
            print("fix_to_image_size(): error input flag.")
            return

    # print(fixed_img.shape)
    return fixed_img

