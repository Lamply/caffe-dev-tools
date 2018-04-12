import numpy as np
import cv2
import os

image_path='/data/dataset/PASCAL_VOC_2012/Merge_Coarse_Fine_Image/'
label_path='result/c/'

if os.path.exists(image_path) and  os.path.exists(label_path):
    lb_files = os.listdir(label_path)
    for lb_file in lb_files:
        if '.png' in lb_file:
            lb_img = cv2.imread(label_path + lb_file)
            if lb_img is None:
                continue

        img_file = lb_file[0:11] + '.jpg'
        img = cv2.imread(image_path + img_file)
        if img is None:
            continue
        img = cv2.resize(img, (lb_img.shape[1], lb_img.shape[0]))
        
        print(lb_file)

        cv2.imshow('Label', lb_img)
        cv2.imshow('Image', img)
        ckey = cv2.waitKey(0)
        
