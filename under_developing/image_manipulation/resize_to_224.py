import numpy as np
import cv2
import os
from image_fix_size import fix_resize
import multiprocessing

image_path='train2017'
label_path='train2017_large_human_segment'

def process_thread(lb_file):
    if '.png' in lb_file:
        lb_img = cv2.imread(label_path + '/' + lb_file, cv2.IMREAD_GRAYSCALE)
        if lb_img is None:
            return

    img_file = lb_file[0:12] + '.jpg'
    img = cv2.imread(image_path + '/' + img_file)
    if img is None:
        return

    lb_img_resize = fix_resize(lb_img, 224, 224, method='nearest', pad_value=1)
    img_resize = fix_resize(img, 224, 224)
    
    cv2.imwrite(label_path+'_resize/'+lb_file, lb_img_resize)
    cv2.imwrite(image_path+'_resize/'+img_file, img_resize)



if os.path.exists(image_path) and  os.path.exists(label_path):
    pool = multiprocessing.Pool(8)
    lb_files = os.listdir(label_path)
    pool.map(process_thread, lb_files)
    pool.close()
    pool.join()
        



