import numpy as np
import cv2
import os
from image_fix_size import fix_resize
import multiprocessing

src_file = 'video_portrait_train.txt'

def process_thread(images):
    images = images.strip()
    for image in images.split(' '):
        image_path_ext = image.split(".")[-1]
        image_name = image.split("/")[-1:][0].replace('.' + image_path_ext, '')

        if 'png' in image_path_ext:
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(image)

        if img is None:
            continue

        if 'png' in image_path_ext:
            img = 1 - 1 * (img > 0)
            img = fix_resize(img, 224, 224, method='nearest', flag='pad', pad_value=1)
        else:
            img = fix_resize(img, 224, 224, method='bilinear', flag='pad')

        out_name = image.replace('/video/', '/video/processed/')
        dir_name = out_name.replace(out_name.split('/')[-1], '')
        if os.path.exists(dir_name) is False:
            os.mkdir(dir_name)
        print("writed in %s" % (image))
        cv2.imwrite(out_name, img)


pool = multiprocessing.Pool(8)

images_name = open(src_file)
pool.map(process_thread, images_name)
pool.close()
pool.join()
    

