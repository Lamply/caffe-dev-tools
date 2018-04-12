import sys
# caffe_root = '/home/sad/ENet/caffe-enet/'  # Change this to the absolute directory to ENet Caffe
caffe_python_root = '/home/sad/ENet/caffe-enet/python'  # Change this to the absolute directory to ENet Caffe
sys.path.insert(0, caffe_python_root)
# sys.path.insert(0, caffe_python_root + 'caffe')
import caffe
import numpy as np
import cv2
import argparse
from argparse import ArgumentParser
from image_fix_size import fix_resize


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file')
    parser.add_argument('--use_mean', help='implement with mean subtraction', action="store_true")
    parser.add_argument('--use_scale', help='implement with value scale', action="store_true")
    return parser

if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(args.model, args.weights, caffe.TEST)

    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['label'].data.shape

    vidIn = cv2.VideoCapture(0)
    # vidIn.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    # vidIn.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    if args.use_mean:
        print('use mean substraction')

    if args.use_scale:
        print('use scale')

    print("Press space to detect the face, press escape to exit")

    while True:
        vis = vidIn.read()[1]
        # print(vis.shape)
        # if len(vis.shape) > 2:
        #     img = np.mean(vis, axis=2).astype(np.uint8)
        # else:
        #     img = vis.astype(np.uint8)
        
        # resize_vis = cv2.resize(vis, (vis.shape[1]*input_shape[2]/vis.shape[0], input_shape[2]))
        # np_img = np.asarray([resize_vis])
        # np_img = np.squeeze(np_img)
        # resize_img = np_img[:, np_img.shape[1]/2-input_shape[3]/2:np_img.shape[1]/2+input_shape[3]/2]
        # input_image = resize_img

        resize_vis = fix_resize(vis, input_shape[2], input_shape[3])
        resize_img = np.squeeze(np.asarray([resize_vis]))
        input_image = resize_img

        if args.use_mean:
            input_image = input_image - [103.939, 116.779, 123.68]

        if args.use_scale:
            input_image = input_image * 0.017

        # print(input_image.shape)
        input_nn = input_image.transpose((2, 0, 1))

        net.blobs['data'].data[...] = input_nn
        net.forward()
        pred = net.blobs['label'].data
        # prediction = net.blobs['deconv6_0_0_2'].data[0].argmax(axis=0)

        # Origin image segment display
        pred = np.squeeze(pred)[0]
        
        seg_img = pred.reshape(pred.shape[0], pred.shape[1], 1) * resize_img
        seg_img = seg_img.astype(np.uint8)

        # prediction = np.squeeze(prediction)
        # prediction = np.resize(prediction, (3, input_shape[2], input_shape[3]))
        # prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

        # prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
        # label_colours_bgr = label_colours[..., ::-1]
        # cv2.LUT(prediction, label_colours_bgr, prediction_rgb)

        cv2.imshow("image", seg_img)
        # cv2.imshow("image", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vidIn.release()
    cv2.destroyAllWindows()
    
