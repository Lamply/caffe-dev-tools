"""
This is a caffe tools for absorbing batch normalization layer into convolution layer.
Modified from github project: TimoSaemann/ENet and FreeApe/VGG-or-MobileNet-SSD

Notice: 

   1. Absorbing pattern: Conv-BN-Scale

   2. BatchNorm and Scale are only support Inplace op temporary

   3. DepthwiseConvolution or ConvolutionDepthwise are both conv op, so also supported, but not support Deconvolution temporary

   4. Not another constraint, feel free to use :D  @Lamply
"""
import numpy as np
import sys
import os
caffe_root = '/home/sad/caffe/'            # Change this to your project's caffe path 
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import argparse

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file')
    parser.add_argument('--weights', type=str, required=False, help='.caffemodel file')
    parser.add_argument('--output', type=str, required=True, help='specify output dir and suffix to store output')
    parser.add_argument('--absorb_weights', help='set if caffemodel to absorb', action="store_true")

    return parser


def add_bias_to_conv(model):
    # load the prototxt file as a protobuf message
    with open(model) as n:
        str1 = n.read()
    msg2 = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg2)

    for i, l2 in enumerate(msg2.layer):
        if l2.type == "Convolution" or l2.type == "DepthwiseConvolution" or l2.type == "ConvolutionDepthwise":
            if i+1 < len(msg2.layer):
                if msg2.layer[i+1].type == "BatchNorm":
                    if l2.convolution_param.bias_term is False:
                        l2.convolution_param.bias_term = True
                        l2.convolution_param.bias_filler.type = 'constant'
                        l2.convolution_param.bias_filler.value = 0.0  # actually default value

    model_temp = "model_temp.prototxt"
    print("Saving temp model...")
    with open(model_temp, 'w') as m:
        m.write(text_format.MessageToString(msg2))

    return model_temp


def bn_absorber_prototxt(model):

    full_model = add_bias_to_conv(model)

    # load the prototxt file as a protobuf message
    with open(full_model) as k:
        str1 = k.read()
    msg1 = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg1)

    # search for bn and scale layer and remove them
    for i, l in enumerate(msg1.layer):
        if l.type == "BatchNorm":
            if msg1.layer[i - 1].type == 'Convolution' or msg1.layer[i - 1].type == 'DepthwiseConvolution' or msg1.layer[i - 1].type == "ConvolutionDepthwise":
                print("remove layer %s..." % l.name)
                msg1.layer.remove(l)
                if msg1.layer[i].type == "Scale":
                    print("remove layer %s..." % msg1.layer[i].name)
                    msg1.layer.remove(msg1.layer[i])
                # msg1.layer[i].bottom.append(msg1.layer[i-1].top[0])

    os.remove(full_model)

    return msg1

def bn_absorber_caffemodel(ori_model, ori_weights, merge_model):
    '''
    merge the batchnorm, scale layer weights to the conv layer, to improve the performance
    var = var + scaleFacotr
    rstd = 1. / sqrt(var + eps)
    w = w * rstd * scale
    b = (b - mean) * rstd * scale + shift
    '''

    net = caffe.Net(ori_model, ori_weights, caffe.TEST)
    net_merge = caffe.Net(merge_model, caffe.TEST)

    for l3 in net_merge.params.keys():
        tmp_min_ = len(net_merge.params[l3])
        if len(net.params[l3]) < len(net_merge.params[l3]):
            tmp_min_ = len(net.params[l3])
        for i in range(tmp_min_):
            net_merge.params[l3][i].data[:] = net.params[l3][i].data[:]
    
    with open(ori_model) as n:
        str1 = n.read()
    msg2 = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg2)

    for i, l2 in enumerate(msg2.layer):
        if l2.type == "Convolution" or l2.type == "DepthwiseConvolution" or l2.type == "ConvolutionDepthwise":
            if i+1 < len(msg2.layer):
                if msg2.layer[i+1].type == "BatchNorm":
                    key = l2.name
                    print("absorbing %s ..." % key)
                    conv = net.params[key]
                    bn = net.params[msg2.layer[i+1].name]
                    scale = net.params[msg2.layer[i+2].name]
                    wt = conv[0].data
                    channels = wt.shape[0]
                    bias = np.zeros(wt.shape[0])
                    if len(conv) > 1:
                        bias = conv[1].data
                    mean = bn[0].data
                    var = bn[1].data
                    scalef = bn[2].data

                    scales = scale[0].data
                    shift = scale[1].data

                    if scalef != 0:
                        scalef = 1. / scalef
                    mean = mean * scalef
                    var = var * scalef
                    rstd = 1. / np.sqrt(var + 1e-5)         # This 1e-5 is specify in caffe.proto BatchNormParameter.eps
                    rstd1 = rstd.reshape((channels, 1, 1, 1))
                    scales1 = scales.reshape((channels, 1, 1, 1))
                    wt = wt * rstd1 * scales1
                    bias = (bias - mean) * rstd * scales + shift

                    net_merge.params[key][0].data[...] = wt
                    net_merge.params[key][1].data[...] = bias

    return net_merge


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()

    train_model = args.model
    train_weights = args.weights
    save_model = args.output + '_merge_bn.prototxt'
    save_weights = args.output + '_merge_bn.caffemodel'

    caffe.set_mode_cpu()
    model_merge = bn_absorber_prototxt(train_model)

    # save prototxt for inference
    print("Saving inference prototxt file...")
    path = os.path.join(save_model)
    with open(path, 'w') as m:
        m.write(text_format.MessageToString(model_merge))

    if args.absorb_weights:
        net_merge = bn_absorber_caffemodel(train_model, train_weights, save_model)

        # save weights
        print("Saving new weights...")
        net_merge.save(os.path.join(save_weights))

    print("Done!")

