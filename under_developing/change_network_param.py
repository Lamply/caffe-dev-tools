'''
This is a caffe tools for change some type of layers' param simply. 

Notice: 

   1. This tool is under develop.
'''
import numpy as np
import sys,os
caffe_root = '/home/sad/caffe/'     # Change this to your project's caffe path 
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


# Add bias to conv for BatchNorm absorb
def add_bias_to_conv(model, output):
    with open(model) as n:
        str1 = n.read()
    msg = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg)

    for i, l in enumerate(msg.layer):
        if l.type == "Convolution" or l.type == "DepthwiseConvolution" or l.type == "ConvolutionDepthwise":
            if msg.layer[i+1].type == "BatchNorm":
                if l.convolution_param.bias_term is False:
                    l.convolution_param.bias_term = True
                    l.convolution_param.bias_filler.type = 'constant'
                    l.convolution_param.bias_filler.value = 0.0  # actually default value

    print "Saving temp model..."
    with open(output, 'w') as m:
        m.write(text_format.MessageToString(msg))

    return output


# Change a model filter's number by a scale factor,  num_output = num_output * factor
def change_filters_num(model, factor, output):
    with open(model) as n:
        str1 = n.read()
    msg = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg)

    for i, l in enumerate(msg.layer):
        if l.type == "Convolution" or l.type == "DepthwiseConvolution" or l.type == "ConvolutionDepthwise":
            if i > 14:
                l.convolution_param.num_output = (int)(l.convolution_param.num_output * factor)
            # if l.convolution_param.group == 3:
            #     l.convolution_param.group = 8

    print("Saving output model %s..." % output)
    with open(output, 'w') as m:
        m.write(text_format.MessageToString(msg))

    return output


# Change a deploy net to lr_mult = 0 for finetuning
def change_lr_mult(model, output):
    with open(model) as n:
        str1 = n.read()
    msg = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg)

    for i, l in enumerate(msg.layer):
        if l.type == "Convolution" or l.type == "DepthwiseConvolution" or l.type == "ConvolutionDepthwise":
            if l.param._values == []:
                lr_param = caffe_pb2.ParamSpec()
                lr_param.lr_mult = 0
                l.param._values.append(lr_param)
                if l.convolution_param.bias_term is True:
                    lr_param = caffe_pb2.ParamSpec()
                    lr_param.lr_mult = 0
                    l.param._values.append(lr_param)
            # You should ensure param num and blobs num is equal
            else:
                for num in range(len(l.param)):
                    l.param[num].lr_mult = 0

    print("Saving output model %s..." % output)
    with open(output, 'w') as m:
        m.write(text_format.MessageToString(msg))

    return output


# Change a deploy net to lr_mult = 0 for finetuning
def change_layers_name(model, output):
    with open(model) as n:
        str1 = n.read()
    msg = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg)

    for i, l in enumerate(msg.layer):
        l.name = 's2_' + l.name
        for idx in range(len(msg.layer[i].top)):
            msg.layer[i].top[idx] = 's2_' + msg.layer[i].top[idx]
        for idx in range(len(msg.layer[i].bottom)):
            msg.layer[i].bottom[idx] = 's2_' + msg.layer[i].bottom[idx]

    print("Saving output model %s..." % output)
    with open(output, 'w') as m:
        m.write(text_format.MessageToString(msg))

    return output


if __name__ == '__main__':
    length = len(sys.argv)
    if length != 3:
        print('require 2 input')
    else:
        network = sys.argv[1]
        output = sys.argv[2]
        change_layers_name(network, output)


