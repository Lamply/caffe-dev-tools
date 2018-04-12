"""
    This is an python script for generate caffe model, originally obtain from __https://github.com/wenwei202/caffe__

    Further improve by Lamplykyz@gmail.com

    Notice:

        1. This code is just generating a simple and raw but general network, often need to modify for specify usage

        2. ReLU/BN is default only support inplace op
"""
import sys
caffe_root = '/home/sad/caffe/'  # Change this to your project's caffe path 
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
import argparse
import os


class CaffeProtoParser:
    def __init__(self,filepath):
        self.filepath = filepath

    def readProtoSolverFile(self):
        solver_config = caffe.proto.caffe_pb2.SolverParameter()
        #TODO how to read proto file?
        return self._readProtoTxtFile(self.filepath, solver_config)
    #enddef

    def readProtoNetFile(self):
        net_config = caffe.proto.caffe_pb2.NetParameter()
        #TODO how to read proto file?
        return self._readProtoTxtFile(self.filepath, net_config)
    #enddef

    def readBlobProto(self):
        blob = caffe.proto.caffe_pb2.BlobProto()
        #TODO how to read proto file?
        return self._readProtoBinFile(self.filepath, blob)
    #enddef

    def getLayerByName(self,net_msg,layername):
        res = []
        for cur_layer in net_msg.layer:
            if layername==cur_layer.name:
                res = cur_layer
                break
        return res
    #enddef

    def _readProtoTxtFile(self, filepath, parser_object):
        file = open(filepath, "r")
        if not file:
            raise self.ProcessException("ERROR (" + filepath + ")!")
        text_format.Merge(str(file.read()), parser_object)
        file.close()
        return parser_object

    def _readProtoBinFile(self, filepath, parser_object):
        file = open(filepath, "rb")
        if not file:
            raise self.ProcessException("ERROR (" + filepath + ")!")
        parser_object.ParseFromString(file.read())
        file.close()
        return parser_object


'''
    Common layer define
'''
def add_conv_layer(net_msg, name, bottom, num_output, pad, kernel_size, stride=1, bias_term=True, group=1, dilation=1):
    conv_layer = net_msg.layer.add()
    conv_layer.name = name
    conv_layer.type = 'Convolution'
    conv_layer.bottom._values.append(bottom)
    conv_layer.top._values.append(name)

    # param info for weight and bias
    # lr_param = caffe_pb2.ParamSpec()
    # lr_param.lr_mult = 1
    # conv_layer.param._values.append(lr_param)
    # if bias_term:
    #     lr_param = caffe_pb2.ParamSpec()
    #     lr_param.lr_mult = 2
    #     conv_layer.param._values.append(lr_param)

    # conv parameters
    conv_layer.convolution_param.num_output = num_output
    conv_layer.convolution_param.pad._values.append(pad)
    conv_layer.convolution_param.kernel_size._values.append(kernel_size)
    if group != 1:
        conv_layer.convolution_param.group = group
    conv_layer.convolution_param.stride._values.append(stride)
    conv_layer.convolution_param.weight_filler.type = 'msra'
    conv_layer.convolution_param.bias_term = bias_term
    if bias_term:
        conv_layer.convolution_param.bias_filler.type = 'constant'
    if dilation != 1:
        conv_layer.convolution_param.dilation._values.append(dilation)
    return name


def add_depthwise_conv_layer(net_msg, name, bottom, num_output, pad, kernel_size, stride=1, bias_term=True, dilation=1):
    conv_layer = net_msg.layer.add()
    conv_layer.name = name
    conv_layer.type = 'ConvolutionDepthwise'
    conv_layer.bottom._values.append(bottom)
    conv_layer.top._values.append(name)

    # param info for weight and bias
    # lr_param = caffe_pb2.ParamSpec()
    # lr_param.lr_mult = 1
    # conv_layer.param._values.append(lr_param)
    # if bias_term:
    #     lr_param = caffe_pb2.ParamSpec()
    #     lr_param.lr_mult = 2
    #     conv_layer.param._values.append(lr_param)

    # conv parameters
    conv_layer.convolution_param.num_output = num_output
    conv_layer.convolution_param.pad._values.append(pad)
    conv_layer.convolution_param.kernel_size._values.append(kernel_size)
    conv_layer.convolution_param.stride._values.append(stride)
    conv_layer.convolution_param.weight_filler.type = 'msra'
    conv_layer.convolution_param.bias_term = bias_term
    if bias_term:
        conv_layer.convolution_param.bias_filler.type = 'constant'
    if dilation != 1:
        conv_layer.convolution_param.dilation._values.append(dilation)
    return name


def add_deconv_layer(net_msg, name, bottom, num_output, pad, kernel_size, stride, bias_term=True, group=1):
    conv_layer = net_msg.layer.add()
    conv_layer.name = name
    conv_layer.type = 'Deconvolution'
    conv_layer.bottom._values.append(bottom)
    conv_layer.top._values.append(name)

    # param info for weight and bias
    # lr_param = caffe_pb2.ParamSpec()
    # lr_param.lr_mult = 1
    # conv_layer.param._values.append(lr_param)
    # if bias_term:
    #     lr_param = caffe_pb2.ParamSpec()
    #     lr_param.lr_mult = 2
    #     conv_layer.param._values.append(lr_param)

    # conv parameters
    conv_layer.convolution_param.num_output = num_output
    conv_layer.convolution_param.pad._values.append(pad)
    conv_layer.convolution_param.kernel_size._values.append(kernel_size)
    if group != 1:
        conv_layer.convolution_param.group = group
    conv_layer.convolution_param.stride._values.append(stride)
    conv_layer.convolution_param.weight_filler.type = 'msra'
    conv_layer.convolution_param.bias_term = bias_term
    if bias_term:
        conv_layer.convolution_param.bias_filler.type = 'constant'
    return name


def add_relu_layer(net_msg, bottom):
    relulayer = net_msg.layer.add()
    relulayer.name = bottom+'_relu'
    relulayer.type = 'ReLU'
    relulayer.bottom._values.append(bottom)
    relulayer.top._values.append(bottom)
    return bottom


def add_eltwise_add_layer(net_msg, name, bottom1, bottom2):
    eltlayer = net_msg.layer.add()
    eltlayer.name = name
    eltlayer.type = 'Eltwise'
    eltlayer.bottom._values.append(bottom1)
    eltlayer.bottom._values.append(bottom2)
    eltlayer.top._values.append(name)
    return name


def add_concat_layer(net_msg, name, bottom1, bottom2, bottom3='', bottom4='', bottom5=''):
    concat_layer = net_msg.layer.add()
    concat_layer.name = name
    concat_layer.type = 'Concat'
    concat_layer.bottom._values.append(bottom1)
    concat_layer.bottom._values.append(bottom2)
    if bottom3 != '':
        concat_layer.bottom._values.append(bottom3)
    if bottom4 != '':
        concat_layer.bottom._values.append(bottom4)
    if bottom5 != '':
        concat_layer.bottom._values.append(bottom5)
    concat_layer.top._values.append(name)
    return name


def add_BN_layer(net_msg, bottom):
    # norm layer
    batchnormlayer = net_msg.layer.add()
    batchnormlayer.name = bottom+'_bn'
    batchnormlayer.type = 'BatchNorm'
    batchnormlayer.bottom._values.append(bottom)
    batchnormlayer.top._values.append(bottom)
    # compatible with old caffe version
    for i in range(0,3):
        lr_param = caffe_pb2.ParamSpec()
        lr_param.lr_mult = 0
        lr_param.decay_mult = 0
        batchnormlayer.param._values.append(lr_param)

    # scale layer
    scalelayer = net_msg.layer.add()
    scalelayer.name = bottom+'_scale'
    scalelayer.type = 'Scale'
    scalelayer.bottom._values.append(bottom)
    scalelayer.top._values.append(bottom)

    # optional: modify lr mult
    # lr_param = caffe_pb2.ParamSpec()
    # lr_param.lr_mult = 1
    # scalelayer.param._values.append(lr_param)
    # lr_param = caffe_pb2.ParamSpec()
    # lr_param.lr_mult = 2
    # lr_param.decay_mult = 0
    # scalelayer.param._values.append(lr_param)

    scalelayer.scale_param.bias_term = True
    scalelayer.scale_param.filler.value = 1
    scalelayer.scale_param.bias_filler.value = 0
    return bottom


def add_global_avg_pooling_layer(net_msg,name,bottom):
    glb_avg_pl_layer = net_msg.layer.add()
    glb_avg_pl_layer.name = name
    glb_avg_pl_layer.type = 'Pooling'
    glb_avg_pl_layer.bottom._values.append(bottom)
    glb_avg_pl_layer.top._values.append(name)
    glb_avg_pl_layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
    glb_avg_pl_layer.pooling_param.global_pooling = True
    return name


def add_pooling_layer(net_msg, name, bottom, pooling_type, kernel_size, stride):
    pooling_layer = net_msg.layer.add()
    pooling_layer.name = name
    pooling_layer.type = 'Pooling'
    pooling_layer.bottom._values.append(bottom)
    pooling_layer.top._values.append(name)
    if pooling_type == 'MAX':
        pooling_layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
    elif pooling_type == 'AVE':
        pooling_layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
    pooling_layer.pooling_param.kernel_size = kernel_size
    pooling_layer.pooling_param.stride = stride
    return name


def add_ip_layer(net_msg,name,bottom,num):
    ip_layer = net_msg.layer.add()
    ip_layer.name = name
    ip_layer.type = 'InnerProduct'
    ip_layer.bottom._values.append(bottom)
    ip_layer.top._values.append(name)
    # param info for weight and bias
    lr_param = caffe_pb2.ParamSpec()
    lr_param.lr_mult = 1
    lr_param.decay_mult = 1
    ip_layer.param._values.append(lr_param)
    lr_param = caffe_pb2.ParamSpec()
    lr_param.lr_mult = 2
    lr_param.decay_mult = 0
    ip_layer.param._values.append(lr_param)
    # inner product parameters
    ip_layer.inner_product_param.num_output = num
    ip_layer.inner_product_param.weight_filler.type = 'msra'
    ip_layer.inner_product_param.bias_filler.type = 'constant'
    ip_layer.inner_product_param.bias_filler.value = 0.0
    return name


def add_accuracy_layer(net_msg,bottom):
    accuracy_layer = net_msg.layer.add()
    accuracy_layer.name = 'accuracy'
    accuracy_layer.type = 'Accuracy'
    accuracy_layer.bottom._values.append(bottom)
    accuracy_layer.bottom._values.append('label')
    accuracy_layer.top._values.append('accuracy')
    include_param = caffe_pb2.NetStateRule()
    include_param.phase = caffe_pb2.TEST
    accuracy_layer.include._values.append(include_param)


def add_loss_layer(net_msg,bottom):
    loss_layer = net_msg.layer.add()
    loss_layer.name = 'loss'
    loss_layer.type = 'SoftmaxWithLoss'
    loss_layer.bottom._values.append(bottom)
    loss_layer.bottom._values.append('label')
    loss_layer.top._values.append('loss')


def add_interp_layer(net_msg, name, bottom, height=0, width=0, zoom_factor=0, shrink_factor=0):
    interp_layer = net_msg.layer.add()
    interp_layer.name = name
    interp_layer.type = 'Interp'
    interp_layer.bottom._values.append(bottom)
    interp_layer.top._values.append(name)
    if height == 0 or width == 0:
        if zoom_factor != 0:
            interp_layer.interp_param.zoom_factor = zoom_factor
        elif shrink_factor != 0:
            interp_layer.interp_param.shrink_factor = shrink_factor
        else:
            print("error: no param set in %s" % name)
    else:
        interp_layer.interp_param.height = height
        interp_layer.interp_param.width = width
    return name


def add_shufflechannel_layer(net_msg, name, bottom, group):
    shufflechannel_layer = net_msg.layer.add()
    shufflechannel_layer.name = name
    shufflechannel_layer.type = 'ShuffleChannel'
    shufflechannel_layer.bottom._values.append(bottom)
    shufflechannel_layer.top._values.append(name)
    shufflechannel_layer.shuffle_channel_param.group = group
    return name


'''

    Here is block structure for popular architecture
    
'''
# common conv block
def add_conv_bn_block(net_msg, name, bottom, num_output, pad=1, kernel_size=3, stride=1, group=1, dilation=1, relu=True):
    padding = 1
    if pad == 'SAME':
        padding = dilation*(kernel_size-1)/2
    else:
        padding = pad
    add_conv_layer(net_msg, name=name, bottom=bottom, num_output=num_output, pad=padding,
                   kernel_size=kernel_size, stride=stride, bias_term=False, group=group, dilation=dilation)
    add_BN_layer(net_msg, bottom=name)
    if relu is True:
        add_relu_layer(net_msg, bottom=name)
    return name

# common conv block
def add_depthwise_conv_bn_block(net_msg, name, bottom, num_output, pad=1, kernel_size=3, stride=1, dilation=1, relu=True):
    padding = 1
    if pad == 'SAME':
        padding = dilation * (kernel_size - 1) / 2
    else:
        padding = pad
    add_depthwise_conv_layer(net_msg, name=name, bottom=bottom, num_output=num_output, pad=padding,
                             kernel_size=kernel_size, stride=stride, bias_term=False, dilation=dilation)
    add_BN_layer(net_msg, bottom=name)
    if relu is True:
        add_relu_layer(net_msg, bottom=name)
    return name


# ShuffleNet
def add_shufflenet_block(net_msg, name, bottom, num_output, group, dilation=1):
    if num_output % 4 != 0:
        print("error occur when generate filter num in %s." % name)

    # reduce dims
    add_conv_layer(net_msg, name=name + '_conv1', bottom=bottom,
                   num_output=num_output/4, pad=0, kernel_size=1, stride=1, bias_term=False, group=group)
    add_BN_layer(net_msg, bottom=name + '_conv1')
    add_relu_layer(net_msg, bottom=name + '_conv1')

    # shuffle channel
    add_shufflechannel_layer(net_msg, name=name+'_shuffle', bottom=name+'_conv1', group=group)

    # bottleneck
    add_depthwise_conv_layer(net_msg, name=name + '_conv2', bottom=name + '_shuffle',
                             num_output=num_output/4, pad=dilation, kernel_size=3, stride=1, bias_term=False, dilation=dilation)
    add_BN_layer(net_msg, bottom=name + '_conv2')

    # restore dims
    add_conv_layer(net_msg, name=name + '_conv3', bottom=name+'_conv2',
                   num_output=num_output, pad=0, kernel_size=1, stride=1, bias_term=False, group=group)
    add_BN_layer(net_msg, bottom=name + '_conv3')

    # eltwise
    add_eltwise_add_layer(net_msg, name + '_eltwise', bottom, name + '_conv3')
    add_relu_layer(net_msg, bottom=name+'_eltwise')

    return name + '_eltwise'


def add_shufflenet_block_stride(net_msg, name, bottom, input_channel, num_output, group, dilation=1):
    if dilation == 1:
        # ave pooling connect
        add_pooling_layer(net_msg, name=name + '_' + bottom + '_pool', bottom=bottom, pooling_type='AVE', kernel_size=3, stride=2)

    if (num_output - input_channel) % 4 != 0:
        print("error occur when generate filter num in %s." % name)
    conv_num = (num_output - input_channel) / 4

    # reduce dims
    add_conv_layer(net_msg, name=name + '_conv1', bottom=bottom,
                   num_output=conv_num, pad=0, kernel_size=1, stride=1, bias_term=False, group=group)
    add_BN_layer(net_msg, bottom=name + '_conv1')
    add_relu_layer(net_msg, bottom=name + '_conv1')

    # shuffle channel
    add_shufflechannel_layer(net_msg, name=name+'_shuffle', bottom=name+'_conv1', group=group)

    # bottleneck
    if dilation == 1:
        add_depthwise_conv_layer(net_msg, name=name + '_conv2', bottom=name + '_shuffle',
                                 num_output=conv_num, pad=dilation, kernel_size=3, stride=2, bias_term=False, dilation=dilation)
    else:
        add_depthwise_conv_layer(net_msg, name=name + '_conv2', bottom=name + '_shuffle',
                                 num_output=conv_num, pad=dilation, kernel_size=3, stride=1, bias_term=False, dilation=dilation)
    add_BN_layer(net_msg, bottom=name + '_conv2')

    # restore dims
    add_conv_layer(net_msg, name=name + '_conv3', bottom=name+'_conv2',
                   num_output=conv_num*4, pad=0, kernel_size=1, stride=1, bias_term=False, group=group)
    add_BN_layer(net_msg, bottom=name + '_conv3')

    # concat
    if dilation == 1:
        add_concat_layer(net_msg, name + '_concat', name + '_' + bottom + '_pool', name + '_conv3')
    else:
        add_concat_layer(net_msg, name + '_concat', bottom, name + '_conv3')
    add_relu_layer(net_msg, bottom=name+'_concat')

    return name + '_concat'


def add_ASPP_module(net_msg, name, bottom, height, width):
    c1 = add_conv_bn_block(net_msg, name=name+'_conv1x1', bottom=bottom, num_output=256, pad=0, kernel_size=1, stride=1, relu=False)
    c3_6 = add_conv_bn_block(net_msg, name=name+'_conv3x3_r6', bottom=bottom, num_output=256, pad=3, kernel_size=3, dilation=3, relu=False)
    c3_12 = add_conv_bn_block(net_msg, name=name+'_conv3x3_r12', bottom=bottom, num_output=256, pad=6, kernel_size=3, dilation=6, relu=False)
    c3_18 = add_conv_bn_block(net_msg, name=name+'_conv3x3_r18', bottom=bottom, num_output=256, pad=9, kernel_size=3, dilation=9, relu=False)
    f_image = add_global_avg_pooling_layer(net_msg, name=name+'_ave_pool', bottom=bottom)
    f_image = add_conv_bn_block(net_msg, name=name+'_f_image_1x1', bottom=f_image, num_output=256, pad=0, kernel_size=1, stride=1, relu=False)
    f_image = add_interp_layer(net_msg, name=name+'_f_image', bottom=f_image, height=height, width=width)
    aspp_feature = add_concat_layer(net_msg, name=name+'_concat', bottom1=c1, bottom2=c3_6, bottom3=c3_12, bottom4=c3_18, bottom5=f_image)
    aspp_feature = add_conv_bn_block(net_msg, name=name+'_merge', bottom=aspp_feature, num_output=256, pad=0, kernel_size=1, stride=1)
    return aspp_feature


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_layer', type=str, required=True, help="template (data layers) to generate net")
    # args = parser.parse_args()
    data_layer = 'input_layer.prototxt'

    caffe.set_mode_cpu()
    net_parser = CaffeProtoParser(data_layer)
    net_msg = net_parser.readProtoNetFile()

    # Data layer size
    # channel = net_msg.layer[0].input_param.shape[0].dim[1]
    height = net_msg.layer[0].input_param.shape[0].dim[2]
    width = net_msg.layer[0].input_param.shape[0].dim[3]

    '''
        ShuffleNet 1x g3:
        You should delete 'resx1_conv1.group' and 'resx1_shuffle' by yourself
    '''
    # add_conv_bn_block(net_msg, name='conv1', bottom='data', num_output=24, pad=1, kernel_size=3, stride=2)
    # add_pooling_layer(net_msg, name='pool1', bottom='conv1', pooling_type='MAX', kernel_size=3, stride=2)
    #
    # top_tmp = add_shufflenet_block_stride(net_msg, name='resx1', bottom='pool1', input_channel=24, num_output=240,
    #                                       group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx2', bottom=top_tmp, num_output=240, group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx3', bottom=top_tmp, num_output=240, group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx4', bottom=top_tmp, num_output=240, group=3)
    #
    # top_tmp = add_shufflenet_block_stride(net_msg, name='resx5', bottom=top_tmp, input_channel=240, num_output=480,
    #                                       group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx6', bottom=top_tmp, num_output=480, group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx7', bottom=top_tmp, num_output=480, group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx8', bottom=top_tmp, num_output=480, group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx9', bottom=top_tmp, num_output=480, group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx10', bottom=top_tmp, num_output=480, group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx11', bottom=top_tmp, num_output=480, group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx12', bottom=top_tmp, num_output=480, group=3)
    #
    # top_tmp = add_shufflenet_block_stride(net_msg, name='resx13', bottom=top_tmp, input_channel=480, num_output=960,
    #                                       group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx14', bottom=top_tmp, num_output=960, group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx15', bottom=top_tmp, num_output=960, group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx16', bottom=top_tmp, num_output=960, group=3)
    #
    # add_global_avg_pooling_layer(net_msg, name='pool_ave', bottom=top_tmp)
    # add_conv_layer(net_msg, name='fc1000', bottom='pool_ave', num_output=1000, pad=0, kernel_size=1, stride=1)

    '''
        ShuffleNet 0.5x g8 ( Not the same as ShuffleNet paper ):
        You should delete 'resx1_conv1.group' and 'resx1_shuffle' by yourself
    '''
    # group_channel = [224, 416, 832]
    # add_conv_bn_block(net_msg, name='conv1', bottom='data', num_output=16, pad=1, kernel_size=3, stride=2)
    # add_pooling_layer(net_msg, name='pool1', bottom='conv1', pooling_type='MAX', kernel_size=3, stride=2)
    #
    # top_tmp = add_shufflenet_block_stride(net_msg, name='resx1', bottom='pool1', input_channel=16, num_output=group_channel[0], group=8)
    # top_tmp = add_shufflenet_block(net_msg, name='resx2', bottom=top_tmp, num_output=group_channel[0], group=8)
    # top_tmp = add_shufflenet_block(net_msg, name='resx3', bottom=top_tmp, num_output=group_channel[0], group=8)
    # top_tmp = add_shufflenet_block(net_msg, name='resx4', bottom=top_tmp, num_output=group_channel[0], group=8)
    #
    # top_tmp = add_shufflenet_block_stride(net_msg, name='resx5', bottom=top_tmp, input_channel=group_channel[0], num_output=group_channel[1], group=8)
    # top_tmp = add_shufflenet_block(net_msg, name='resx6', bottom=top_tmp, num_output=group_channel[1], group=8)
    # top_tmp = add_shufflenet_block(net_msg, name='resx7', bottom=top_tmp, num_output=group_channel[1], group=8)
    # top_tmp = add_shufflenet_block(net_msg, name='resx8', bottom=top_tmp, num_output=group_channel[1], group=8)
    # top_tmp = add_shufflenet_block(net_msg, name='resx9', bottom=top_tmp, num_output=group_channel[1], group=8)
    # top_tmp = add_shufflenet_block(net_msg, name='resx10', bottom=top_tmp, num_output=group_channel[1], group=8)
    #
    # top_tmp = add_shufflenet_block_stride(net_msg, name='resx11', bottom=top_tmp, input_channel=group_channel[1], num_output=group_channel[2], group=8)
    # top_tmp = add_shufflenet_block(net_msg, name='resx12', bottom=top_tmp, num_output=group_channel[2], group=8)
    # top_tmp = add_shufflenet_block(net_msg, name='resx13', bottom=top_tmp, num_output=group_channel[2], group=8)
    # top_tmp = add_shufflenet_block(net_msg, name='resx14', bottom=top_tmp, num_output=group_channel[2], group=8)
    #
    # add_global_avg_pooling_layer(net_msg, name='pool_ave', bottom=top_tmp)
    # add_conv_layer(net_msg, name='fc1000', bottom='pool_ave', num_output=1000, pad=0, kernel_size=1, stride=1)
    #
    # filepath = './shufflenet_generate.prototxt'
    # file = open(filepath, "w")
    # if not file:
    #     raise IOError("ERROR (" + filepath + ")!")
    # file.write(str(net_msg))
    # file.close()


    '''
        Test Net:
    '''
    # top_tmp = add_conv_bn_block(net_msg, name='conv1', bottom='data', num_output=24, pad=1, kernel_size=3)
    # top_tmp = add_conv_bn_block(net_msg, name='conv2', bottom=top_tmp, num_output=56, pad=1, kernel_size=3, group=8)
    # top_tmp = add_depthwise_conv_bn_block(net_msg, name='conv3', bottom=top_tmp, num_output=56, pad=1, kernel_size=3)
    # top_tmp = add_pooling_layer(net_msg, name='pool1', bottom=top_tmp, pooling_type='MAX', kernel_size=2, stride=2)
    # top_tmp = add_conv_bn_block(net_msg, name='conv4', bottom=top_tmp, num_output=128, pad=1, kernel_size=3, group=8)
    # top_tmp = add_shufflechannel_layer(net_msg, name='conv4_shuffle', bottom=top_tmp, group=8)
    # top_tmp = add_depthwise_conv_bn_block(net_msg, name='conv5', bottom=top_tmp, num_output=128, pad=1, kernel_size=3)
    # top_tmp = add_pooling_layer(net_msg, name='pool2', bottom=top_tmp, pooling_type='MAX', kernel_size=2, stride=2)
    # top_tmp = add_conv_bn_block(net_msg, name='conv6', bottom=top_tmp, num_output=256, pad=1, kernel_size=3, group=8)
    # top_tmp = add_shufflechannel_layer(net_msg, name='conv6_shuffle', bottom=top_tmp, group=8)
    # top_tmp = add_depthwise_conv_bn_block(net_msg, name='conv7', bottom=top_tmp, num_output=256, pad=1, kernel_size=3)
    # top_tmp = add_global_avg_pooling_layer(net_msg, name='pool3', bottom=top_tmp)
    # add_ip_layer(net_msg, name='ip2', bottom=top_tmp, num=10)

    '''
        ShuffleNet 1x g3 deeplab v3:
        You should delete 'resx1_conv1.group' and 'resx1_shuffle' by yourself
    '''
    # group_channel = [240, 480, 960]
    # add_conv_bn_block(net_msg, name='conv1', bottom='data', num_output=24, pad=1, kernel_size=3, stride=2)
    # add_pooling_layer(net_msg, name='pool1', bottom='conv1', pooling_type='MAX', kernel_size=3, stride=2)
    #
    # top_tmp = add_shufflenet_block_stride(net_msg, name='resx1', bottom='pool1', input_channel=24, num_output=240,
    #                                       group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx2', bottom=top_tmp, num_output=240, group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx3', bottom=top_tmp, num_output=240, group=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx4', bottom=top_tmp, num_output=240, group=3)
    #
    # # top_tmp = add_shufflenet_block_stride(net_msg, name='resx5', bottom=top_tmp, input_channel=240, num_output=480,
    # #                                       group=3)
    # top_tmp = add_shufflenet_block_stride(net_msg, name='resx5', bottom=top_tmp, input_channel=240, num_output=480, group=3, dilation=2)
    # top_tmp = add_shufflenet_block(net_msg, name='resx6', bottom=top_tmp, num_output=480, group=3, dilation=4)
    # top_tmp = add_shufflenet_block(net_msg, name='resx7', bottom=top_tmp, num_output=480, group=3, dilation=6)
    # top_tmp = add_shufflenet_block(net_msg, name='resx8', bottom=top_tmp, num_output=480, group=3, dilation=2)
    # top_tmp = add_shufflenet_block(net_msg, name='resx9', bottom=top_tmp, num_output=480, group=3, dilation=4)
    # top_tmp = add_shufflenet_block(net_msg, name='resx10', bottom=top_tmp, num_output=480, group=3, dilation=6)
    # top_tmp = add_shufflenet_block(net_msg, name='resx11', bottom=top_tmp, num_output=480, group=3, dilation=2)
    # top_tmp = add_shufflenet_block(net_msg, name='resx12', bottom=top_tmp, num_output=480, group=3, dilation=4)
    #
    # # top_tmp = add_shufflenet_block_stride(net_msg, name='resx13', bottom=top_tmp, input_channel=480, num_output=960,
    # #                                       group=3)
    # top_tmp = add_shufflenet_block_stride(net_msg, name='resx13', bottom=top_tmp, input_channel=480, num_output=960, group=3, dilation=8)
    # top_tmp = add_shufflenet_block(net_msg, name='resx14', bottom=top_tmp, num_output=960, group=3, dilation=12)
    # top_tmp = add_shufflenet_block(net_msg, name='resx15', bottom=top_tmp, num_output=960, group=3, dilation=4)
    # top_tmp = add_shufflenet_block(net_msg, name='resx16', bottom=top_tmp, num_output=960, group=3, dilation=8)
    #
    # top_tmp = add_conv_bn_block(net_msg, name='resx17', bottom=top_tmp, num_output=256, pad=0, kernel_size=1, stride=1)
    #
    # top_tmp = add_ASPP_module(net_msg, name='aspp', bottom=top_tmp, height=height/8, width=width/8)
    #
    # top_tmp = add_conv_layer(net_msg, name='score', bottom=top_tmp, num_output=2, pad=0, kernel_size=1, stride=1)
    # top_tmp = add_interp_layer(net_msg, name='score_final', bottom=top_tmp, height=height, width=width)


    # '''
    #     ShuffleNet 0.5x g8 deeplab v3:
    #     You should delete 'resx1_conv1.group' and 'resx1_shuffle' by yourself
    # '''
    # group_channel = [96, 192, 384]
    # group = 8
    # add_conv_bn_block(net_msg, name='conv1', bottom='data', num_output=24, pad=1, kernel_size=3, stride=2)
    # add_pooling_layer(net_msg, name='pool1', bottom='conv1', pooling_type='MAX', kernel_size=3, stride=2)
    # 
    # top_tmp = add_shufflenet_block_stride(net_msg, name='resx1', bottom='pool1', input_channel=24, num_output=group_channel[0],
    #                                       group=group)
    # top_tmp = add_shufflenet_block(net_msg, name='resx2', bottom=top_tmp, num_output=group_channel[0], group=group)
    # top_tmp = add_shufflenet_block(net_msg, name='resx3', bottom=top_tmp, num_output=group_channel[0], group=group)
    # top_tmp = add_shufflenet_block(net_msg, name='resx4', bottom=top_tmp, num_output=group_channel[0], group=group)
    # 
    # top_tmp = add_shufflenet_block_stride(net_msg, name='resx5', bottom=top_tmp, input_channel=group_channel[0],
    #                                       num_output=group_channel[1], group=group, dilation=2)
    # top_tmp = add_shufflenet_block(net_msg, name='resx6', bottom=top_tmp, num_output=group_channel[1], group=group, dilation=4)
    # top_tmp = add_shufflenet_block(net_msg, name='resx7', bottom=top_tmp, num_output=group_channel[1], group=group, dilation=1)
    # top_tmp = add_shufflenet_block(net_msg, name='resx8', bottom=top_tmp, num_output=group_channel[1], group=group, dilation=2)
    # top_tmp = add_shufflenet_block(net_msg, name='resx9', bottom=top_tmp, num_output=group_channel[1], group=group, dilation=4)
    # top_tmp = add_shufflenet_block(net_msg, name='resx10', bottom=top_tmp, num_output=group_channel[1], group=group, dilation=1)
    # top_tmp = add_shufflenet_block(net_msg, name='resx11', bottom=top_tmp, num_output=group_channel[1], group=group, dilation=2)
    # top_tmp = add_shufflenet_block(net_msg, name='resx12', bottom=top_tmp, num_output=group_channel[1], group=group, dilation=4)
    # 
    # top_tmp = add_shufflenet_block_stride(net_msg, name='resx13', bottom=top_tmp, input_channel=group_channel[1],
    #                                       num_output=group_channel[2], group=group, dilation=2)
    # top_tmp = add_shufflenet_block(net_msg, name='resx14', bottom=top_tmp, num_output=group_channel[2], group=group, dilation=3)
    # top_tmp = add_shufflenet_block(net_msg, name='resx15', bottom=top_tmp, num_output=group_channel[2], group=group, dilation=5)
    # top_tmp = add_shufflenet_block(net_msg, name='resx16', bottom=top_tmp, num_output=group_channel[2], group=group, dilation=8)
    # 
    # top_tmp = add_conv_bn_block(net_msg, name='resx17_conv1', bottom=top_tmp, num_output=256, pad=0, kernel_size=1, group=group, stride=1)
    # 
    # top_tmp = add_ASPP_module(net_msg, name='aspp', bottom=top_tmp, height=height / 8, width=width / 8)
    # 
    # top_tmp = add_conv_layer(net_msg, name='score', bottom=top_tmp, num_output=2, pad=0, kernel_size=1, stride=1)
    # top_tmp = add_interp_layer(net_msg, name='score_final', bottom=top_tmp, height=height, width=width)


    '''
            simplify ShuffleNet 0.25x g8 + DeepLab v3 + skip connect:
            You should delete 'resx1_conv1.group' and 'resx1_shuffle' by yourself, also need modify aspp for shufflenet utilize
            And resx1's conv channel may not fix with group, you need to modify yourself
    '''
    group_channel = [96, 192, 384]
    group = 8
    add_conv_bn_block(net_msg, name='conv1', bottom='data', num_output=24, pad=1, kernel_size=3, stride=2)
    add_pooling_layer(net_msg, name='pool1', bottom='conv1', pooling_type='MAX', kernel_size=3, stride=2)

    top_tmp = add_shufflenet_block_stride(net_msg, name='resx1', bottom='pool1', input_channel=24, num_output=group_channel[0], group=group)
    top_tmp = add_shufflenet_block(net_msg, name='resx2', bottom=top_tmp, num_output=group_channel[0], group=group)
    top_tmp = add_shufflenet_block_stride(net_msg, name='resx3', bottom=top_tmp, input_channel=group_channel[0], num_output=group_channel[1], group=group, dilation=2)
    top_tmp = add_shufflenet_block(net_msg, name='resx4', bottom=top_tmp, num_output=group_channel[1], group=group)
    top_tmp = add_shufflenet_block_stride(net_msg, name='resx5', bottom=top_tmp, input_channel=group_channel[1], num_output=group_channel[2], group=group, dilation=2)
    top_tmp = add_shufflenet_block(net_msg, name='resx6', bottom=top_tmp, num_output=group_channel[2], group=group, dilation=4)
    top_tmp = add_shufflenet_block(net_msg, name='resx7', bottom=top_tmp, num_output=group_channel[2], group=group, dilation=2)

    top_tmp = add_conv_bn_block(net_msg, name='resx8_conv1', bottom=top_tmp, num_output=256, pad=0, kernel_size=1, group=group, stride=1)
    top_tmp = add_shufflechannel_layer(net_msg, name='resx8_shuffle', bottom=top_tmp, group=group)
    top_tmp = add_ASPP_module(net_msg, name='aspp', bottom=top_tmp, height=height / 8, width=width / 8)

    top_tmp = add_interp_layer(net_msg, name='upsample1', bottom=top_tmp, zoom_factor=2)
    add_conv_bn_block(net_msg, name='resx9_project', bottom='pool1', num_output=256, pad=0, kernel_size=1, stride=1)
    top_tmp = add_eltwise_add_layer(net_msg, name='resx9_eltwise', bottom1=top_tmp, bottom2='resx9_project')

    top_tmp = add_conv_layer(net_msg, name='score', bottom=top_tmp, num_output=2, pad=0, kernel_size=1, stride=1)

    top_tmp = add_interp_layer(net_msg, name='score_final', bottom=top_tmp, height=height, width=width)


    filepath = './net.prototxt'
    file = open(filepath, "w")
    if not file:
        raise IOError("ERROR (" + filepath + ")!")
    file.write(str(net_msg))
    file.close()


