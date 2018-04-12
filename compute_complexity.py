"""
This is a caffe tools for estimating caffe model's params, FLOPs and memory access counting.

Notice:

   1. Just consider net.params[layer.name][0], Ignore extra term such as bias

   2. In here FLOPs are calculated as MA( multiply-add )

   3. Strongly recommend absorbing batch normalization before estimate

"""
import sys
caffe_root = '/home/sad/caffe/python/'        # Change this to your project's caffe path 
sys.path.insert(0, caffe_root)
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import tempfile
import os
import sys

__author__ = 'Lamply'
__email__ = 'Lamplykyz@gmail.com'

'''
    Simple handle to create netspec files. This code snippet is on lines
    of: https://github.com/BVLC/caffe/blob/master/python/caffe/test/test_net.py
'''
def _create_file_from_netspec(netspec):
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(netspec.to_proto()))
    return f.name


'''
    This is a utility function which computes the complexity and memory access of a given network.
    This is a modified version from github project pynetbuilder by @Jay Mahadeokar
'''
def get_complexity(netspec=None, prototxt_file=None):
    # One of netspec, or prototxt_path params should not be None
    assert (netspec is not None) or (prototxt_file is not None)

    if netspec is not None:
        prototxt_file = _create_file_from_netspec(netspec)
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_file, caffe.TEST)

    total_params = 0
    total_flops = 0
    memory_access = 0

    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt_file).read(), net_params)

    print("\n%-20s\t%-10s\t%-10s\t%s\n" % ('name', 'params', 'memory', 'flops'))

    for idx, layer in enumerate(net_params.layer):

        # print("layer name: %s" % layer.name)

        # Compute input/output memory access
        input_size = 0
        output_size = 0
        for bnum in range(len(layer.bottom)):
            # print(net.blobs[layer.bottom[bnum]].data.shape)
            input_size += net.blobs[layer.bottom[bnum]].data.size
        for tnum in range(len(layer.top)):
            # print(net.blobs[layer.top[tnum]].data.shape)
            output_size += net.blobs[layer.top[tnum]].data.size
        # print("input size: %d" % input_size)
        # print("output size: %d" % output_size)

        layer_memory_size = input_size + output_size
        memory_access += layer_memory_size


        # Compute weights size and FLOPs
        if layer.name in net.params:
            params_size = net.params[layer.name][0].data.size

            # If convolution layer, multiply flops with receptive field
            # i.e. #params * datawidth * dataheight
            if layer.type == 'Convolution':  # 'conv' in layer:
                data_width = net.blobs[layer.name].data.shape[3]    # net.blobs[layer.top[0]].data.shape
                data_height = net.blobs[layer.name].data.shape[2]
                flops = net.params[layer.name][0].data.size * data_width * data_height
                if layer.convolution_param.stride != []:
                    flops /= layer.convolution_param.stride[0]
                if layer.convolution_param.dilation != []:
                    flops /= layer.convolution_param.dilation[0]
                # print >> sys.stderr, layer.name, params, flops
            elif layer.type == 'ConvolutionDepthwise' or layer.type == 'DepthwiseConvolution':  # 'depthwise_conv' in layer:
                data_channel = net.blobs[layer.name].data.shape[1]
                data_width = net.blobs[layer.name].data.shape[3]
                data_height = net.blobs[layer.name].data.shape[2]
                kernel_width = net.params[layer.name][0].data.shape[3]
                kernel_height = net.params[layer.name][0].data.shape[2]
                flops = kernel_width * kernel_height * data_width * data_height * data_channel
                if layer.convolution_param.stride != []:
                    flops /= layer.convolution_param.stride[0]
                if layer.convolution_param.dilation != []:
                    flops /= layer.convolution_param.dilation[0]
            elif layer.type == 'Deconvolution':
                tmp = idx-1
                for inp in range(idx-1, -1, -1):
                    if len(net_params.layer[inp].bottom) == len(net_params.layer[inp].top) and net_params.layer[inp].bottom[0] == net_params.layer[inp].top[0]:   # Check inplace, may bug
                        tmp = inp-1
                    else:
                        break
                data_width = net.blobs[net_params.layer[tmp].name].data.shape[3]
                data_height = net.blobs[net_params.layer[tmp].name].data.shape[2]
                # print(net.blobs[net_params.layer[tmp].name].data.shape)
                flops = net.params[layer.name][0].data.size * data_width * data_height
            else:
                flops = net.params[layer.name][0].data.size

            print("%-20s\t%-10s\t%-10s\t%s" % (layer.name, digit2string(params_size), digit2string(layer_memory_size+params_size), digit2string(flops)))
            total_params += params_size
            total_flops += flops
            memory_access += params_size

    if netspec is not None:
        os.remove(prototxt_file)

    memory_access = memory_access * 4      # float point 4 Byte

    return total_params, total_flops, memory_access

def digit2string(x):
    x = float(x)
    if x < 10 ** 3:
        return "%.2f" % float(x)
    elif x < 10 ** 6:
        x = x / 10 ** 3
        return "%.2f" % float(x) + 'K'
    elif x < 10 ** 9:
        x = x / 10 ** 6
        return "%.2f" % float(x) + 'M'
    else:
        x = x / 10 ** 9
        return "%.2f" % float(x) + 'G'


if __name__ == '__main__':
    length = len(sys.argv)
    if length == 1:
        filepath = 'deploy.prototxt'
    else:
        filepath = sys.argv[1]
    params, flops, memory = get_complexity(prototxt_file=filepath)
    print('\n ########### result ###########')
    print('#params=%s, #FLOPs=%s, #Memory Access=%s' % (digit2string(params), digit2string(flops), digit2string(memory)))





