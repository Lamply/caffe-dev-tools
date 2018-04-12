'''
Untest
'''
import numpy as np
import sys,os
caffe_root = '/home/sad/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

src_model = 'examples/fashion_mnist/mnist_resnet_train_test_n3.prototxt'
dst_model = 'examples/fashion_mnist/mnist_resnet_train_test_inplace.prototxt'

'''
 Set all BatchNorm and Scale ReLU as inplace op
'''
def inplace_conv_block(model):

    # load the prototxt file as a protobuf message
    with open(model) as k:
        str1 = k.read()
    msg1 = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg1)

    # search for bn and scale layer and remove them
    for i, l in enumerate(msg1.layer):
        if l.type == "BatchNorm":
            print("inplace layer %s..." % l.name)
            msg1.layer[i].top.remove(msg1.layer[i].top[0])
            msg1.layer[i].top.append(msg1.layer[i].bottom[0])
            msg1.layer[i+1].bottom.remove(msg1.layer[i+1].bottom[0])
            msg1.layer[i+1].bottom.append(msg1.layer[i].top[0])
        if l.type == "Scale":
            print("inplace layer %s..." % l.name)
            msg1.layer[i].top.remove(msg1.layer[i].top[0])
            msg1.layer[i].top.append(msg1.layer[i].bottom[0])
            msg1.layer[i+1].bottom.remove(msg1.layer[i+1].bottom[0])
            msg1.layer[i+1].bottom.append(msg1.layer[i].top[0])
        if l.type == "ReLU":
            print("inplace layer %s..." % l.name)
            msg1.layer[i].top.remove(msg1.layer[i].top[0])
            msg1.layer[i].top.append(msg1.layer[i].bottom[0])
            msg1.layer[i+1].bottom.remove(msg1.layer[i+1].bottom[0])
            msg1.layer[i+1].bottom.append(msg1.layer[i].top[0])

    return msg1

model_inplace = inplace_conv_block(src_model)

# save prototxt for inference
print "Saving inplace prototxt file..."
path = os.path.join(dst_model)
with open(path, 'w') as m:
    m.write(text_format.MessageToString(model_inplace))
