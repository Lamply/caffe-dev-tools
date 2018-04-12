#coding=utf-8
'''
@author: kangkai
'''
import sys
sys.path.insert(0, '/home/sad/caffe/python')
from caffe.proto import caffe_pb2

def toPrototxt(modelName, deployName):
    with open(modelName, 'rb') as f:
        caffemodel = caffe_pb2.NetParameter()
        caffemodel.ParseFromString(f.read())

    for item in caffemodel.layers:
        item.ClearField('blobs')
    for item in caffemodel.layer:
        item.ClearField('blobs')
        
    with open(deployName, 'w') as f:
        f.write(str(caffemodel))

if __name__ == '__main__':
    modelName = sys.argv[1]
    deployName = 'deploy.prototxt'
    toPrototxt(modelName, deployName)

