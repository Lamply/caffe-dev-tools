# -*- coding:utf-8 -*-
caffe_root = '/home/sad/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np


def pruning_zero_filters(proto, weights):
    """
    This code is for pruning network which has zero filters exist ( L1-norm is zero ).
    It require input BatchNorm-absorbed network.
    :param proto:
    :param weights:
    :return:
    """
    caffe.set_mode_cpu()
    net0 = caffe.Net(proto, weights, caffe.TEST)

    prune = 0.1
    for i in range(len(L1_name)):
        print(L1_name[i])
        sort_vec = np.argsort(L1_value[i], axis=0)  # return small-first sorted index of L1_value via axis 0
        filtes_num = len(sort_vec)
        prune_num = int(filtes_num * prune)
        print("prune %d filters this layer." % prune_num)
        prune_idx = []
        for cnt in range(prune_num):
            prune_idx.append(sort_vec[cnt])

        print("prune filters idx: %s" % prune_idx)

        #     if net0.params[L1_name[i]][2].size
        print(net0.params[L1_name[i]].size)

        # Cut conv layer
        net0.params[L1_name[i]][0].data = np.delete(net0.params[L1_name[i]][0].data, prune_idx, axis=0)

        # Cut Next conv layer input
        if L1_name.has_key(i + 1):
            net0.params[L1_name[i + 1]][0].data = np.delete(net0.params[L1_name[i + 1]][0].data, prune_idx, axis=1)

        # Cut BN layer
        if 'bn' in keys0[L1_layer_idx[i] + 1]:
            print("BN layer cut is needed.")
            bn_param = net0.params[keys0[L1_layer_idx[i] + 1]][0].data
            net0.params[keys0[L1_layer_idx[i] + 1]][0].data = np.delete(bn_param, prune_idx, axis=0)

        # Cut scale layer
        if 'scale' in keys0[L1_layer_idx[i] + 2]:
            print("scale layer cut is needed.")
            scale_param = net0.params[keys0[L1_layer_idx[i] + 2]][0].data
            net0.params[keys0[L1_layer_idx[i] + 2]][0].data = np.delete(scale_param, prune_idx, axis=0)

        print("\n")
