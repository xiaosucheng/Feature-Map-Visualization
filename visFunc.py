# -*- coding: UTF-8 -*-
import numpy as np
import cv2

class VisFunc:
    def __init__(self, net):
	    self.net = net

    def reverse(self, inp):
        ''' reverse the vector of every row of array

        inp: C x N tensor

        return: C x N tensor
        '''
        _, col = inp.shape
        out = inp[:, [ix for ix in range(col-1, -1, -1)]]
        return out

    def refc(self, inp, name):
        ''' calculate reversal of fully-connected layer

        inp: N-dimension vector
        name: name of the current layer

        return: M-dimension vector
        '''
        filters = self.net.params[name][0].data
        out = np.dot(self.reverse(filters).transpose(1, 0), inp)
        return out

    def deconv(self, inp, name):
        ''' calculate deconvolution

        inp: feature map, C x H x W tensor
        name: name of the current layer

        return: C' x H x W tensor， where C' is the number of channels of any kernel
        '''
        filters = self.net.params[name][0].data
        shp1 = inp.shape
        shp2 = filters.shape
        out = np.zeros((shp2[1], shp1[1], shp1[2]))
        inp = inp * (inp >= 0)
        for i in range(shp2[1]):
            for j in range(shp1[0]):
                rev_filter = np.rot90(np.rot90(filters[j, i]))
                out[i] = out[i] + cv2.filter2D(inp[j], -1, rev_filter)
        return out

    def unpool(self, inp, nameF, nameC):
        ''' calculate unpooling

        inp: feature map, C x H x W tensor
        nameF: name of the former layer
        nameC: name of the current layer

        return: C x 2*H x 2*W tensor
        '''
        former_pool = self.net.blobs[nameF].data[0, :]
        pool = self.net.blobs[nameC].data[0,:]
        inp = np.kron(inp, np.array([[1,1],[1,1]]))
        find_zeros = pool==0
        pool = pool + find_zeros
        pool = np.kron(pool, np.array([[1,1],[1,1]]))
        set_zeros = pool==former_pool
        out = inp * set_zeros
        return out
