# -*- coding: UTF-8 -*-
import caffe
import cv2
import numpy as np
import matplotlib.pyplot as plt
from visFunc import VisFunc

caffe.set_mode_gpu() # set GPU mode

# load caffe net
model_def = '/your/path/to/VGG16_deploy.prototxt'
model_weights = '/your/path/to/VGG16.caffemodel'
net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)

# preprocessing
mu = np.load('/your/path/to/mean.npy')
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

# load a image and preprocessing
image = caffe.io.load_image('/your/path/to/img.jpg')
transformed_image = transformer.preprocess('data', image)

# forward
net.blobs['data'].data[...] = transformed_image
net.forward()

# visualization
vis_func = VisFunc(net)
layers = []
for layer_name, _ in net.blobs.iteritems():
    layers.append(layer_name)
vis = net.blobs['fc8'].data[0, :]
for i in range(len(layers)-2, 0, -1):
    if layers[i][:2] == 'fc':
        vis = vis_func.refc(vis, layers[i])
    elif layers[i][:2] == 'po':
        vis = vis_func.unpool(vis, layers[i-1], layers[i])
    elif layers[i][:2] == 'co':
        vis = vis_func.deconv(vis, layers[i])
    if layers[i] == 'fc6':
        vis = vis.reshape(512, 7, 7)
vis = vis.transpose(1, 2, 0)
vis = (vis - vis.min()) / (vis.max() - vis.min())
plt.imsave('vis.png', vis)




