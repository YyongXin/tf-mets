""" CIFAR10 dataset is avaliable at https://www.cs.toronto.edu/~kriz/cifar.html.
It includes 5 binary dataset, each contains 10000 images. 1 row (1 image)
includes 1 label & 3072 pixels. 3072 pixels are 3 channels of a 32x32 image
"""
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
try:
    import pickle
except ImportError:
    import cPickle as pickle
import numpy as np
import os
import argparse
from datetime import datetime
import time

import sys
SINGA_PATH='/home/extend/lijiansong/work-space/anaconda2/envs/intel-caffe/lib/python3.6/site-packages'
sys.path.append(SINGA_PATH)

from singa import utils
from singa import optimizer
from singa import device
from singa import tensor
from singa.proto import core_pb2

import vgg

def load_dataset(filepath):
    print('Loading data file %s' % filepath)
    with open(filepath, 'rb') as fd:
        cifar10 = pickle.load(fd, encoding='latin1')
    image = cifar10['data'].astype(dtype=np.uint8)
    image = image.reshape((-1, 3, 32, 32))
    label = np.asarray(cifar10['labels'], dtype=np.uint8)
    label = label.reshape(label.size, 1)
    return image, label

def load_train_data(dir_path, num_batches=5):
    labels = []
    batchsize = 10000
    images = np.empty((num_batches * batchsize, 3, 32, 32), dtype=np.uint8)
    for did in range(1, num_batches + 1):
        fname_train_data = dir_path + "/data_batch_{}".format(did)
        image, label = load_dataset(fname_train_data)
        images[(did - 1) * batchsize:did * batchsize] = image
        labels.extend(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels

def load_test_data(dir_path):
    images, labels = load_dataset(dir_path + "/test_batch")
    return np.array(images,  dtype=np.float32), np.array(labels, dtype=np.int32)

def normalize_for_vgg(train_x, test_x):
    mean = train_x.mean()
    std = train_x.std()
    train_x -= mean
    test_x -= mean
    train_x /= std
    test_x /= std
    return train_x, test_x

def vgg_lr(epoch):
    return 0.1 / float(1 << (epoch // 25))

def train(data, net, max_epoch, get_lr, weight_decay, batch_size=100,
          use_cpu=False):
    print('==> start intialization...')
    if use_cpu:
        print('==> Using CPU')
        dev = device.get_default_device()
    else:
        print('==> Using GPU')
        dev = device.create_cuda_gpu_on(0)

    net.to_device(dev)
    opt = optimizer.SGD(momentum=0.9, weight_decay=weight_decay)
    for (p, specs) in zip(net.param_names(), net.param_specs()):
        opt.register(p, specs)

    tx = tensor.Tensor((batch_size, 3, 32, 32), dev)
    ty = tensor.Tensor((batch_size,), dev, core_pb2.kInt)
    train_x, train_y, test_x, test_y = data
    num_train_batch = train_x.shape[0] // batch_size
    num_test_batch = test_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)
    fileTimeLog =open("epochTimeLog.text","a")
    for epoch in range(1):
        np.random.shuffle(idx)
        loss, acc = 0.0, 0.0
        print('Epoch %d' % epoch)
        # miliseconds
        print(datetime.now().timetz())
        print(int(round(time.time()*1000)))
        fileTimeLog.write('Epoch %d: ' % epoch)
        fileTimeLog.write(str(int(round(time.time()*1000))))
        fileTimeLog.write('\n')
        #num_train_batch
        for b in range(20):
            print ("start of iteration %d: " %b)
            #time.sleep(1)
            fileTimeLog.write('iteration %d: ' % b)
            fileTimeLog.write(str(int(round(time.time()*1000))))
            fileTimeLog.write('\n')
            x = train_x[idx[b * batch_size: (b + 1) * batch_size]]
            y = train_y[idx[b * batch_size: (b + 1) * batch_size]]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            grads, (l, a) = net.train(tx, ty)
            loss += l
            acc += a
            for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
                opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s), b)
            # update progress bar
            utils.update_progress(b * 1.0 / num_train_batch,
                                  'training loss = %f, accuracy = %f' % (l, a))
        info = '\ntraining loss = %f, training accuracy = %f, lr = %f' \
            % ((loss / num_train_batch), (acc / num_train_batch), get_lr(epoch))
        print(info)

        loss, acc = 0.0, 0.0
        for b in range(0):
            x = test_x[b * batch_size: (b + 1) * batch_size]
            y = test_y[b * batch_size: (b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            l, a = net.evaluate(tx, ty)
            loss += l
            acc += a

        print('test loss = %f, test accuracy = %f' %
              ((loss / num_test_batch), (acc / num_test_batch)))
    fileTimeLog.close()
    net.save('model', 20)  # save model params into checkpoint file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train cnn for cifar10')
    parser.add_argument('model', choices=['vgg', 'alexnet'],
                        default='vgg')
    parser.add_argument('data', default='cifar-10-batches-py')
    parser.add_argument('--use_cpu', action='store_false')
    parser.add_argument('batch_size', type=int, default=100)
    args = parser.parse_args()
    assert os.path.exists(args.data), \
        'Pls download the cifar10 dataset via "download_data.py py"'
    print('==> loading data ...')
    train_x, train_y = load_train_data(args.data)
    test_x, test_y = load_test_data(args.data)
    if args.model == 'vgg':
        train_x, test_x = normalize_for_vgg(train_x, test_x)
        #net = vgg.create_net(args.use_cpu)
        net = vgg.create_net(args.use_cpu)
        train((train_x, train_y, test_x, test_y), net, 250, vgg_lr, 0.0005,
              use_cpu=args.use_cpu,batch_size=args.batch_size)
    else:
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        net = resnet.create_net(args.use_cpu)
        train((train_x, train_y, test_x, test_y), net, 200, resnet_lr, 1e-4,
              use_cpu=args.use_cpu,batch_size=args.batch_size)
