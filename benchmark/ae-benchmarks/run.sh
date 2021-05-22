#!/bin/bash

#python3 train.py alexnet cifar100 --epoch 5
#python3 train.py alexnet cifar100 --epoch 1
python3 train.py vgg cifar10 --epoch 1 --bs 1
