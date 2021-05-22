#!/bin/bash
rm -f *.csv
python3 mnist_cnn.py mnist.npz --use_swap_cuda_gpu --batch-size 300
