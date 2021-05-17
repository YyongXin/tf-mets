from __future__ import print_function
import tensorflow as tf
import numpy as np

import argparse
from datetime import datetime
import time

'''
Gradient Boosted Decision Tree (GBDT).
Implement a Gradient Boosted Decision tree with TensorFlow to classify
handwritten digit images. This example is using the MNIST database of
handwritten digits as training samples (http://yann.lecun.com/exdb/mnist/).

REFs:
- <https://github.com/aymericdamien/TensorFlow-Examples>
'''

img_h, img_w = 224, 224
num_features = img_h*img_w
# NOTE: boosted_trees.py, for now pruning is not supported with multi class!!!
# So num_classes can only be binary class.
num_classes = 2
num_points = 1000

x_train = np.random.randn(num_points, num_features).astype(np.int32)
y_train = np.random.randint(num_classes, size=num_points)

# x_train shape: (455, 13), where 13 is num_features.
# y_train shape: (455,)
print('===> x_train shape: {}'.format(x_train.shape))
print('===> y_train shape: {}'.format(y_train.shape))

# Training parameters.
max_steps = 10
batch_size = 512
learning_rate = 1.0
l1_regul = 0.0
l2_regul = 0.1

NUM_EXAMPLES = len(y_train)
def make_input_fn(X, y, shuffle=True, num_epochs=None):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(({'x': X}, y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset
    return input_fn

#train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
#    x={'x': x_train}, y=y_train,
#    batch_size=batch_size, num_epochs=None, shuffle=True)train_input_fn = make_input_fn(x_train, y_train, num_epochs=10)

train_input_fn = make_input_fn(x_train, y_train, num_epochs=1)
feature_columns = [tf.feature_column.numeric_column(key='x', shape=(num_features,))]

# GBDT parameters.
num_batches_per_layer = 1000
num_trees = 10
max_depth = 16

gbdt_classifier = tf.estimator.BoostedTreesClassifier(
    n_batches_per_layer=num_batches_per_layer,
    feature_columns=feature_columns,
    n_classes=num_classes,
    learning_rate=learning_rate,
    n_trees=num_trees,
    max_depth=max_depth,
    l1_regularization=l1_regul,
    l2_regularization=l2_regul
)

gbdt_classifier.train(train_input_fn, max_steps=max_steps)
