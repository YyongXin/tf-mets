from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

import argparse
from datetime import datetime
import time

"""
Auto-Encoder Example.
Build a 2 layers auto-encoder with TensorFlow v2 to compress images
to a lower latent space and then reconstruct them.

REFs: https://github.com/aymericdamien/TensorFlow-Examples
"""

# MNIST Dataset parameters.
# data features (img shape: 28*28).
img_h, img_w = 300, 300
num_features = img_h*img_w

# Training parameters.
learning_rate = 0.01
training_steps = 5
batch_size = 4
display_step = 1

# Network Parameters
# 1st layer num features.
num_hidden_1 = 3072
#num_hidden_1 = 256
# 2nd layer num features (the latent dim).
num_hidden_2 = 1024

"""
# Prepare MNIST data.
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""

num_imgs = 5000
x_train = np.random.randn(num_imgs, img_h, img_w).astype(np.float32)
y_train = np.random.randn(num_imgs).astype(np.float32)

# x_train shape: (60000, 28, 28)
# y_train shape: (60000,)
print('x_train shape: {}'.format(x_train.shape))
print('y_train shape: {}'.format(y_train.shape))

# Convert to float32.
x_train = x_train.astype(np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train = x_train.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train = x_train / 255.


# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
shuffle_size = int(num_imgs/10)
train_data = train_data.repeat().shuffle(shuffle_size).batch(batch_size).prefetch(batch_size)

# Store layers weight & bias
# A random value generator to initialize weights.
random_normal = tf.initializers.RandomNormal()

weights = {
    'encoder_h1': tf.Variable(random_normal([num_features, num_hidden_1])),
    'encoder_h2': tf.Variable(random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(random_normal([num_hidden_1, num_features])),
}
biases = {
    'encoder_b1': tf.Variable(random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(random_normal([num_features])),
}

# Building the encoder.
def encoder(x):
    # Encoder Hidden layer with sigmoid activation.
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation.
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder.
def decoder(x):
    # Decoder Hidden layer with sigmoid activation.
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation.
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# Mean square loss between original images and reconstructed ones.
def mean_square(reconstructed, original):
    return tf.reduce_mean(tf.pow(original - reconstructed, 2))

# Adam optimizer.
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# Optimization process.
def run_optimization(x):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        reconstructed_image = decoder(encoder(x))
        loss = mean_square(reconstructed_image, x)
    # Variables to update, i.e. trainable variables.
    # FIXME: TypeError: unsupported operand type(s) for +: 'dict_values' and 'dict_values'
    trainable_variables = list(weights.values()) + list(biases.values())
    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss

# miliseconds
print(datetime.now().timetz())
time_list = []
# time in ms
cur_time = int(round(time.time()*1000))
# Run training for the given number of steps.
for step, (batch_x, _) in enumerate(train_data.take(training_steps + 1)):
    # Run the optimization.
    #print('batch_x shape: {}'.format(batch_x.shape))
    loss = run_optimization(batch_x)
    next_time = int(round(time.time()*1000))
    time_list.append(next_time - cur_time)
    cur_time = next_time
    if step % display_step == 0:
        print("step: %i, loss: %f" % (step, loss))

print('throughput: {} ms!!!'.format(np.average(np.array(time_list))))
