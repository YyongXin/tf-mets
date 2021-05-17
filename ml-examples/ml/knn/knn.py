import numpy as np
import tensorflow as tf
import argparse

from datetime import datetime
import time

"""
REFs:
- https://github.com/artifabrian/dynamic-knn-gpu
- https://zhuanlan.zhihu.com/p/30210438
"""

def random_gen_dataset(num_points=4096, num_kinds_of_features=128):
    label_set = ['label'+str(i) for i in range(num_kinds_of_features)]
    # Input data points: [[4.7 3.2 1.3 0.2], [4.6 3.1 1.5 0.2], ...]
    # Original iris shape: (150, 4).
    x = np.random.randn(num_points, 128).astype(np.float32)
    # Label info: [0, 0, 0, ..., 0, 1, 1, 1, ...,1, 2, 2, 2, ...,2]
    # Origin iris shape: (150)
    y = np.random.randint(num_kinds_of_features, size=num_points)
    """
    One-hot encode the labels.
    Note: np.eye returns a two-dimensional array with ones on a diagonal,
    defaulting to the main diagonal.
    Indexing with y then gives us the required one-hot encoding of y.
    """
    y = np.eye(len(set(y)))[y]

    # Normalize our features to be in the range of zero to one.
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

    # Create indices for the train-test split.
    np.random.seed(42)
    split = 0.5
    train_indices = np.random.choice(len(x), round(len(x) * split), replace=False)
    test_indices =np.array(list(set(range(len(x))) - set(train_indices)))

    # Train and test split.
    train_x = x[train_indices]
    test_x = x[test_indices]
    train_y = y[train_indices]
    test_y = y[test_indices]

    return train_x, test_x, train_y, test_y, label_set

train_x, test_x, train_y, test_y, label_set = random_gen_dataset()
k = 16

# knn has no training process!!!
def prediction(train_x, test_x, train_y, k, num_itreations=20):
    # miliseconds
    print(datetime.now().timetz())
    time_list = []
    # time in ms
    cur_time = int(round(time.time()*1000))
    for i in range(num_itreations):
        print('==> iteration {}'.format(i))
        distances = tf.reduce_sum(tf.abs(tf.subtract(train_x, tf.expand_dims(test_x, axis =1))), axis=2)
        # sleep 5 ms
        #time.sleep(0.005)
        _, top_k_indices = tf.nn.top_k(tf.negative(distances), k=k)
        top_k_labels = tf.gather(train_y, top_k_indices)
        predictions_sum = tf.reduce_sum(top_k_labels, axis=1)
        pred = tf.argmax(predictions_sum, axis=1)
        next_time = int(round(time.time()*1000))
        time_list.append(next_time - cur_time)
        cur_time = next_time
    print('throughput: {} ms!!!'.format(np.average(np.array(time_list))))


if __name__ == '__main__':
    prediction(train_x, test_x, train_y, k)
