if __name__ == "__main__":
    import numpy as np
    # generate the boundary
    f = lambda x: (5 * x + 1)
    bd_x = np.linspace(-1.0, 1, 200)
    bd_y = f(bd_x)
    # generate the training data
    input_size = 400* 1024
    x = np.random.uniform(-1, 1, input_size)
    y = f(x) + 2 * np.random.randn(len(x))
    # convert training data to 2d space
    label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)]).astype(np.int32)
    data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np.float32)
    print('data shape: {}'.format(data.shape))
    print('label shape: {}'.format(label.shape))
    print('x shape: {}'.format(x.shape))
    print('y shape: {}'.format(y.shape))
