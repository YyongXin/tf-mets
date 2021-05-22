import pickle, gzip
import numpy as np

# load dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, _ = pickle.load(f, encoding='latin1')
f.close()

print(train_set[0].shape, train_set[1].shape)
print(valid_set[0].shape, valid_set[1].shape)
# (50000, 784) (50000,)
# (10000, 784) (10000,)

train_x = np.reshape(train_set[0], (50000, 1, 28, 28)).astype(np.float32, copy=False)
valid_x = np.reshape(valid_set[0], (10000, 1, 28, 28)).astype(np.float32, copy=False)
train_y = np.array(train_set[1]).astype(np.int32, copy=False)
valid_y = np.array(valid_set[1]).astype(np.int32, copy=False)

train_x /= 255
valid_x /= 255

# train the cnn model
import sys
SINGA_PATH='/home/extend/lijiansong/work-space/anaconda2/envs/intel-caffe/lib/python3.6/site-packages'
sys.path.append(SINGA_PATH)
from singa import tensor
from singa import autograd
from singa import module
from singa import device
from singa import opt

class CNN(module.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = autograd.Conv2d(1, 20, 5, padding=0)
        self.conv2 = autograd.Conv2d(20, 40, 5, padding=0)
        self.linear1 = autograd.Linear(4 * 4 * 40, 200)
        self.linear2 = autograd.Linear(200, 10)
        self.pooling1 = autograd.MaxPool2d(2, 2, padding=0)
        self.pooling2 = autograd.MaxPool2d(2, 2, padding=0)

    # feed the data through operations
    def forward(self, x):
        y = self.conv1(x)
        y = autograd.relu(y)
        y = self.pooling1(y)
        y = self.conv2(y)
        y = autograd.relu(y)
        y = self.pooling2(y)
        y = autograd.flatten(y)
        y = self.linear1(y)
        y = autograd.relu(y)
        y = self.linear2(y)
        return y

    def loss(self, out, ty):
        return autograd.softmax_cross_entropy(out, ty)

    def optim(self, loss):
        return self.optimizer.backward_and_update(loss)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

# create device (set default device) before build model
dev = device.create_cuda_gpu_on(0)
model = CNN()

# define the evaluation metric
def accuracy(pred, target):
    # y is binary array for softmax output
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    return correct


# conduct training
sgd = opt.SGD(lr=0.01, momentum=0.9, weight_decay=1e-5)
max_epoch = 30
batch_size = 100
tx = tensor.Tensor((batch_size, 1, 28, 28), dev, tensor.float32)
ty = tensor.Tensor((batch_size, ), dev, tensor.int32)
num_train_batch = train_x.shape[0] // batch_size
num_valid_batch = valid_x.shape[0] // batch_size
idx = np.arange(train_x.shape[0], dtype=np.int32)
# attached model to graph
model.on_device(dev)
model.set_optimizer(sgd)
model.graph(True, False)

# the training phase
for epoch in range(max_epoch):
    np.random.shuffle(idx)
    print('Starting Epoch %d:' % (epoch))

    # training phase
    train_correct = np.zeros(shape=[1],dtype=np.float32)
    valid_correct = np.zeros(shape=[1],dtype=np.float32)
    train_loss = np.zeros(shape=[1],dtype=np.float32)
    model.train()
    for b in range(num_train_batch):
        x = train_x[idx[b * batch_size: (b + 1) * batch_size]]
        y = train_y[idx[b * batch_size: (b + 1) * batch_size]]
        tx.copy_from_numpy(x)
        ty.copy_from_numpy(y)
        out = model(tx)
        loss = model.loss(out, ty)
        model.optim(loss)
        train_correct += accuracy(tensor.to_numpy(out), y)
        train_loss += tensor.to_numpy(loss)[0]

    # dump the training loss and accuracy
    print('training loss = %f, training accuracy = %f' %(train_loss, train_correct / (num_train_batch*batch_size)))


# inference
model.eval()
valid_correct = np.zeros(shape=[1],dtype=np.float32)
for b in range(num_valid_batch):
    x = valid_x[b * batch_size: (b + 1) * batch_size]
    y = valid_y[b * batch_size: (b + 1) * batch_size]
    tx.copy_from_numpy(x)
    ty.copy_from_numpy(y)
    out_valid = model(tx)
    valid_correct += accuracy(tensor.to_numpy(out_valid), y)

print('Validation Samples = %d, Correctly Classified = %d, Test Accuracy = %f' %(batch_size * num_valid_batch, valid_correct, valid_correct / (batch_size * num_valid_batch)))
