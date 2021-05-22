from singa import autograd
from singa import module

import numpy as np

# TODO: JSON LEE, refacting to be more pythonic!!!
class VGGNet(module.Module):

    def __init__(self, num_classes=10, num_channels=3, in_size=184):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        self.input_size = in_size
        self.dimension = 4

        # autograd.Conv2d, default stride is 1
        self.conv1 = autograd.Conv2d(num_channels, 64, 3, padding=1)
        self.bn1 = autograd.BatchNorm2d(64)
        self.pooling1 = autograd.MaxPool2d(2, 2, padding=0)

        self.conv2 = autograd.Conv2d(64, 64, 3, padding=1)
        self.bn2 = autograd.BatchNorm2d(64)

        self.conv3 = autograd.Conv2d(64, 128, 3, padding=1)
        self.bn3 = autograd.BatchNorm2d(128)

        self.conv4 = autograd.Conv2d(128, 128, 3, padding=1)
        self.bn4 = autograd.BatchNorm2d(128)
        self.pooling2 = autograd.MaxPool2d(2, 2, padding=0)

        self.conv5 = autograd.Conv2d(128, 256, 3, padding=1)
        self.bn5 = autograd.BatchNorm2d(256)

        self.conv6 = autograd.Conv2d(256, 256, 3, padding=1)
        self.bn6 = autograd.BatchNorm2d(256)

        self.conv7 = autograd.Conv2d(256, 256, 3, padding=1)
        self.bn7 = autograd.BatchNorm2d(256)
        self.pooling3 = autograd.MaxPool2d(2, 2, padding=0)

        self.conv8 = autograd.Conv2d(256, 512, 3, padding=1)
        self.bn8 = autograd.BatchNorm2d(512)

        self.conv9 = autograd.Conv2d(512, 512, 3, padding=1)
        self.bn9 = autograd.BatchNorm2d(512)
        self.pooling4 = autograd.MaxPool2d(2, 2, padding=0)

        self.conv13 = autograd.Conv2d(512, 512, 3, padding=1)
        self.bn13 = autograd.BatchNorm2d(512)

        self.conv10 = autograd.Conv2d(512, 512, 3, padding=1)
        self.bn10 = autograd.BatchNorm2d(512)

        self.conv11 = autograd.Conv2d(512, 512, 3, padding=1)
        self.bn11 = autograd.BatchNorm2d(512)

        self.conv12 = autograd.Conv2d(512, 512, 3, padding=1)
        self.bn12 = autograd.BatchNorm2d(512)
        self.pooling5 = autograd.MaxPool2d(2, 2, padding=0)

        if self.input_size == 224:
            self.linear1 = autograd.Linear(25088, 512)
        elif self.input_size == 184:
            self.linear1 = autograd.Linear(12800, 512)
        elif self.input_size == 32 or\
        self.input_size == 28 or\
        self.input_size == 16:
            self.linear1 = autograd.Linear(512, 512)
        self.linear2 = autograd.Linear(512, num_classes)

    def forward(self, x):
        # conv1_1: conv_bn_relu
        y = self.conv1(x)
        y = self.bn1(y)
        y = autograd.relu(y)
        y = autograd.dropout(y, 0.3)
        # conv1_2: conv_bn_relu
        y = self.conv2(y)
        y = self.bn2(y)
        y = autograd.relu(y)
        print(y.shape)
        y = self.pooling1(y)
        print(y.shape)
        # conv2_1: conv_bn_relu
        y = self.conv3(y)
        y = self.bn3(y)
        y = autograd.relu(y)
        y = autograd.dropout(y, 0.4)
        # conv2_2: conv_bn_relu
        y = self.conv4(y)
        y = self.bn4(y)
        y = autograd.relu(y)
        print(y.shape)
        y = self.pooling2(y)
        print(y.shape)
        # conv3_1: conv_bn_relu
        y = self.conv5(y)
        y = self.bn5(y)
        y = autograd.relu(y)
        y = autograd.dropout(y, 0.4)
        # conv3_2: conv_bn_relu
        y = self.conv6(y)
        y = self.bn6(y)
        y = autograd.relu(y)
        y = autograd.dropout(y, 0.4)
        # conv3_3: conv_bn_relu
        y = self.conv7(y)
        y = self.bn7(y)
        y = autograd.relu(y)
        print(y.shape)
        y = self.pooling3(y)
        print(y.shape)
        # conv4_1: conv_bn_relu
        y = self.conv8(y)
        y = self.bn8(y)
        y = autograd.relu(y)
        y = autograd.dropout(y, 0.4)
        # conv4_2: conv_bn_relu
        y = self.conv9(y)
        y = self.bn9(y)
        y = autograd.relu(y)
        y = autograd.dropout(y, 0.4)
        # conv4_3: conv_bn_relu
        y = self.conv13(y)
        y = self.bn13(y)
        y = autograd.relu(y)
        print(y.shape)
        y = self.pooling4(y)
        print(y.shape)
        # conv5_1: conv_bn_relu
        y = self.conv10(y)
        y = self.bn10(y)
        y = autograd.relu(y)
        y = autograd.dropout(y, 0.4)
        # conv5_2: conv_bn_relu
        y = self.conv11(y)
        y = self.bn11(y)
        y = autograd.relu(y)
        y = autograd.dropout(y, 0.4)
        # conv5_3: conv_bn_relu
        y = self.conv12(y)
        y = self.bn12(y)
        y = autograd.relu(y)
        y = self.pooling5(y)
        # (512, 7, 7) ==> (25088)
        print('===>', y.shape)
        y = autograd.flatten(y)
        flat_num = np.product(y.shape)
        print('---------> flat_num', flat_num)
        y = autograd.dropout(y, 0.5)
        print('===>', y.shape)
        y = self.linear1(y)
        print('===>', y.shape)
        y = self.bn9(y)
        y = autograd.relu(y)
        y = autograd.dropout(y, 0.5)
        y = self.linear2(y)
        return y

    def loss(self, out, ty):
        return autograd.softmax_cross_entropy(out, ty)

    def optim(self, loss, dist_option, spars):
        if dist_option == 'fp32':
            self.optimizer.backward_and_update(loss)
        elif dist_option == 'fp16':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


def create_model(pretrained=False, **kwargs):
    """Constructs a AlexNet model.

    Args:
        pretrained (bool): If True, returns a model pre-trained
    """
    model = VGGNet(**kwargs)

    return model


__all__ = ['VGGNet', 'create_model']
