# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from mxnet import gluon, np, npx
from mxnet import autograd
npx.set_np()

batch_size = 256
def load_data_fashion_mnist(batch_size):
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=4),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=4))
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()

def relu(X):
    return np.maximum(X, 0)

def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2

def softmax(y_hat):
    exps = np.exp(y_hat - np.max(y_hat, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def loss(y_hat, y):
    m = y.shape[0]
    p = -np.log(softmax(y_hat))
    return np.sum(p[range(m), y])

num_epochs, lr, wd = 10, 0.5, 0.001
def sgd(params, lr, wd, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size - param * wd

def accuracy(y_hat, y):
    if y_hat.shape[1] > 1:
        return float((y_hat.argmax(axis=1) == y.astype('float32')).sum())
    else:
        return float((y_hat.astype('int32') == y.astype('int32')).sum())

def evaluate_accuracy(net, data_iter):
    num_correct_example = 0
    total_num_example = 0

    for X, y in data_iter:
        num_correct_example += accuracy(net(X), y)
        total_num_example += y.size
    return num_correct_example / total_num_example

for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0
    num_examples = 0
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        sgd(params, lr, wd, X.shape[0])
        train_loss += l.sum()
        train_acc += accuracy(y_hat, y)
        num_examples += y.size
    # Return training loss and training accuracy
    train_loss, train_acc = train_loss/num_examples, train_acc/num_examples
    test_acc = evaluate_accuracy(net, test_iter)
    print("epoch {}: train loss {} train accuracy {} test accuracy {}".format(epoch, train_loss, train_acc, test_acc))
