# 신경망학습: 신경망에서의 기울기
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, relu, sigmoid, cross_entropy_error
except ImportError:
    print('Library Module Can Not Found')


# params = {
#     'w1': np.array([[0.02, 0.224, 0.135], [0.01, 0.052, 0.345]]),
#     'b1': np.array([0.45, 0.23, 0.11])
# }
params = dict()

def initialize(input_size, hidden_size, output_size, init_weight=0.01):
    params['w1'] = init_weight * np.random.randn(input_size, hidden_size)
    params['b1'] = np.zeros(hidden_size)
    params['w2'] = init_weight * np.random.randn(hidden_size, output_size)
    params['b2'] = np.zeros(output_size)


def forward_propagation(x):
    w1 = params['w1']
    b1 = params['b1']
    a1 = np.dot(x, w1) + b1

    # z1 = sigmoid(a1)
    z1 = relu(a1)

    w2 = params['w2']
    b2 = params['b2']
    a2 = np.dot(z1, w2) + b2

    y = softmax(a2)

    return y


def loss(x, t):
    y = forward_propagation(x)
    e = cross_entropy_error(y, t)
    return e


def accuracy(x, t):
    y = forward_propagation(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    acc = np.sum(y == t) / float(x.shape[0])
    return acc


def numerical_gradient_net(x, t):
    h = 1e-4
    gradient = dict()

    for key in params:
        param = params[key]
        param_gradient = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            temp = param[idx]

            param[idx] = temp + h
            h1 = loss(x, t)

            param[idx] = temp - h
            h2 = loss(x, t)

            param_gradient[idx] = (h1 - h2) / (2 * h)

            param[idx] = temp  # 꼭 기억!!! 값 복원!!!
            it.iternext()

        gradient[key] = param_gradient

    return gradient

