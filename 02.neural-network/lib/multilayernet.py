import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import ReLU, Affine, SoftmaxWithLoss
except ImportError:
    print('Library Module Can Not Found')


params = dict()
layers = []


def initialize(input_size, hidden_size, output_size, init_weight=0.01, init_params=None):
    hidden_count = len(hidden_size)
    if init_params is None:
        params['w1'] = init_weight * np.random.randn(input_size, hidden_size[0])
        params['b1'] = np.zeros(hidden_size[0])
        for idx in range(1, hidden_count):
            params[f'w{idx+1}'] = init_weight * np.random.randn(hidden_size[idx-1], hidden_size[idx])
            params[f'b{idx+1}'] = np.zeros(hidden_size[idx])
        params[f'w{hidden_count+1}'] = init_weight * np.random.randn(hidden_size[hidden_count-1], output_size)
        params[f'b{hidden_count+1}'] = np.zeros(output_size)
    else:
        globals()['params'] = init_params

    layers.append(Affine(params['w1'], params['b1']))
    layers.append(ReLU())
    for idx in range(1, hidden_count):
        layers.append(Affine(params[f'w{idx+1}'], params[f'b{idx+1}']))
        layers.append(ReLU())
    layers.append(Affine(params[f'w{hidden_count+1}'], params[f'b{hidden_count+1}']))
    layers.append(SoftmaxWithLoss())


def forward_propagation(x, t=None):
    for layer in layers:
        x = layer.forward(x, t) if type(layer).__name__ == 'SoftmaxWithLoss' and t is not None else layer.forward(x)
#            if t is None:
#                x = layer.forward(x)
#            else:
#                x = layer.forward(x, t)
#        else:
#            x = layer.forward(x)
    return x


def backward_propagation(dout):
    for layer in layers[::-1]:
        dout = layer.backward(dout)
    return dout


def loss(x, t):
    y = forward_propagation(x, t)
    return y


def accuracy(x, t):
    y = forward_propagation(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    acc = np.sum(y == t) / float(x.shape[0])
    return acc


def backpropagation_gradient_net(x, t):
    forward_propagation(x, t)
    backward_propagation(1)

    idxaffine = 0
    gradient = dict()

    for layer in layers:
        if type(layer).__name__ == 'Affine':
            idxaffine += 1
            gradient[f'w{idxaffine}'] = layer.dw
            gradient[f'b{idxaffine}'] = layer.db

    return gradient


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

