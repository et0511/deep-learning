# MNIST(Modified National Institude of Standard and Technology)
# MNIST 손글씨 숫자 분류 신경망(Neural Network for MNIST Handwritten Digit): 데이터 살펴보기

import os
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import init_network, load_mnist
except ImportError:
    print('Library Module Can Not Found')

# 1. 매개변수(w, b) 데이터 셋 가져오기
network = init_network()
w1, w2, w3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

print(w1.shape)     # 784 x 50 Matrix
print(w2.shape)     # 50 x 100 Matrix
print(w3.shape)     # 100 x 10 Matrix

print(b1.shape)     # 50 Vector
print(b2.shape)     # 100 Vector
print(b3.shape)     # 10 Vector


# 2. 학습/시험 데이터 가져오기
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

xlen = len(test_x)
randidx = np.random.randint(0, xlen, 1).reshape(())

# 3. 신호전달
print('\n== 신호전달 구현1: 은닉 1층 전달 ====================')

x = test_x[randidx]
print(f'x dimension: {x.shape}')        # 784 vector

w1 = network['W1']
print(f'w1 dimension: {w1.shape}')      # 784 x 50 matrix

b1 = network['b1']
print(f'b1 dimension: {b1.shape}')      # 50 vector


