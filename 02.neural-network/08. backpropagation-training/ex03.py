# Training Neural Network
# Data Set: MNIST Handwritten Digit Dataset
# Network: TwoLayerNet2
# Test: backpropagation Gradient vs Numerical Gradient

import os
import sys
import time
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import multilayernet as network
except ImportError:
    print('Library Module Can Not Found')

# 1. load training / test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. initialize network
network.initialize(input_size=train_x.shape[1], hidden_size=[50], output_size=train_t.shape[1])

# 3. batch by 3
train_x_batch = train_x[:3]
train_t_batch = train_t[:3]

# 4. Gradient
gradient_numerical = network.numerical_gradient_net(train_x_batch, train_t_batch)
gradient_backpropagation = network.backpropagation_gradient_net(train_x_batch, train_t_batch)

# mean of mudules
for key in gradient_numerical:
    diff = np.average(np.abs((gradient_numerical[key] - gradient_backpropagation[key])))
    print(f'{key} difference: {diff}')

# 6. 결론: 거의 차이 없음!
# w1 difference: 3.878692123191609e-10
# b1 difference: 2.062824756347251e-09
# w2 difference: 5.088050640802983e-09
# b2 difference: 1.404991117739951e-07





