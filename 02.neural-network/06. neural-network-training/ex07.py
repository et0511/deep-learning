# Training Neural Network
# Data Set: MNIST Handwritten Digit Dataset
# Network: TwoLayerNet
# Estimation: Training Accuracy

import os
import pickle
import sys
from pathlib import Path
from matplotlib import pyplot as plt
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet as network
except ImportError:
    print('Library Module Can Not Found')


# 1. load train/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. load params dataset trained
trainacc_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_train_accuracy.pkl')
testacc_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_test_accuracy.pkl')

train_accuracies = None
test_accuracies = None

with open(trainacc_file, 'rb') as f_trainacc, open(testacc_file, 'rb') as f_testacc:
    train_accuracies = pickle.load(f_trainacc)
    test_accuracies = pickle.load(f_testacc)


plt.plot(train_accuracies, label='train accuracy')
plt.plot(test_accuracies, label='test accuracy')

plt.xlim(0, 20, 1)
plt.ylim(0., 1., 0.5)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.show()


# train_accuracy와 test_accuracy가 일치하는 것은 Overfiiting이 발생하지 않았다는 것이다.
# 학습 중에 1 epoch 별로 train/test accuracy를 각각 기록하여 추이를 비교하여야 한다.
# 두 accuracy가 마지막까지 차이가 없는 것이 가장 바람직 하며,
# 만약 차이가 나는 그 시점에서 찾아서 학습을 중지 하여야 한다. - early stopping
#

