# MNIST(Modified National Institude of Standard and Technology)
# MNIST 손글씨 숫자 분류 신경망(Neural Network for MNIST Handwritten Digit): 데이터 살펴보기

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
except ImportError:
    print('Library Module Can Not Found')


(train_x, train_t), (test_x, test_t) = load_mnist(normalize=False, flatten=True, one_hot_label=False)
print(train_x.shape)    # 60,000 x 784 (Matrix)
print(train_t.shape)    # 60,000 (Vector)

t = train_t[0]
print(t)                # 5

x = train_x[0]
print(x, x.shape)          # 784
x = x.reshape(28, 28)       # 형사을 원래 이미지 크기로 변경
print(x.shape)          # 28 x 28

# 이미지 보기: PIL(Python Image Library) 사용
pil_Image = Image.fromarray(x)
pil_Image.show()



