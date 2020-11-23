# 계단함수
# 총합을 구해서 1, 0으로 활성화

import os
import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
try:
    sys.path.append(os.path.join(Path(os.getcwd()), 'lib'))
    from common import step
except ImportError:
    print('Library Module Can Not Found')


x = np.arange(-5.0, 5.0, 0.1)
y = step(x)

plt.plot(x, y)
plt.show()


