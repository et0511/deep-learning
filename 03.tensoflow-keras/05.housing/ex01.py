# Housing price prediction Model(보스턴 집값 예측 모델)
# Explore Dataset(데이터 탐색)

import pandas as pd

dataset_file = './dataset/housing.csv'
df = pd.read_csv(dataset_file, header=None)
print(df.info())
print(df.head())






