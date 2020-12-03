# Sonar Mineral Binary Classification Model(초음파 광물 예측 모델)
# Explore Dataset(데이터 참색)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

dataset_file = './dataset/sonar.csv'
df = pd.read_csv(dataset_file, header=None)
print(df.info())
print(df.head())



