# Iris Prediction Model(Iris 품종 예측 모델)
# Model Fitting(데이터 탐색)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# 1. load training/teat data
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

dataset_file = './dataset/iris.csv'
df = pd.read_csv(dataset_file, names=['sepal length', 'sepal width', 'petal lenth', 'petal width', 'species'])

dataset = df.values
x, t = dataset[:, 0:4].astype(float), dataset[:, 4]

e = LabelEncoder()
e.fit(t)
t = e.transform(t)
t = tf.keras.utils.to_categorical(t)

# 2. model frame config
model = Sequential()
model.add(Dense(20, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. model fitting config
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
history = model.fit(x, t, epochs=50, batch_size=1, verbose=1)

# 5. result
loss = history.history['loss']
result = model.evaluate(x, t, verbose=0)
print(f'\n(Loss, Accuracy)=({result[0]}, {result[1]})')

# 6. predict
data = np.array([[7.2, 3.2, 6.0, 1.8]])     # from dataset
predict = model.predict(data)
index = np.argmax(predict)
species = ['Iris-setosa', 'Iris-virsicolor', 'Iris-virginica']
print(f'예측 되는 품종은 {species[index]} 입니다.')

# 7. graph
xlen = np.arange(len(loss))
plt.plot(xlen, loss, marker='.', c='blue', label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


