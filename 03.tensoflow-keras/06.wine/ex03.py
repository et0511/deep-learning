# Wine Binary Classification Model(와인 종류 분류 모델)
# model fitting #2 - model update(좋은 것만 저장)
import os
import pandas as pd
import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint


# 1. load training/test data
dataset_file = './dataset/wine.csv'
df = pd.read_csv(dataset_file, header=None)
df = df.sample(frac=1)

dataset = df.values
x = dataset[:, 0:12]
t = dataset[:, 12]

t = t[:, np.newaxis]
t = np.c_[t, t == 0]

# 2. model frame config
model = Sequential()
model.add(Dense(30, input_dim=x.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 3. model fitting config
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model check point config
model_directory = os.path.join(os.getcwd(), 'model')
if not os.path.exists(model_directory):
    os.mkdir(model_directory)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_directory, '{epoch:03d}-{val_loss: 4f}.h5'),
    minitor='val_loss',         # val_loss(시험셋 오차), loss(학습셋 오차), val_accuracy(시험셋 정확도), accuracy(학습셋 정확도)
    verbose=1,
    save_best_only=True
)

# 5. model fitting
model.fit(x, t, validation_split=0.2, epochs=200, batch_size=100, verbose=1, callbacks=[checkpoint])

# 6. result
result = model.evaluate(x, t, verbose=0)
print(f'\n(Loss, Accuracy)=({result[0]}, {result[1]})')

# 7. predict
data = np.array([[8.5, 0.21, 0.26, 9.25, 0.034, 73, 142, 0.9945, 3.05, 0.37, 11.4, 6]])
predict = model.predict(data)
index = np.argmax(predict)
wines = ['Red Wine', 'White Wine']
print(wines[index])
