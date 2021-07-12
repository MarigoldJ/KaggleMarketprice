#-*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ProgbarLogger
import sys


os.chdir(os.path.dirname(os.path.realpath(__file__)))

# log 기록 시작
sys.stdout = open('./data/training_log3.txt', 'w')

#seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

data = pd.read_csv("./data/data_frame_v2.csv", index_col=None)

x_train = data[data.data_block_num <= 33].values[:, :10]
y_train = data[data.data_block_num <= 33].values[:, 10]

# x_test = data[data.data_block_num == 33].values[:, :10]
# y_test = data[data.data_block_num == 33].values[:, 10]



scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scale = scaler.transform(x_train)
# x_test_scale = scaler.transform(x_test)

model = Sequential()
model.add(Dense(32, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='sigmoid'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer='adam')

now = pd.datetime.now().strftime('%Y%m%d_%H%M')
model_path = f'./Model_check_point/final_checkpoint_{now}.ckpt'
es = EarlyStopping(patience=5, monitor='loss')
mc = ModelCheckpoint(filepath=model_path, \
                    monitor='val_loss',
                    save_weights_only=True,
                    verbose=1
                    )

model.fit(x_train_scale, y_train, epochs=50, batch_size=128, callbacks=[es, mc], verbose=2)

# 예측 값과 실제 값의 비교
# y_pred = model.predict(x_test_scale).flatten()
# rmse = (((y_test - y_pred) ** 2).mean()) ** 0.5
# print(rmse)
print("Mission Complete!!!")

# log 기록 종료
sys.stdout.close()