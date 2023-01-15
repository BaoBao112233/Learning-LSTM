import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ file
body_df = pd.read_csv("SWING.txt")
hand_df = pd.read_csv("HANDSWING.txt")

# Ghép 10 fram liên tiếp (Time Step)
X = []
Y = []
num_of_ts = 10

ds = body_df.iloc[:,1:].values
n_sample = len(ds)
# Lấy 10 timestep 1 lần để tạo thành 1 input
for i in range(num_of_ts, n_sample):
    X.append(ds[i-num_of_ts:i,:])
    # Quy định 1 là Body Swing (Lắc thân)
    Y.append(1)

ds = hand_df.iloc[:,1:].values
n_sample = len(ds)
# Lấy 10 timestep 1 lần để tạo thành 1 input
for i in range(num_of_ts, n_sample):
    X.append(ds[i-num_of_ts:i,:])
    # Quy định 0 làHand Swing (Lắc tay)
    Y.append(0)

X, Y = np.array(X), np.array(Y)
# print(X.shape , " - " , Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = Sequential()
model.add(LSTM(50,return_sequences = True,input_shape=(num_of_ts,X.shape[2])))
# return_sequences = True được dùng để lớp sau nhận được các output timestep trước
model.add(Dropout(0.2))
# Tránh bị Overfit
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation="sigmoid"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

model.fit(X_train, Y_train, epochs=16, batch_size=32,validation_data=(X_test, Y_test))
model.save("model_LSTM.h5")

