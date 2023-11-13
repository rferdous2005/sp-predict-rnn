import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from process import *
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_df = collectDataForASymbol(2016, 2022, "KOHINOOR", scaler)
time_steps = 60
advance_steps = 10
X_train = []
y_train = []

print(scaled_df.loc[59])
for i in range(time_steps, len(scaled_df)):
    X_train.append(scaled_df.loc[i-time_steps: i-1, "Close"])
    y_train.append(scaled_df.loc[i, "Close"])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.1))

regressor.add(Dense(units=50))

#regressor.compile(optimizer="adam", loss="mean_squared_error")
#regressor.fit(X_train, y_train, epochs=100, batch_size=64)
print(X_train, y_train)


# Getting the real stock price of 2017
# dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
past30_stock_price = scaled_df.iloc[len(scaled_df)-1-30:len(scaled_df)-1, "Close"].values

# Getting the predicted stock price of 2017
# dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = scaled_df.loc[len(scaled_df)-1-time_steps:len(scaled_df)-1, "Close"].values
print(inputs)
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
X_test.append(inputs.loc[len(scaled_df)-1-time_steps:len(scaled_df)-1, "Close"])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# # Visualising the results
plt.plot(past30_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()