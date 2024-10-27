import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

data = pd.read_csv('stock.csv', parse_dates=["Date"])

print(data.head())
original = data.iloc[5500:5808, [0, 1]]
# open = data["Open"][5002:5748]
open2 = data.iloc[5500:5746, [0, 1]]

print(open2)


# data process
open_prices = open2["Open"].values.astype(float).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
open_prices = scaler.fit_transform(open_prices)

# print("Open_prices", open_prices)

# spilit the dataset
train_size = int(len(open_prices) * 0.8)
train, test = open_prices[:train_size], open_prices[train_size:]

# create dataset
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:i + look_back]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape the data into [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# build and train the LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)

# predict data
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
# transfer trainY and testY into binary array then scale
trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY.reshape(-1, 1))

trainScore = np.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))
print('Train RMSE:', trainScore)
testScore = np.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
print('Test RMSE:', testScore)

# compare the original data and prediction
trainPredictPlot = np.empty_like(open_prices)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = np.empty_like(open_prices)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2):len(open_prices), :] = testPredict

# plot
plt.figure(figsize=(12, 8))
plt.plot(open2["Date"], scaler.inverse_transform(open_prices), label='Original Data')
plt.plot(open2["Date"], trainPredictPlot, label='Train Prediction')
plt.plot(open2["Date"], testPredictPlot, label='Test Prediction')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.gcf().autofmt_xdate()
plt.xlabel('Month')
plt.ylabel('Stock Open Price')
plt.title('Stock Open Price Prediction vs Original Data')
plt.legend()
plt.show()

# set future predicts days
future_days = 30

last_sequence = testX[-1]

future_predictions = []

# predict the future date
for _ in range(future_days):
    # use model to predict next day
    next_pred = model.predict(last_sequence.reshape(1, 1, look_back))

    # add predict result to future list
    future_predictions.append(next_pred[0, 0])

    # update last_sequence, and use it to be the next input
    last_sequence = np.append(last_sequence[:, 1:], next_pred, axis=1)

# turn to original scale
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

plt.figure(figsize=(12, 8))

# plot original data
# plt.plot(open2["Date"], scaler.inverse_transform(open_prices), label='Original Data')
plt.plot(original["Date"], original["Open"], label='Original Data')
future_dates = pd.date_range(open2["Date"].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
plt.plot(future_dates, future_predictions, label='Future Predictions', color='red')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.gcf().autofmt_xdate()
plt.xlabel('Month')
plt.ylabel('Stock Open Price')
plt.title('Stock Open Price with Future Predictions')
plt.legend()
plt.show()

