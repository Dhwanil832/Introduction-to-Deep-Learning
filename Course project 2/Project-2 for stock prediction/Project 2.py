import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 讀取數據
data_path = 'TSMC stock price.csv'
df = pd.read_csv(data_path)

# 確保 'Date' 列為日期格式，並篩選2016年的數據
df['Date'] = pd.to_datetime(df['Date'])
df_2016 = df[(df['Date'] >= '2016-01-01') & (df['Date'] <= '2016-12-31')]

# 取2016年1月到11月的數據作為訓練集，12月作為測試集
train_df = df_2016[df_2016['Date'] < '2016-12-01']
test_df = df_2016[df_2016['Date'] >= '2016-12-01']

# 選擇需要的特徵，例如 'Open', 'High', 'Low', 'Close'
train_data = train_df[['Open', 'High', 'Low', 'Close']]
test_data = test_df[['Open', 'High', 'Low', 'Close']]

# 進行數據縮放 (0,1) 之間
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

# 創建訓練數據集
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), :])
        Y.append(dataset[i + time_step, 3])  # 預測 'Close' 列
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(scaled_train_data, time_step)

# 重塑輸入數據為 LSTM 所需的形狀 [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)

# LSTM 模型構建
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 4)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))  # 預測 'Close' 列

# 編譯模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 訓練模型
model.fit(X_train, y_train, batch_size=64, epochs=10)

# 創建12月份數據的輸入 (使用最後60天的數據進行預測)
X_test = []
input_data = np.concatenate((scaled_train_data[-time_step:], scaled_test_data), axis=0)

for i in range(time_step, len(input_data)):
    X_test.append(input_data[i-time_step:i, :])

X_test = np.array(X_test).reshape(-1, time_step, 4)

# 使用模型預測12月份的股價
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 3)), predictions), axis=1))[:, 3]

# 繪製圖表，展示三條線：1-11月的實際數據，預測的12月數據，和實際的12月數據
plt.figure(figsize=(16,8))

# 1-11月的實際股價
plt.plot(train_df['Date'], train_df['Close'], label='Real Close Price (Jan-Nov 2016)')

# 預測的12月股價
plt.plot(test_df['Date'], predictions, label='Predicted Close Price (Dec 2016)')

# 實際的12月股價
plt.plot(test_df['Date'], test_df['Close'], label='Real Close Price (Dec 2016)')

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('TSMC Stock Price Prediction for December 2016')
plt.legend()
plt.show()
