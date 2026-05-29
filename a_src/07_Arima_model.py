import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("_data/cleaning_AAPL.csv")
series = df["Close"]

train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))

rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"ARIMA RMSE: {rmse:.2f}")

plt.figure(figsize=(12,6))
plt.plot(test.values, label="Actual")
plt.plot(forecast.values, label="Predicted")
plt.title("ARIMA Prediction")
plt.legend()
plt.show()