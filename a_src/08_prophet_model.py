import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
df = pd.read_csv("_data/cleaning_AAPL.csv")
df = df[["Date", "Close"]]
df.columns = ["ds", "y"]

train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

model = Prophet()
model.fit(train)

future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

pred = forecast["yhat"].tail(len(test)).values
actual = test["y"].values

rmse = np.sqrt(mean_squared_error(actual, pred))
print(f"Prophet RMSE: {rmse:.2f}")

model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()