import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("_data/02_features_engineering_AAPL.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# Sort by Date and take last 300 rows
df = df.sort_values("Date")
df = df.tail(300)

plt.figure(figsize=(12,6))

plt.plot(df["Date"], df["Close"], label="Close Price", linewidth=1)
plt.plot(df["Date"], df["SMA_20"], label="SMA 20", linewidth=1)
plt.plot(df["Date"], df["SMA_50"], label="SMA 50", linewidth=1)

plt.title("AAPL Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()