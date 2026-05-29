import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("_data/signals_AAPL.csv")

# Optimize
df = df.tail(200)
df["Date"] = pd.to_datetime(df["Date"])

buy = df[df["Signal"] == 1]
sell = df[df["Signal"] == -1]

plt.style.use("fast")

plt.figure(figsize=(14,6))
plt.plot(df["Date"], df["Close"], label="Close Price")

plt.scatter(buy["Date"], buy["Close"], marker="^", label="Buy", s=50)
plt.scatter(sell["Date"], sell["Close"], marker="v", label="Sell", s=50)

plt.title("Buy/Sell Signals")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()