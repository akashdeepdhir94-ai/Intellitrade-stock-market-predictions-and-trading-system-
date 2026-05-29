import pandas as pd

df = pd.read_csv("_data/02_features_engineering_AAPL.csv")

# Generate Buy/Sell signals
df["Signal"] = 0
df.loc[df["SMA_20"] > df["SMA_50"], "Signal"] = 1
df.loc[df["SMA_20"] < df["SMA_50"], "Signal"] = -1

# Save signals
df.to_csv("_data/signals_AAPL.csv", index=False)

print("Signals saved as signals_AAPL.csv")