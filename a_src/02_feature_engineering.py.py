import pandas as pd
import ta

# Load cleaned data
df = pd.read_csv("_data/cleaning_AAPL.csv")

# Add Technical Indicators
df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
df["MACD"] = ta.trend.macd(df["Close"])

# Drop rows with NaN values after indicators
df.dropna(inplace=True)

# Save features file
df.to_csv("_data/02_features_engineering_AAPL.csv", index=False)

print(" Features saved as 02_features_engineering_AAPL.csv")