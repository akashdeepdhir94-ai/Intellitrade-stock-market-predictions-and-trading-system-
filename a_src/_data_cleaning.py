import pandas as pd

# Load raw data
df = pd.read_csv("_data/AAPL.csv")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Remove missing values
df.dropna(inplace=True)

# Sort by Date
df.sort_values("Date", inplace=True)

# Save cleaned data
df.to_csv("_data/cleaning_AAPL.csv", index=False)

print("Cleaned data saved as cleaning_AAPL.csv")