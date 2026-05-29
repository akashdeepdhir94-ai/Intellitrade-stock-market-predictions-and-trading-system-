import pandas as pd
import numpy as np

df = pd.read_csv("_data/signals_AAPL.csv")

initial_capital = 100000
cash = initial_capital
position = 0
transaction_cost = 0.001  # 0.1%
stop_loss_pct = 0.05      # 5%

buy_price = 0
portfolio_values = []

for i in range(len(df)):
    price = df.loc[i, "Close"]
    signal = df.loc[i, "Signal"]

    # BUY
    if signal == 1 and cash > 0:
        position = (cash * (1 - transaction_cost)) / price
        buy_price = price
        cash = 0

    # STOP LOSS
    elif position > 0 and price < buy_price * (1 - stop_loss_pct):
        cash = position * price * (1 - transaction_cost)
        position = 0

    # SELL
    elif signal == -1 and position > 0:
        cash = position * price * (1 - transaction_cost)
        position = 0

    # Track portfolio
    value = cash if cash > 0 else position * price
    portfolio_values.append(value)

final_value = portfolio_values[-1]
returns = ((final_value - initial_capital) / initial_capital) * 100

print(f"Initial Capital: ₹{initial_capital}")
print(f"Final Value: ₹{final_value:.2f}")
print(f"Return: {returns:.2f}%")
# Convert to pandas series
portfolio_series = pd.Series(portfolio_values)

# Daily returns
returns_series = portfolio_series.pct_change().dropna()

#  Sharpe Ratio
sharpe_ratio = (returns_series.mean() / returns_series.std()) * np.sqrt(252)

#  Drawdown
rolling_max = portfolio_series.cummax()
drawdown = (portfolio_series - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")