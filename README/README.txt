# 📈 IntelliTradeAI — Stock Market Prediction and Trading system project 

An AI-powered stock market analysis and prediction system built 
during internship. Analyzes Apple (AAPL) stock using multiple ML 
models and displays results on an interactive Streamlit dashboard.

## Features
-  Data Cleaning & Feature Engineering
-  Technical Indicators (RSI, MACD, Moving Averages)
-  Buy/Sell Signal Generation
-  Backtesting Strategy
-  ARIMA Model (Statistical Forecasting)
-  Prophet Model (Facebook's Time Series)
-  LSTM Model (Deep Learning)
-  Model Comparison Dashboard
-  Interactive Streamlit Web Dashboard

## 🗂️ Project Structure
IntelliTradeAI/
├── a_src/
│   ├── _data_cleaning.py
│   ├── 02_feature_engineering.py
│   ├── 03_visualization.py
│   ├── 04_strategy.py
│   ├── 05_signal_plot.py
│   ├── 06_backtest.py
│   ├── 07_Arima_model.py
│   ├── 08_prophet_model.py
│   ├── Lstm_model.py
│   └── model_comparison.py
├── _data/
│   ├── AAPL.csv
│   └── ...processed CSVs...
└── dashboard/
└── dashboard.py

## ⚙️ Installation
```bash
pip install pandas numpy matplotlib streamlit ta statsmodels prophet scikit-learn tensorflow
```

## ▶️ How to Run

### Step 1 — Run pipeline scripts in order:
```bash
python a_src/_data_cleaning.py
python a_src/02_feature_engineering.py
python a_src/04_strategy.py
python a_src/model_comparison.py
```

### Step 2 — Launch Dashboard:
```bash
streamlit run dashboard/dashboard.py
```

## 🛠️ Tech Stack
- Python 3
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras (LSTM)
- Facebook Prophet
- Statsmodels (ARIMA)
- Streamlit (Dashboard)
- Matplotlib

## 👨‍💻 Author
akashdeepdhir
