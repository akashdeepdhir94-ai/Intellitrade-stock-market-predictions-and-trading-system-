import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="IntelliTradeAI", layout="wide")

# TITLE
st.title("📈 IntelliTradeAI - Stock Market Dashboard")
st.markdown("### Apple (AAPL) Stock Analysis")

# LOAD DATA
@st.cache_data
def load_data():

    cleaned_df = pd.read_csv("_data/cleaning_AAPL.csv")
    features_df = pd.read_csv("_data/02_features_engineering_AAPL.csv")
    signals_df = pd.read_csv("_data/signals_AAPL.csv")
    comparison_df = pd.read_csv("_data/model_comparison.csv")

    # Convert Date columns
    for df in [cleaned_df, features_df, signals_df]:
        df["Date"] = pd.to_datetime(df["Date"])

    return cleaned_df, features_df, signals_df, comparison_df


cleaned_df, features_df, signals_df, comparison_df = load_data()

# SIDEBAR
option = st.sidebar.selectbox(
    "Choose Section",
    [
        "Data Preview",
        "Price Chart",
        "Technical Indicators",
        "Buy/Sell Signals",
        "ARIMA Prediction",
        "Prophet Prediction",
        "LSTM Prediction",
        "Model Comparison"
    ]
)




# 1. DATA PREVIEW
if option == "Data Preview":

    st.header("1. Cleaned Data Preview")

    st.dataframe(cleaned_df.head(50))


# 2. PRICE CHART
elif option == "Price Chart":

    st.header("2. Closing Price Chart")

    st.line_chart(
        cleaned_df.set_index("Date")["Close"]
    )


# 3. TECHNICAL INDICATORS
elif option == "Technical Indicators":

    st.header("3. Technical Indicators")

    st.line_chart(
        features_df.set_index("Date")[["Close", "SMA_20", "SMA_50"]]
    )


# 4. BUY / SELL SIGNALS
elif option == "Buy/Sell Signals":

    st.header("4. Buy / Sell Signals")

    st.line_chart(
        signals_df.set_index("Date")["Close"]
    )

    buy = signals_df[signals_df["Signal"] == 1]
    sell = signals_df[signals_df["Signal"] == -1]

    st.subheader("Buy Signals")
    st.dataframe(buy[["Date", "Close"]])

    st.subheader("Sell Signals")
    st.dataframe(sell[["Date", "Close"]])


# 5. ARIMA PREDICTION
elif option == "ARIMA Prediction":

    st.header("5. ARIMA Prediction")

    df = cleaned_df.copy()

    series = df["Close"]

    train_size = int(len(series) * 0.8)

    train = series[:train_size]
    test = series[train_size:]

    model = ARIMA(train, order=(5,1,0))

    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test))

    rmse = np.sqrt(mean_squared_error(test, forecast))

    st.subheader(f"ARIMA RMSE: {rmse:.2f}")

    chart_df = pd.DataFrame({
        "Actual": test.values,
        "Predicted": forecast.values
    })

    st.line_chart(chart_df)


# 6. PROPHET PREDICTION
elif option == "Prophet Prediction":

    st.header("6. Prophet Prediction")

    df = cleaned_df.copy()

    prophet_df = df[["Date", "Close"]]

    prophet_df.columns = ["ds", "y"]

    train_size = int(len(prophet_df) * 0.8)

    train = prophet_df[:train_size]
    test = prophet_df[train_size:]

    model = Prophet()

    model.fit(train)

    future = model.make_future_dataframe(periods=len(test))

    forecast = model.predict(future)

    pred = forecast["yhat"].tail(len(test)).values

    actual = test["y"].values

    rmse = np.sqrt(mean_squared_error(actual, pred))

    st.subheader(f"Prophet RMSE: {rmse:.2f}")

    chart_df = pd.DataFrame({
        "Actual": actual,
        "Predicted": pred
    })

    st.line_chart(chart_df)


# 7. LSTM PREDICTION
elif option == "LSTM Prediction":

    st.header("7. LSTM Prediction")

    df = cleaned_df.copy()

    data = df["Close"].values.reshape(-1,1)

    # SCALE DATA
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    # CREATE SEQUENCES
    X, y = [], []

    sequence_length = 50

    for i in range(sequence_length, len(scaled_data)):

        X.append(scaled_data[i-sequence_length:i, 0])

        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # RESHAPE
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # TRAIN TEST SPLIT
    train_size = int(len(X) * 0.8)

    X_train = X[:train_size]
    X_test = X[train_size:]

    y_train = y[:train_size]
    y_test = y[train_size:]

    # BUILD MODEL
    model = Sequential()

    model.add(
        LSTM(
            50,
            return_sequences=False,
            input_shape=(X_train.shape[1], 1)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    # TRAIN MODEL
    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        verbose=0
    )

    # PREDICTION
    predictions = model.predict(X_test)

    predictions = scaler.inverse_transform(
        predictions.reshape(-1,1)
    )

    y_test_actual = scaler.inverse_transform(
        y_test.reshape(-1,1)
    )

    rmse = np.sqrt(
        mean_squared_error(y_test_actual, predictions)
    )

    st.subheader(f"LSTM RMSE: {rmse:.2f}")

    chart_df = pd.DataFrame({
        "Actual": y_test_actual.flatten(),
        "Predicted": predictions.flatten()
    })

    st.line_chart(chart_df)


# 8. MODEL COMPARISON
elif option == "Model Comparison":

    st.header("8. Model Comparison")

    st.dataframe(comparison_df)

    st.bar_chart(
        comparison_df.set_index("Model")["RMSE"]
    )


# FOOTER
st.success("IntelliTradeAI Dashboard Loaded Successfully!")