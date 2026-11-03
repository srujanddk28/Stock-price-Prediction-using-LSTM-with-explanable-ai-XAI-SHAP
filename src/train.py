"""Train a small LSTM on historical closing prices.

Example:
    python src/train.py --ticker AAPL --period 5y
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
WINDOW = 30


def download_close(ticker: str, period: str) -> pd.Series:
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No market data returned for {ticker}. Check the ticker and internet connection.")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype("float32")
    if len(close) <= WINDOW + 20:
        raise ValueError("Not enough price history. Choose a longer period.")
    return close


def make_sequences(values: np.ndarray, window: int):
    x, y = [], []
    for i in range(window, len(values)):
        x.append(values[i - window:i, 0])
        y.append(values[i, 0])
    return np.asarray(x, dtype="float32").reshape(-1, window, 1), np.asarray(y, dtype="float32")


def build_model(window: int) -> Sequential:
    model = Sequential([
        LSTM(48, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(24),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train an LSTM stock-price demo.")
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker, e.g. AAPL or MSFT")
    parser.add_argument("--period", default="5y", help="Yahoo Finance period, e.g. 1y, 2y, 5y, max")
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    close = download_close(args.ticker.upper(), args.period)
    split = int(len(close) * 0.8)

    # Fit the scaler only on the training section to avoid leaking test information.
    scaler = MinMaxScaler()
    train_values = scaler.fit_transform(close.iloc[:split].to_numpy().reshape(-1, 1))
    all_values = scaler.transform(close.to_numpy().reshape(-1, 1))

    x_all, y_all = make_sequences(all_values, WINDOW)
    split_sequence = split - WINDOW
    x_train, y_train = x_all[:split_sequence], y_all[:split_sequence]
    x_test, y_test = x_all[split_sequence:], y_all[split_sequence:]

    model = build_model(WINDOW)
    stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    model.fit(
        x_train, y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.1,
        callbacks=[stop],
        shuffle=False,
        verbose=1,
    )

    predicted_scaled = model.predict(x_test, verbose=0)
    predicted = scaler.inverse_transform(predicted_scaled).ravel()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"Ticker: {args.ticker.upper()}")
    print(f"Test MAE:  {mae:.3f}")
    print(f"Test RMSE: {rmse:.3f}")

    model.save(ARTIFACTS / f"{args.ticker.lower()}_lstm.keras")
    joblib.dump(scaler, ARTIFACTS / f"{args.ticker.lower()}_scaler.joblib")

    test_dates = close.index[split:]
    plt.figure(figsize=(11, 5))
    plt.plot(test_dates, actual, label="Actual close")
    plt.plot(test_dates, predicted, label="LSTM prediction")
    plt.title(f"{args.ticker.upper()} — actual vs predicted closing price")
    plt.xlabel("Date")
    plt.ylabel("Adjusted close price")
    plt.legend()
    plt.tight_layout()
    chart_path = ARTIFACTS / f"{args.ticker.lower()}_prediction.png"
    plt.savefig(chart_path, dpi=150)
    print(f"Saved model, scaler and chart in: {ARTIFACTS}")


if __name__ == "__main__":
    main()
