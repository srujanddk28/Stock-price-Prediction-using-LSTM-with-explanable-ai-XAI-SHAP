"""Train and evaluate an LSTM stock-price prediction model.

Example:
    python src/train.py --ticker AAPL --period 5y
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
WINDOW = 30
SEED = 42


def download_close(ticker: str, period: str) -> pd.Series:
    """Download daily prices, falling back to Stooq if Yahoo Finance is unavailable."""
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if not data.empty:
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna().astype("float32")
            if len(close) > WINDOW + 30:
                print("Data source: Yahoo Finance")
                return close
    except Exception as exc:
        print(f"Yahoo Finance download failed: {exc}")

    # Stooq is a free fallback and avoids Yahoo's browser-impersonation dependency.
    stooq_symbol = f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    try:
        data = pd.read_csv(url)
        if data.empty or "Close" not in data.columns:
            raise ValueError("Stooq returned no usable price data.")
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date").set_index("Date")
        close = data["Close"].dropna().astype("float32")
        if period.endswith("y"):
            years = int(period[:-1])
            close = close.loc[close.index >= (pd.Timestamp.today() - pd.DateOffset(years=years))]
        elif period.endswith("mo"):
            months = int(period[:-2])
            close = close.loc[close.index >= (pd.Timestamp.today() - pd.DateOffset(months=months))]
        if len(close) <= WINDOW + 30:
            raise ValueError("Stooq did not return enough history for this period.")
        print(f"Data source: Stooq ({stooq_symbol})")
        return close
    except Exception as exc:
        raise ValueError(
            f"Could not download {ticker} prices from Yahoo Finance or Stooq. "
            f"Check the ticker and network. Stooq error: {exc}"
        ) from exc


def make_sequences(values: np.ndarray, window: int = WINDOW):
    """Create rolling windows: previous window days -> next day's price."""
    x_values, y_values = [], []
    for index in range(window, len(values)):
        x_values.append(values[index - window:index, 0])
        y_values.append(values[index, 0])
    x_array = np.asarray(x_values, dtype="float32").reshape(-1, window, 1)
    y_array = np.asarray(y_values, dtype="float32")
    return x_array, y_array


def build_model(window: int = WINDOW) -> Sequential:
    """Build a small stacked LSTM for one-step forecasting."""
    model = Sequential([
        tf.keras.Input(shape=(window, 1)),
        LSTM(48, return_sequences=True),
        Dropout(0.2),
        LSTM(24),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def save_plots(history, dates, actual, predicted, ticker: str):
    """Save prediction comparison and training-loss charts."""
    RESULTS.mkdir(exist_ok=True)

    plt.figure(figsize=(11, 5))
    plt.plot(dates, actual, label="Actual close", linewidth=1.5)
    plt.plot(dates, predicted, label="LSTM prediction", linewidth=1.2)
    plt.title(f"{ticker} - Actual vs Predicted Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Adjusted close price")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / f"{ticker.lower()}_actual_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Training loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title("Model Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / f"{ticker.lower()}_training_loss.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train an LSTM stock-price demo.")
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker, e.g. AAPL or MSFT")
    parser.add_argument("--period", default="5y", help="Yahoo Finance period, e.g. 1y, 2y, 5y, max")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    ARTIFACTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    close = download_close(ticker, args.period)
    split = int(len(close) * 0.8)

    # Fit preprocessing only on the training period to reduce data leakage.
    scaler = MinMaxScaler()
    scaler.fit(close.iloc[:split].to_numpy().reshape(-1, 1))
    scaled_values = scaler.transform(close.to_numpy().reshape(-1, 1))

    x_all, y_all = make_sequences(scaled_values)
    split_sequence = split - WINDOW
    x_train, y_train = x_all[:split_sequence], y_all[:split_sequence]
    x_test, y_test = x_all[split_sequence:], y_all[split_sequence:]

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("The selected period did not produce enough train/test windows.")

    model = build_model()
    early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    history = model.fit(
        x_train, y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        shuffle=False,
        verbose=1,
    )

    predicted_scaled = model.predict(x_test, verbose=0)
    predicted = scaler.inverse_transform(predicted_scaled).ravel()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Persistence baseline: estimate the next close using the latest observed close.
    baseline_scaled = x_test[:, -1, 0].reshape(-1, 1)
    baseline = scaler.inverse_transform(baseline_scaled).ravel()

    lstm_mae = float(mean_absolute_error(actual, predicted))
    lstm_rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    baseline_mae = float(mean_absolute_error(actual, baseline))
    baseline_rmse = float(np.sqrt(mean_squared_error(actual, baseline)))

    test_dates = close.index[split:]
    predictions = pd.DataFrame({
        "date": test_dates.strftime("%Y-%m-%d"),
        "actual_close": actual,
        "lstm_prediction": predicted,
        "previous_day_baseline": baseline,
    })
    predictions.to_csv(RESULTS / f"{ticker.lower()}_predictions.csv", index=False)

    metrics = {
        "ticker": ticker,
        "period_requested": args.period,
        "data_points": int(len(close)),
        "train_points": int(split),
        "test_points": int(len(actual)),
        "lookback_days": WINDOW,
        "lstm_mae": lstm_mae,
        "lstm_rmse": lstm_rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Historical test-set evaluation; not a guarantee of future performance.",
    }
    with (RESULTS / f"{ticker.lower()}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    model.save(ARTIFACTS / f"{ticker.lower()}_lstm.keras")
    joblib.dump(scaler, ARTIFACTS / f"{ticker.lower()}_scaler.joblib")
    save_plots(history, test_dates, actual, predicted, ticker)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Ticker:              {ticker}")
    print(f"Training samples:    {len(x_train)}")
    print(f"Testing samples:     {len(x_test)}")
    print(f"LSTM MAE:            {lstm_mae:.4f}")
    print(f"LSTM RMSE:           {lstm_rmse:.4f}")
    print(f"Baseline MAE:        {baseline_mae:.4f}")
    print(f"Baseline RMSE:       {baseline_rmse:.4f}")
    print(f"Saved predictions:   results/{ticker.lower()}_predictions.csv")
    print(f"Saved metrics:       results/{ticker.lower()}_metrics.json")
    print(f"Saved charts:        results/{ticker.lower()}_*.png")
    print(f"Saved model:         artifacts/{ticker.lower()}_lstm.keras")
    print("\nNote: Results depend on the ticker and the date you run the project.")


if __name__ == "__main__":
    main()
