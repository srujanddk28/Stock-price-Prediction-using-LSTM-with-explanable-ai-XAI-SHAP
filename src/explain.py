"""Explain recent LSTM predictions with SHAP KernelExplainer.

Run after training:
    python src/explain.py --ticker AAPL

KernelExplainer is model-agnostic. We flatten each 30-day input window into
30 lag features, then reshape it before passing it to the saved LSTM model.
This is a small educational example; SHAP can take several minutes to run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import json
import urllib.parse
import urllib.request
from sklearn.preprocessing import MinMaxScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
WINDOW = 30


def load_recent_windows(ticker: str, period: str, scaler):
    """Load adjusted daily prices using Yahoo's chart API."""
    errors = []
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{urllib.parse.quote(ticker)}?range={period}&interval=1d&events=history"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            chart = payload.get("chart", {})
            if chart.get("error") or not chart.get("result"):
                raise ValueError(str(chart.get("error") or "Yahoo returned no chart data"))
            result = chart["result"][0]
            quote = result["indicators"]["quote"][0]
            values = result["indicators"].get("adjclose", [{}])[0].get("adjclose") or quote.get("close")
            close = pd.Series(values, index=pd.to_datetime(result["timestamp"], unit="s")).dropna().astype("float32")
            print(f"Data source: Yahoo Finance chart API ({host})")
            break
        except Exception as exc:
            errors.append(f"{host}: {exc}")
            print(f"Yahoo chart request failed: {host}: {exc}")
    else:
        raise ValueError(f"Could not download {ticker} market data: " + " | ".join(errors))

    scaled = scaler.transform(close.to_numpy().reshape(-1, 1))
    windows = []
    for i in range(WINDOW, len(scaled)):
        windows.append(scaled[i - WINDOW:i, 0])
    if not windows:
        raise ValueError("Not enough data to create input windows.")
    return np.asarray(windows, dtype="float32")

def main():
    parser = argparse.ArgumentParser(description="Explain an LSTM model with SHAP.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="6mo")
    parser.add_argument("--background", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    ticker = args.ticker.upper()

    model_path = ARTIFACTS / f"{ticker.lower()}_lstm.keras"
    scaler_path = ARTIFACTS / f"{ticker.lower()}_scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Model files not found for {ticker}. Run src/train.py first."
        )

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    windows = load_recent_windows(ticker, args.period, scaler)

    background_count = min(args.background, len(windows))
    sample_count = min(args.samples, len(windows))
    background = windows[-(background_count + sample_count):-sample_count or None]
    explain_rows = windows[-sample_count:]

    # KernelExplainer accepts a 2D matrix; reshape rows back to LSTM tensors.
    def predict_from_flat(flat_rows):
        shaped = np.asarray(flat_rows, dtype="float32").reshape(-1, WINDOW, 1)
        return model.predict(shaped, verbose=0).reshape(-1)

    explainer = shap.KernelExplainer(predict_from_flat, background)
    shap_values = explainer.shap_values(explain_rows, nsamples=100)
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[0]
    values = values.reshape(len(explain_rows), WINDOW)

    # Lag 1 is the most recent day; lag 30 is the oldest day in each window.
    lag_labels = [f"lag_{lag}_days_ago" for lag in range(WINDOW, 0, -1)]
    importance = pd.DataFrame({
        "feature": lag_labels,
        "mean_absolute_shap_value": np.abs(values).mean(axis=0),
    }).sort_values("mean_absolute_shap_value", ascending=False)

    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / f"{ticker.lower()}_shap_importance.csv"
    plot_path = RESULTS / f"{ticker.lower()}_shap_importance.png"
    importance.to_csv(csv_path, index=False)

    top = importance.sort_values("mean_absolute_shap_value")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["mean_absolute_shap_value"])
    plt.title(f"{ticker} - SHAP feature importance by input lag")
    plt.xlabel("Mean absolute SHAP value (scaled model output)")
    plt.ylabel("Input day")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(f"Saved SHAP importance table: {csv_path}")
    print(f"Saved SHAP importance chart: {plot_path}")
    print("\nTop 5 input lags:")
    print(importance.head().to_string(index=False))
    print("\nInterpretation: SHAP values describe this model's output sensitivity")
    print("for the selected background and recent windows; they do not explain")
    print("real-world causes of stock-price movements or guarantee future outcomes.")


if __name__ == "__main__":
    main()
