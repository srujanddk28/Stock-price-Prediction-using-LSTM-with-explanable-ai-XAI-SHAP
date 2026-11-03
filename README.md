# Stock Price Prediction using LSTM

A beginner-friendly educational project that downloads historical stock prices, trains an LSTM model on past closing prices, evaluates predictions on a chronological test split, and plots actual vs predicted values.

> **Important:** This is a learning/demo project, not financial advice. Stock prices are uncertain; model predictions should not be used alone to make investment decisions.

## Features
- Download historical market data with `yfinance`
- Build rolling 30-day sequences for an LSTM
- Keep the train/test split chronological to reduce data leakage
- Report MAE and RMSE
- Save a prediction chart and trained model

## Requirements
- Python 3.10 or newer
- Internet connection for downloading market data

## Setup (Windows)
```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
Download data and train the model:
```bash
python src/train.py --ticker AAPL --period 5y
```

Try another ticker:
```bash
python src/train.py --ticker MSFT --period 5y
```

The script saves the model, scaler and chart in the `artifacts/` folder.

## Project structure
```
src/
  train.py       # download, prepare, train and evaluate
requirements.txt
artifacts/       # generated model and charts (created when you run it)
```

## How it works
1. Fetch daily historical closing prices.
2. Scale the training data.
3. Turn the time series into 30-day input windows.
4. Train an LSTM to predict the next closing price.
5. Evaluate predictions on the later, unseen portion of the series.

## Next improvements
- Add SHAP-based explanations and compare feature contributions.
- Add a Streamlit dashboard for ticker selection.
- Compare LSTM results with a simple baseline such as the previous day's close.
- Add transaction costs and walk-forward validation before any strategy experiments.

## Author
Srujan S — learning project. Update this section with your own notes as you build and understand each part.
