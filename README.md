# Stock Price Prediction using LSTM and Explainable AI (SHAP)

**A hands-on machine-learning project by Srujan S**  
Python · TensorFlow/Keras · LSTM · SHAP · Pandas · scikit-learn · GitHub Actions

This project trains an LSTM neural network to forecast the next daily closing price from the previous 30 trading days. It evaluates predictions on a chronological holdout set, compares the model with a simple previous-day baseline, and uses SHAP to inspect which input lags influence selected model outputs.

> **Important evaluation finding:** In the currently published AAPL run, the LSTM performs worse than the previous-day baseline. This is reported transparently rather than presenting the neural network as a successful trading predictor. Improving this result is part of the ongoing project.

> **Disclaimer:** This is an educational project, not financial advice. Historical test performance does not guarantee future results.

## Project highlights

- Downloads historical adjusted closing prices from the Yahoo Finance chart endpoint.
- Uses chronological 80/20 train/test separation and fits preprocessing on training data only.
- Creates rolling sequences from the previous 30 trading days.
- Trains a stacked LSTM with early stopping.
- Reports MAE and RMSE for both the LSTM and a previous-day persistence baseline.
- Generates actual-vs-predicted and training-loss charts.
- Uses SHAP KernelExplainer to estimate the influence of each input lag.
- Runs automatically through GitHub Actions and publishes result files to this repository.

## Latest published AAPL evaluation

<text>**Data snapshot:** 1,254 daily observations requested over 5 years; 1,003 training observations and 251 test observations. Generated on 3 November 2026.</text>

| Metric | LSTM | Previous-day baseline |
|---|---:|---:|
| MAE | 14.61 | **3.20** |
| RMSE | 18.47 | **4.47** |

Lower MAE and RMSE are better. In this run, the baseline is substantially better than the LSTM. The model should not be described as outperforming the baseline.

See the full [metrics JSON](results/aapl_metrics.json) and [date-wise predictions CSV](results/aapl_predictions.csv).

## Visual results

### Actual vs. predicted closing prices

![AAPL actual vs predicted](results/aapl_actual_vs_predicted.png)

### Training and validation loss

![AAPL training loss](results/aapl_training_loss.png)

### SHAP input-lag importance

![AAPL SHAP feature importance](results/aapl_shap_importance.png)

The SHAP chart summarizes mean absolute SHAP values across selected recent windows. A larger value indicates greater influence on the model output for those selected examples and background data; it does not identify real-world causes of stock-price movements.

## Repository structure

```text
.
├── .github/workflows/
│   └── train-and-publish.yml   # Automated training and result publishing
├── src/
│   ├── train.py               # Download, train, evaluate and plot
│   └── explain.py             # SHAP explanation generation
├── results/
│   ├── aapl_actual_vs_predicted.png
│   ├── aapl_training_loss.png
│   ├── aapl_predictions.csv
│   ├── aapl_metrics.json
│   ├── aapl_shap_importance.png
│   └── aapl_shap_importance.csv
├── artifacts/                 # Generated model and scaler (not committed)
├── requirements.txt
└── README.md
```

## Run locally on Windows

Open Command Prompt or the VS Code terminal in the project folder.

```bash
py -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py` is unavailable, use `python` instead.

### 1. Train and evaluate

```bash
python src/train.py --ticker AAPL --period 5y
```

Try another ticker:

```bash
python src/train.py --ticker MSFT --period 5y
```

### 2. Generate SHAP explanations

Run this after training the same ticker:

```bash
python src/explain.py --ticker AAPL --period 6mo --background 10 --samples 5
```

SHAP KernelExplainer may take several minutes. The explanation script expects the trained model and scaler in the `artifacts/` directory.

## Evaluation methodology

1. Download adjusted daily closing prices.
2. Keep the observations in chronological order.
3. Use the first 80% for training and the remaining 20% for testing.
4. Fit MinMaxScaler only on the training portion.
5. Use a 30-trading-day lookback to predict the next observation.
6. Compare the LSTM with a persistence baseline that predicts the next close using the latest observed close.

- **MAE (Mean Absolute Error):** average absolute prediction error; lower is better.
- **RMSE (Root Mean Squared Error):** gives larger errors more weight; lower is better.

## Automated workflow

The GitHub Actions workflow runs the AAPL training and SHAP scripts and publishes generated charts, CSV files and metrics to `results/`.

To run it manually:
1. Open the repository's **Actions** tab.
2. Select **Train model and publish graphs**.
3. Select **Run workflow**.

[View workflow runs](../../actions).

## Current limitations and next improvements

- The current model uses only one input feature: historical adjusted closing price.
- The current holdout evaluation is not a full walk-forward backtest.
- The published LSTM result is worse than the persistence baseline; this needs investigation before claiming predictive value.
- SHAP explains model behaviour for selected inputs, not causal market factors.
- Market-data endpoint availability and compatibility can change.

Planned improvements:
- Investigate scaling, target alignment and model under/overfitting.
- Add volume and carefully selected technical indicators.
- Tune lookback length, model size and training parameters using validation data only.
- Add walk-forward evaluation and compare with ARIMA and other simple baselines.
- Build a Streamlit dashboard only after the evaluation pipeline is reliable.

## Skills demonstrated

Python programming · Data cleaning · Time-series preprocessing · LSTM neural networks · Model evaluation · Baseline comparison · Data visualization · Explainable AI (SHAP) · Git/GitHub · GitHub Actions

## Author

**Srujan S**  
[GitHub profile](https://github.com/srujanddk28)
