# Results folder

Run the training script to generate your own results in this folder:

- `<ticker>_actual_vs_predicted.png` — actual and predicted test prices
- `<ticker>_training_loss.png` — training and validation loss
- `<ticker>_predictions.csv` — dated actual, LSTM and baseline predictions
- `<ticker>_metrics.json` — MAE and RMSE for the LSTM and baseline
- `<ticker>_shap_importance.png` — SHAP importance by input lag
- `<ticker>_shap_importance.csv` — SHAP importance values

These files are generated on the computer that runs the project. They are not
pre-filled with invented scores. After running the scripts, review the charts
and metrics before committing selected outputs to GitHub.
