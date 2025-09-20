# LSTM Multivariate Time Series - Air Temperature (AT) Prediction
Multivariate LSTM pipeline to predict hourly air temperature (AT) 1 hour ahead using meteorological and pollutant measurements (AP003.csv). Includes EDA, preprocessing, baseline & tuned LSTM models, and evaluation.

---

## Project Overview

This repository demonstrates a complete workflow for a **multivariate time-series forecasting** task: predicting **air temperature (AT)** 1 hour ahead using the previous 5 hours of multivariate observations (pollutants, wind, etc.). The model uses LSTM networks implemented in TensorFlow/Keras and includes exploratory data analysis, preprocessing, a baseline LSTM, and a tuned LSTM using Keras Tuner.

Problem formulation: given 5 consecutive hourly observations (window size = 5), predict the `AT (degree C)` value 1 hour after the last observation.

---

## Dataset

* File: `AP003.csv`
* Temporal resolution: **hourly** (difference between `From Date` and `To Date` is 3600 seconds = 1 hour)
* Total rows in the notebook example: \~50,400 observations

---

## Key Results

* **Baseline LSTM (10 units)** — `MAE ≈ 1.64`, `MSE ≈ 4.68`, `R² ≈ 0.71`
* **Tuned LSTM (Hyperband)** — best found hyperparameters: `units=64`, `dropout=0.2`, `learning_rate≈8.3e-4` → `MAE ≈ 0.58`, `MSE ≈ 0.77`, `R² ≈ 0.95`

---

## Exploratory Data Analysis (EDA) — summary

Key insights from EDA performed in the notebook:

* Data frequency is hourly; timestamps are in `From Date` / `To Date`.
* Temperature distribution: most `AT` values are around **20–30 °C**, with some high outliers.
* Clear **seasonal pattern** (annual): warmest months around **May–June**, coldest around **Dec–Jan**.
* High correlations observed between pollutant variables: e.g., `PM2.5` and `PM10` (≈0.81), and among `NO`, `NO2`, `NOx`.
* Based on redundancy and low relevance, several columns were dropped prior to modeling (e.g., `NO`, `NO2`, `PM10`, `VWS`, and `Temp (degree C)` — see notebook for exact reasoning).
* Missing values were handled with forward-fill then backfill (`ffill` then `bfill`) because the series is temporal and short-term propagation is reasonable.

---

## Preprocessing & Feature Engineering

* **Train/validation/test split**: sequential split (no shuffling) — `80% train / 10% val / 10% test` to preserve temporal order.

  * Example counts used in the notebook: `40,320 train / 5,040 val / 5,040 test` (from \~50,400 rows)
* **Scaling**: `RobustScaler` applied to the target `AT (degree C)` and to other numerical features to reduce outlier impact.
* **Windowing**: sliding-window creation with `window_size = 5` (5 previous hours as input) and horizon = 1 (predict the next hour). The helper `generate_windowed_samples` builds `(X, y)` arrays for LSTM.
* Final datasets are converted to `tf.data.Dataset` for batching and training.

---

## Modeling

### Baseline model

* Architecture: single LSTM layer with **10 units** followed by a Dense(1) regressor (linear activation).
* Optimizer: `Adam(learning_rate=1e-4)`
* Loss: `mean_squared_error`
* Training: ran for 20 epochs in the notebook with loss/val\_loss monitoring.

### Tuned model (Keras Tuner)

* Tuner: `keras_tuner.Hyperband` searching over:

  * `units` (LSTM) ∈ {32, 64, 96, 128}
  * `dropout` ∈ \[0.1, 0.5]
  * `learning_rate` ∈ \[1e-5, 1e-2] (log sampling)
* Callback: `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`
* Best hyperparameters recovered by tuner (noted in the notebook): **units=64**, **dropout=0.2**, **learning\_rate≈8.3e-4**

---

## Evaluation

Evaluation metrics used: **MAE**, **MSE**, and **R²** (all computed on inverse-transformed predictions / ground-truth values):

* Baseline: MAE ≈ 1.64 | MSE ≈ 4.68 | R² ≈ 0.71
* Tuned: MAE ≈ 0.58 | MSE ≈ 0.77 | R² ≈ 0.95

**Interpretation:** the tuned model significantly improved accuracy and explained variance, indicating the hyperparameter search and regularization (dropout) improved generalization.

---

## Limitations & next steps

* Model is short-horizon (1 hour) and uses only 5 previous timesteps — consider longer windows for different horizons.
* Seasonal effects and exogenous features: include explicit time features (hour, day, month) or Fourier seasonal embeddings.
* RobustScaler mitigates outliers but advanced outlier handling and imputation strategies could improve results.
* Try deeper architectures (stacked LSTM), CNN-LSTM, or attention-based models (Transformers) for longer-range dependencies.
* Perform k-fold time series cross-validation (sliding windows) to better estimate generalization.

