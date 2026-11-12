# Resource and payment success-rate forecasts

This experiment adds six five-minute-ahead point forecasts to the existing processing-time model. The target is the single observation at `t+5`, with inputs available through the one-minute reading at origin `t`.

| Resource readings | Payment-channel readings |
|---|---|
| `cpu_utilization` | `neft_success_rate` |
| `memory_usage` | `imps_success_rate` |
| `db_connection_pool_usage` | `rtgs_success_rate` |

Each target is fitted separately. The rate columns and resource readings are percentages on a 0–100 scale. Errors are reported in **percentage points**, not percentages of the observed values. These forecasts do not predict service availability, fraud, incidents, or the number of transactions.

## Inputs and leakage boundaries

`infrastructure_metrics_train.csv` contains 8,064 consecutive minute readings ending at `2025-10-25 14:23`. The held-out test file contains the next 2,016 minutes. Discovery already covered this monitoring period. The supplied test split is fixed, but it is not a newly collected sample.

The model uses only the target's own lagged history: current and past observations at lags 0, 1, 5, 15, and 60 minutes; trailing means and standard deviations over 5, 15, and 60 minutes; five-minute change; yesterday's reading at the **target** clock time; and known calendar features. For a `t+5` target, yesterday's matching clock minute is `t-1435`, not `t-1440`.

The first 1,435 training origins are excluded because they have insufficient history. The final five training origins have unknown `t+5` targets. We fit on the remaining 6,624 training examples per target. The target label is never a feature. Training and inference reject missing minutes, duplicate or timezone-aware timestamps, nulls, and out-of-range source percentages; neither interpolates missing data nor clips observed measurements.

## Model selection

Each target compares persistence (`y[t]`), the last 15-minute mean, the last 60-minute mean, the matching reading from the previous day, scaled Ridge regression, and histogram gradient boosting. Baselines can win. Predictions are clipped to the physically meaningful range 0–100, **after** inference.

Three expanding chronological folds use 1,200-minute validation blocks and a five-minute gap from training origins. The last training target must precede the first validation origin. Mean fold MAE selects the candidate, with mean RMSE and then name used to break ties. Every candidate is evaluated on training data alone before the test file is opened. Selected candidates are refitted on all eligible training examples and serialized separately under `models/kpi_5min/<target>/`.

The test replay begins at `2025-10-25 14:24`. There are 2,011 scored test origins per target; the last five origins have no observed `t+5` label in the supplied file. The test segment is not used for fitting or selecting the candidate. The report contains all six candidates' test errors, with the selected model and persistence replay in `test_predictions.csv`.

## Observed comparison

These figures were reproduced from the supplied local CSVs. The selected model is fixed from training-only validation; test MAE is a subsequent evaluation.

| Target | Selected | CV MAE (pp) | Test MAE (pp) | Persistence test MAE (pp) |
|---|---|---:|---:|---:|
| CPU utilization | trailing 15-minute mean | 3.0249 | 2.7893 | 3.6725 |
| Memory usage | trailing 15-minute mean | 1.9844 | 1.7156 | 2.1637 |
| DB connection-pool usage | trailing 15-minute mean | 5.3389 | 5.1532 | 6.8206 |
| NEFT success rate | Ridge | 0.9984 | 1.0021 | 1.3520 |
| IMPS success rate | Ridge | 0.4928 | 0.5040 | 0.6569 |
| RTGS success rate | Ridge | 0.2500 | 0.2460 | 0.3365 |

Resource utilization shifts between the train and test periods. The table shows how these models performed on the supplied test interval, not how they will behave on later infrastructure readings. The forecasts have no uncertainty intervals or incident labels.

## Run locally

From the repository root, with the original training, test, and combined monitoring CSVs already in ignored `data/raw/`:

```bash
.venv/bin/python scripts/train_kpi_forecasts.py --data-dir data/raw
.venv/bin/python tests/test_kpi_forecasting.py --data-dir data/raw
.venv/bin/python scripts/predict_kpi.py --target cpu_utilization \
  --history data/raw/infrastructure_metrics.csv \
  --as-of "2025-10-25 14:24:00"
```

Training refuses to overwrite existing model and report directories; specify new `--model-dir` and `--report-dir` paths for repeated experiments. Saved-model inference checks the model hash, feature-code hash, package versions, target name, and timestamp contract. Load only trusted joblib artifacts. Inference discards future rows before feature construction and does not return a future actual value.

Outputs remain outside Git: model directories under `models/kpi_5min/` and validation, test, and inference records under `reports/kpi_forecast/`. Notebook `08_kpi_forecasting.ipynb` inspects those reports; its committed cells have no saved execution results.
