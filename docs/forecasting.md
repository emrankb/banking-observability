# Processing-time forecasting

## Target and inputs

At origin `t`, predict the single `avg_transaction_processing_time_ms` reading at
`t + 5 minutes`. Five minutes is the initial demo horizon, not an operational SLO.
The measurements have no timezone or arrival-delay definition. The experiment
assumes each minute's readings have arrived before issuing its forecast.

Latency lags include the current reading and the preceding 1, 5, 15, 60, and
1,440 minutes. Trailing means and standard deviations use 5, 15, and 60 readings.
Resource histories cover CPU, memory, database connections, network throughput,
and recorded hourly transaction volume. Volume is not summed into payment counts.
Calendar features describe the known target time; future resource measurements
are never inputs. The resulting feature table has 64 columns.

Daily persistence predicts `y[t + 5 - 1440]`, not `y[t - 1440]`. All candidates use
the same eligible forecast origins. The common loader requires 1,441 consecutive
readings even though the 15-minute trailing-mean baseline needs only 15. This
keeps the baseline comparisons and learned candidates on the same interface.

## Candidate comparison

Five baselines are compared with three ridge penalties and twelve histogram
boosting configurations. Ridge uses a scaler fitted inside each training fold.
Boosting compares clock-only, latency-plus-clock, and full resource-history
features. Each group tries 7 or 15 leaves and two combinations of iteration count,
minimum leaf size, and L2 regularization. The grid is defined in
`forecast_models.py`.

The boosting candidates use absolute-error loss, aligned with selection by MAE.
Their internal random early stopping is disabled; chronological validation
compares the declared iteration budgets. The reproducibility seed comes from the
training file's SHA-256 prefix and is not tuned for performance.

## Validation and replay

Three expanding folds each validate 1,440 origins. A five-row gap keeps the final
fitted label strictly earlier than the next validation origin. The winning
candidate has the smallest mean validation MAE, breaking ties by RMSE and name.
Baselines and learned models use the same ranking. Saved alternatives are
listed separately in the model registry.

The winner and comparison set are frozen before this script reads the supplied
test CSV. Final fitting uses the training file only. Replay then issues a new
five-minute forecast for each eligible observed test minute, without refitting.
Earlier test observations can become lag inputs; later ones cannot.

The full monitoring interval was inspected during discovery, so the test
period was not unseen during exploration. It is excluded from fitting and model
selection. Changes based on its results need evaluation on another interval.

## Nonstationarity and limits

The training data cover five weekdays and part of Saturday. The same recorded
hours have different processing-time levels on the weekend. Autocorrelation and
a strong same-minute CPU relationship do not establish stable weekly behavior.
The experiment keeps latency in milliseconds, exposes recent history and known
calendar features, and retains the difficult chronological validation period.
This compares recent-history features against fixed-mean and clock-based
baselines when the processing-time level changes.

Reports include MAE, RMSE, signed bias, and tail errors by time slice. They also
compare prediction variation with observed variation. A smooth forecast can follow
changes in the average level while still missing individual spikes.
There are no calibrated intervals, live ingestion, or future-incident labels in
this experiment. Criticality classification remains a separate problem.

## Artifacts and notebooks

Training writes its plan, selection, predictions, runtime versions, source hashes,
and model metadata. Inference loads that artifact and reuses the training feature
code. Version or checksum mismatches fail explicitly. Joblib files are trusted
local artifacts, not an untrusted upload format.

The notebook reads generated results and recomputes evaluation scores. It is
stored without outputs; run all cells using the project kernel after training.
Generated reports and model files are ignored by Git. Review notebook outputs
before committing any local changes.
