# Incident triage and monitoring anomalies

The incident model classifies a current record as critical or noncritical. The
monitoring detector flags unusual readings as they arrive. Both work on current
measurements, not future incidents.

## Incident model

The target is the supplied `is_critical` column. Inputs are the nine numeric KPI
columns shared by the incident and monitoring tables. IDs, timestamps, severity,
and incident type are excluded from fitting.

The monitoring training cutoff divides the 168 hourly incident records into 135
for development and 33 for evaluation. The development records contain 20 critical
labels; the evaluation records contain six. Three expanding validation windows
each cover 24 records, with a one-record gap before validation.

The comparison uses logistic regression with C values 0.1, 1, and 10, plus trees of
depth 1 and 2 with at least five records per leaf. Every candidate uses balanced
class weights calculated from its own training fold. Logistic regression scales
inputs inside the pipeline. There is no resampling across the time boundary.

Thresholds 0.2, 0.35, 0.5, 0.65, and 0.8 are compared on validation scores. Selection
uses mean F2, giving recall more weight than precision, then average precision
and precision. Remaining ties prefer the earlier, simpler candidate and a
threshold closer to 0.5. The choice is refitted on all development records. The
score is not a calibrated probability of an incident.

The incident table is easy to classify for a problematic reason. Its critical
records disagree with the original monitoring values at the same timestamps in
CPU, memory, database-pool usage, application errors, and NEFT success rate. The
CPU > 85 rule found during discovery reproduces every supplied label. The source
of the labels and the reason for these differences are unknown.

The report checks how much the result depends on those five fields. First, it
scores the original monitoring rows with the same model. Then it refits the model
without the five fields, keeping the selected settings. Neither check is used
to select a new model from the test results. Until the source differences are
resolved, the incident model is limited to demonstrations using incident records;
it is not used for automatic monitoring alerts.

## Monitoring detector

The detector uses the unmodified minute-level monitoring train/test files. Each
load metric is expressed as its current value minus the mean of the previous
60 readings. The current reading is excluded from that reference mean. This
helps distinguish short-term changes from ordinary sustained load, but can miss
slow degradation. Absolute operational limits need a separate rule layer.

The remaining features are the three payment non-success percentages,
authentication non-success, the recorded application-error metric, and uptime
loss. The error metric has no documented denominator and is not called a percent.
All 12 features use measurements available at the scoring timestamp.

The last 2,880 training minutes are reserved for two successive 1,440-minute
blocks: parameter selection and score-threshold calibration. The earlier block
fits candidate Isolation Forests with 128 or 256 trees and subsamples of 128 or
512 rows. Each combination uses three file-derived seeds. Selection favors
agreement in the highest-scoring readings across seeds, then Spearman ranking
agreement. This tests ranking repeatability, not detection accuracy.

The chosen configuration is refitted through the selection block. Its threshold
is then calculated on the separate calibration block; the forest is not refitted
after that. Neither selection nor calibration uses the monitoring test period.

`--review-fraction` defaults to 0.01. This sets the threshold at the 99th percentile
of calibration scores; it does not estimate how often incidents occur. Only
scores strictly above the saved threshold are flagged. Tied scores can reduce
the number of flags, and later periods can have a different flag rate. The
Isolation Forest's built-in cutoff is not used.

An additional check flags changes to features that were constant during fitting.
A forest cannot learn a split on a feature that never varied. This matters for the
single uptime drop in the supplied data. That drop is in the calibration period,
not the test period. A robust-deviation baseline is reported beside the forest;
it uses the same calibration period and review budget.

The response lists the three largest deviations from the reference values to
help inspect a flagged reading. These are separate from the forest score, not
feature attributions or a root-cause explanation. Adjacent flagged minutes are
grouped into review episodes. There are no verified per-minute incident labels
for this series, so these episodes cannot be counted as confirmed outages or
used to calculate anomaly precision, recall, or F1.

## Artifacts and reproduction

Training saves both detectors under `models/detection/` and reports under
`reports/detection/`. Existing output directories are not overwritten. Use new
output paths for another experiment. The original CSVs are never changed.

Each artifact records its features, training cutoff, selection results, source
hashes, package versions, and checksum. The loader checks that metadata against
the artifact. Only load trusted local joblib files; a matching checksum does not
make an unknown file safe.

The notebook checks source hashes and recalculates scores before displaying
results. Its committed source has empty outputs. Save executed copies outside
version control. The full week was inspected during discovery, including the
later evaluation period; it is not an independent unseen dataset.
