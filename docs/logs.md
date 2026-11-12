# Log grouping

The input is `application_logs.csv`: `timestamp`, `log_level`, and `log_message`.
The aim here is to browse related wording and route known messages. There are no
independent incident-type labels for training a supervised text classifier.

## Cleaning the message

Normalization masks transaction IDs, user IDs, server IDs, timestamps, and numeric
values, including numbers attached to `ms` or `%`. A leading severity prefix is
removed from the text features. The structured level and the original message
are kept alongside the result so disagreements remain visible.

TF-IDF uses word unigrams and bigrams. Placeholders and a short list of structural
words are excluded. Payment channels, failure reasons, and words such as
`failed`, `successfully`, and `OOM` remain. Resource readings are extracted for
quality checks, not used to decide the text cluster. Values outside 0–100 are
flagged without clipping; their measurement scope is unknown.

## Fitting and choosing the number of groups

The last timestamp in `infrastructure_metrics_train.csv` sets the development
cutoff. Only earlier log records determine the vocabulary and centroids. Later
records are assigned without refitting. Discovery already examined the full log
file, so this split is not an independently collected sample.

Each distinct normalized development pattern contributes one vector. Repeating
the same message thousands of times should not dominate the vocabulary weights
or make the grouping look stronger. Occurrence counts and duplicate rows are
retained in the reports.

The experiment compares every k from 2 to one less than the number of distinct
patterns. The one-cluster and all-singleton partitions are excluded because the
silhouette score is undefined there. Each candidate uses 20 k-means++ starts,
up to 500 iterations, and three seeds derived from the development patterns.
KMeans minimizes squared Euclidean distance on L2-normalized vectors; comparison
uses cosine silhouette in that same feature space.

Mean template-level silhouette selects k. Ties use adjusted Rand agreement across
seeds, then smaller k. The final fit uses the first seed, not whichever seed has
the best score. Group numbering is ordered by member text for consistent display.
The report also computes silhouette on a fixed sample of repeated development
rows and on an exact-template partition to show how duplication affects it.

## Reading a group

Centroid terms provide the group labels. Every fitted pattern has a real source
example, its counts, and its category in the report. A group may contain routine
operations and failures together. Shared wording does not imply shared severity.

`config/log_patterns.json` separately assigns names and categories to the supplied
wording: infrastructure, business KPI, or routine. This is an explicit catalog,
not a label learned by KMeans. For example, insufficient funds is a payment-related
message, not evidence of an infrastructure outage. The catalog does not verify
that the reported event actually happened.

## New messages

A message gets a catalog label only when its normalized wording exactly matches a
pattern present during fitting. Unseen wording stays `unreviewed`, even when it
resembles a known message. It can receive a tentative cluster for browsing; text
with no known vocabulary has no cluster. Similarity is not a confidence score.

`label_review_required` concerns unknown wording, missing catalog entries, or
input-quality flags. It is separate from the monitoring detector's incident-review
flag. A known error message can have `label_review_required=false`; this only
means its label needs no additional review under these checks.

Training saves the vectorizer, centroids, fitted patterns, and catalog together
under `models/log_clustering/`. Inference loads this artifact and checks its hash,
code, metadata, and dependency versions. Only load trusted local joblib files.
Reports under `reports/logs/` include source messages and remain outside Git.

The notebook recalculates the selected fit's score and checks raw-data fingerprints.
Its source has empty outputs; save executed copies outside version control.

## References

- [scikit-learn 1.8: text clustering](https://scikit-learn.org/1.8/auto_examples/text/plot_document_clustering.html)
- [scikit-learn 1.8: KMeans](https://scikit-learn.org/1.8/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn 1.8: silhouette_score](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.silhouette_score.html)
