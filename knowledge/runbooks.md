# Log investigation notes

These project-authored notes cover the non-routine patterns in the supplied log
catalog. They are proposed investigation steps, not records of completed fixes.
The supplied knowledge-base text is indexed separately and kept unchanged.

## NEFT gateway timeout
Event type: neft_gateway_timeout

The message reports a NEFT failure with Gateway timeout as its stated reason.
It does not identify whether the delay occurred in the gateway, network, or an
upstream service. Start with the recorded timestamp and payment channel.

### Checks
Compare `neft_success_rate` and aggregate processing time around the message.
Inspect nearby gateway and network messages, then request a transaction trace or
gateway status from the operator. The supplied log IDs do not establish a join
to the fraud-training transactions. Before any retry, confirm the transaction's
final status and the application's duplicate-payment controls. Retrieval does
not authorize a retry or prove that the payment was not completed.

## IMPS invalid account
Event type: imps_invalid_account

The stated reason is Invalid account. This is different from an IMPS timeout or
an unavailable server. Keep the beneficiary-validation path separate from
infrastructure diagnosis.

### Checks
Inspect the validation response and account details in an authorized system.
Compare `imps_success_rate` around the timestamp. The dataset has no beneficiary
fields or validation trace, so it cannot establish which account detail failed.
Do not propose increasing server capacity as a fix for this message alone.

## RTGS insufficient funds
Event type: rtgs_insufficient_funds

The log reports Insufficient funds as the reason for an RTGS rejection. It does
not show an application outage, account balance, or settlement state.

### Checks
Check the authorized balance and transaction-status systems before advising a
customer. Compare `rtgs_success_rate` with surrounding readings, but keep a
business rejection separate from a gateway or infrastructure failure. No balance
or reversal can be inferred from the provided CSVs.

## Duplicate transaction
Event type: duplicate_transaction

The message identifies a possible duplicate transaction. It does not say whether
a duplicate payment was posted, prevented, or merely submitted twice.

### Checks
Request the original transaction status and idempotency record. Review adjacent
retry messages before recommending resubmission. No idempotency key or settlement
record is present in the supplied data, and log transaction IDs cannot be assumed
to match the fraud-training IDs.

## Authentication failure
Event type: authentication_failure

The log records Customer authentication failed. Authentication failures should be
examined alongside `auth_success_rate`, not counted as payment failures.

### Checks
Compare authentication non-success (`100 - auth_success_rate`) with surrounding
minutes. Inspect the identity-provider response, recent configuration changes,
and request traces if the operator can supply them. The CSVs contain no password,
provider response code, or verified attack label. Do not conclude credential theft
or change access controls from this message alone.

## Payment gateway response time
Event type: payment_gateway_sla_warning

The warning says the gateway response time exceeded an SLA. The message supplies
a duration, but neither the contractual threshold nor the measured endpoint is
defined in the dataset.

### Checks
Compare the logged duration with a documented gateway limit when one is available.
Inspect gateway and payment-channel messages around the timestamp. Keep this logged
duration separate from `avg_transaction_processing_time_ms`, which is an aggregate
metric rather than a channel-specific gateway latency measurement.

## Database connection pool exhaustion
Event type: database_pool_exhaustion

The message reports that the database connection pool is exhausted and includes
a connection count. The count is not the same unit as `db_connection_pool_usage`.
The file does not identify the pool, its configured limit, or the affected service.

### Checks
Compare the pool-usage KPI with processing time and application errors nearby.
Request pool capacity, active connection counts, and long-running query details.
Connection leaks and slow queries are hypotheses in the supplied runbook, not
verified causes. A temporary capacity change needs an operator's assessment of
the database's limits.

## High CPU utilization
Event type: high_cpu

The log reports high CPU on a named server. Some logged percentages exceed 100,
while the monitoring file has no host key. A per-process or per-core measurement
cannot automatically be compared with an aggregate 0–100 percentage.

### Checks
Check the CPU measurement scope first. Compare validated CPU readings with
processing time and traffic, and request process-level telemetry when needed.
The supplied runbook mentions 85%, but the monitoring history did not contain
such a breach during discovery. Do not create a CPU breach from the runbook text
or treat a correlation as a confirmed cause.

## Memory threshold warning
Event type: memory_threshold_warning

The warning reports memory approaching a threshold. The threshold is not given,
and some logged percentages exceed 100. Preserve the reported value for review.

### Checks
Request the host or container memory limit and measurement scope. Compare valid
memory readings with nearby OOM restart messages. A falling `memory_usage` value
after a load change does not by itself establish a leak or recovery. Heap profiles
and process-level history are not available in the supplied files.

## Container out-of-memory restart
Event type: container_oom_restart

The message says a container is restarting due to OOM. The supplied metrics do
not include container IDs, configured memory limits, or restart counters.

### Checks
Request the container termination reason, memory limit, and preceding usage.
Compare nearby memory warnings and errors, without assuming the aggregate metrics
belong to that container. A restart message does not establish that recovery
completed or that changing the memory limit is the appropriate fix.

## Network timeout
Event type: network_timeout

The text begins with CRITICAL but the structured log level may be ERROR. Keep
both values. A timeout message contains a latency value and server name, but no
remote endpoint or request trace.

### Checks
Review adjacent gateway messages and obtain endpoint-level connectivity traces.
The recorded `network_throughput_mbps` is throughput, not latency or packet loss.
A larger throughput value alone does not establish network congestion. Do not
promote the text prefix into an independently verified severity label.

## API rate limit exceeded
Event type: api_rate_limit

The message names `/api/transactions` and reports a rate limit being exceeded.
No configured quota, request window, client identifier, or response header is
available in the log.

### Checks
Request the gateway policy and request counts for the affected client or tier.
Compare traffic and error readings without calling `application_error_rate` a
percentage; its denominator is undocumented. Distinguish a legitimate traffic
increase from a client retry loop before proposing a policy change. Do not
remove throttling automatically.

## Disk I/O bottleneck
Event type: disk_io_bottleneck

The message reports a disk I/O bottleneck on a named server. The monitoring table
has no disk latency, queue-depth, filesystem, or device-utilization measurement.

### Checks
Request disk and filesystem telemetry for that server and inspect surrounding
application messages. Aggregate processing time can indicate coincident slowness,
but it cannot identify a saturated device. Do not claim disk capacity exhaustion
or recommend deleting files without the missing evidence.
