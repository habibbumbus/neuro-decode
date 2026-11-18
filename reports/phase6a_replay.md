# Phase 6A: sequential replay equivalence and CPU processing time

All three decoders reproduce the saved validation predictions within the
predeclared numerical tolerances. Warmed preparation-plus-model p95 is below
20 ms for each decoder, but maxima exceed 20 ms. These are recorded-data
processing measurements, not a live BCI or a hard real-time guarantee.

## Fixed protocol and state

Protocol configs/replay.toml was written before results: validation only,
batch size one, one torch/native thread, 200 untimed warm-up predictions per
model, then a fresh scientific state and one chronological timing pass through
all 2,775 eligible bins. Model order is ridge, representative GRU seed 2026,
Kalman. Linear percentile interpolation; no outlier trimming or fastest-run
selection. Equivalence must pass for all models before warmed benchmarking.

Use the existing recording/binning/quality preparation, channel/unit mapping,
frozen Phase 2D scaler and unchanged checkpoints. The stream reveals 2,784
completed validation count bins one at a time. Nine initial bins supply history
without predictions; all 2,775 original eligible targets are emitted, ending
1550.200–1605.680 s. Predictions retain exact source IDs, timestamps, partition
and segment identities. The source file and bins are prepared in memory first;
this is a sequential decoder test, not incremental disk/event acquisition.

Each packet contains counts and the fixed existing quality flags, never movement
values. The quality adapter reuses preparation's exact validation and identity
helpers; its up-front structural checks include label finiteness. The decoder
receives only the resulting flags. Therefore this is not a claim of a deployed
label-free quality detector. No movement value affects a prediction.

The buffer owns the last ten consecutive valid raw count vectors, oldest first.
It reuses HistoryWindows and the existing ridge_inputs, gru_inputs and
current_bin_observations adapters. No alternate normalization, interpolation,
lag, target definition, channel selection or event binning was introduced.
The singleton history has NaN scoring placeholders; prediction APIs never read
them. Ridge receives [1,960] float64 features. GRU receives [1,10,96] float32,
fresh zero h0 [1,1,32] on every window, then the existing two-output head.
No GRU hidden state carries between overlapping windows.

Kalman uses only the current standardized bin and its previous estimated
velocity/covariance. It reuses kalman_step, including the numerical solve and
Joseph covariance update. It starts at the first common eligible row, updating
the saved training prior without an extra transition. The nine history-only
bins are not extra Kalman updates. Later eligible rows predict then update.
An invalid source bin clears history. Missing grid bins, segment/session or
partition changes also clear history; a non-emission deactivates Kalman state.
Only after complete eligible support returns does Kalman reset to its training
prior. Zero counts remain valid observations, not missing packets. No state is
propagated across invalid time, and no held-out movement initializes state.

## Equivalence

Compare every emitted identity and both axes with the saved Phase 5B CSV and
fresh calls to the offline interfaces. All 2,775 identities match exactly;
each real stream has one initial reset and no later discontinuity. Synthetic
tests cover later gaps, invalid bins and boundary resets.

| Decoder | Maximum absolute replay–saved difference, mm/s | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: | ---: |
| Ridge | 3.126388e-13 | 1e-10 | 1e-12 |
| GRU, seed 2026 | 1.220703125e-4 | 1e-4 | 1e-6 |
| Kalman | 0 | 1e-10 | 1e-12 |

The rule is abs(replay − saved) <= atol + rtol * abs(saved), not absolute
tolerance alone. At the largest GRU absolute difference, bin 13930 x, saved
velocity is 125.784500122 and replay is 125.784622192 mm/s. Allowed difference
is 0.0002257845 mm/s. The largest difference-to-allowed-tolerance ratio across
all axes/rows is 0.901394, below one. Tolerances were not relaxed after results.

Fresh offline batch prediction reproduces the saved values exactly for every
model. Replay-versus-offline differences equal those above. The artifact audit
also calls the offline GRU with batch_size=1 and obtains exact equality with
all replay outputs: maximum difference zero. This isolates the small float32
difference to batch arithmetic, rather than changed histories or carried state.
Ridge's batch/singleton float64 roundoff is negligible; Kalman is bit-identical.
The timed pass also produces exactly the same values as the untimed replay.

## Measured latency

Headline run: runs/phase6a-validation-timing, after the full test suite finished.
An earlier runs/phase6a-validation-replay run is preserved as diagnostic evidence;
it overlapped the assistant's full test run, so its timings were not used for
the headline table. The clean run used the same protocol and code, with one
timing pass and no competing assistant test job. It was not selected for faster
scores: some clean-run times are worse. Other system activity was not controlled.

Hardware: 13th Gen Intel Core i9-13900H, 20 logical CPUs available, Windows
kernel build 26300. Python 3.11.17, torch 2.14.1+cpu, NumPy 2.4.6,
SciPy 1.17.1, scikit-learn 1.7.2, h5py 3.16.0, threadpoolctl 3.7.0.
CPU only; torch intra-op and discovered native BLAS/OpenMP pools limited to one
thread. Full pool versions and platform details are saved in run.json.

Clock: time.perf_counter_ns, Windows QueryPerformanceCounter, monotonic and
non-adjustable, reported resolution 100 ns. Per-emission timers are nested;
their overhead and OS scheduling pauses are retained. No clock calibration,
CPU affinity, power/thermal control or hard real-time scheduling was attempted.

Every table row uses 2,775 warmed eligible emissions, including the initial
scientific reset row. Times are milliseconds. Service rate is 1000/mean timed
milliseconds, not an observed acquisition or complete application throughput.

| Decoder | Boundary | p50 | p95 | p99 | Maximum | Timed service rate, bins/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | Model API | 0.8161 | 1.7750 | 3.0928 | 28.2351 | 1026.55 |
| Ridge | Preparation + model | 2.3785 | 4.2600 | 7.0481 | 79.9212 | 371.90 |
| GRU | Model API | 2.2428 | 3.6974 | 6.5368 | 64.6608 | 413.82 |
| GRU | Preparation + model | 3.8030 | 6.2640 | 10.0801 | 161.6228 | 236.81 |
| Kalman | Model API | 1.7981 | 3.1333 | 5.8446 | 21.7631 | 500.99 |
| Kalman | Preparation + model | 3.4502 | 5.7962 | 10.6909 | 27.2538 | 264.51 |

Model API starts with already normalized, indexed singleton inputs and includes
the existing checked predict API for ridge/GRU, or kalman_step with current
estimated state for Kalman. It includes parameter/input validation, scaler
identity checking, GRU tensor copying/conversion, inference-mode setup, NumPy
output conversion and Kalman covariance/numerical diagnostics. It is not a
bare matrix multiplication or isolated recurrent-kernel microbenchmark.

Preparation + model starts with one ReplayBin already in memory. It includes
packet validation, history copying/ordering, singleton HistoryWindows creation,
the existing frozen normalization/feature adapter, model API and state update.
Kalman reconstructs the same common history for eligibility and adapter reuse,
although its observation uses only the final bin. That extra preparation is
included, rather than replaced by a new optimized scientific path.

Both boundaries exclude imports/model loading, recording reads, spike-event
binning, target/quality preparation, packet construction, acquisition time,
plotting, playback sleep, CSV/JSON logging and returned-record packaging.
Warm-up/non-emission bins are fed, but only emitted predictions enter the
latency distribution. This is not the full sensor-to-output pipeline time.

First eligible calls after load are recorded separately below. These occur
before each model's offline comparison in this process; they are not fresh
process startup measurements and exclude model loading. Libraries/imports and
some earlier work were already initialized. No cold-start guarantee is claimed.

| Decoder | First model API, ms | First preparation + model, ms |
| --- | ---: | ---: |
| Ridge | 0.7442 | 2.3606 |
| GRU | 4.9606 | 6.5680 |
| Kalman | 2.7681 | 5.0019 |

The acquisition bin still takes 20 ms to complete; initial history requires ten
bins (200 ms). Once filled, a new trailing window is available every 20 ms.
The measured processing happens after its current bin closes. Playback speed
does not change these measurements. Although p95 is below 20 ms, retained worst
cases exceed 20 ms for all models, including a 161.623 ms total GRU outlier.
These wall-clock observations do not identify the source of every pause or
establish deadline compliance, closed-loop control or clinical suitability.

## One replayed instant

At 1550.200 s, bin 13110 has just completed. Only bins 13101–13110, covering
(1550.000,1550.200], have been revealed. No bin after 13110 or movement value
is supplied. Ridge flattens the frozen-normalized history and predicts
[31.390114,12.002288] mm/s. GRU starts its ten-bin recurrence at zero and predicts
[-3.388831,15.284597] mm/s. Kalman updates its saved training prior using bin
13110's 96 standardized counts, producing [-0.807717,-6.165349] mm/s.
The actual [14.162167,3.788288] mm/s is accessed separately for reporting.

At the next bin, 13111 ending 1550.220 s, ridge and GRU slide to bins
13102–13111 and recompute; GRU again starts at zero. Kalman carries only its own
previous estimate/covariance, performs A/a prediction then the current-bin
measurement update, and emits [-28.666149,-40.008431] mm/s. Neither the prior
actual target nor the new one enters any model. This is recorded neural replay.

## Evidence and reproduction

Phase 5B checkpoint 83a11da was verified on local and GitHub main before work.
During implementation main advanced to a90988b (README redesign); its README,
banner and reproduction documentation were preserved. The clean run records
base a90988b2d3c6982c070d3077759ce6fb0cc4a3da plus dirty/untracked replay files
and actual LF-normalized implementation hashes. These results are not from
that commit alone. No assistant staging, commit, push, merge or branch switch.

All original source/split/target/scaler/channel identities and trusted checkpoint
hashes are saved. Normalization hash remains
603aa299f0a1d75625acea41a18b9ecf56c58e4e0515ec8322ea1aa34abe68dd.
The selected models are the unchanged ridge lambda=1000, Phase 4A Kalman and
Phase 5B lr=0.003/seed=2026/epoch=50 GRU. No training, setting selection or
final-test evaluation was performed. Common source/global metadata loading
still reads the recording's bytes; no test examples or timing were constructed.

Generated evidence stays under ignored runs/: protocol.json, equivalence.json,
predictions.csv, bin_status.csv, timing_samples.csv, latency.json, run.json and
verification.json. The artifact-only/statistical audit checks all recorded
hashes, reproduces every timing summary exactly from raw nanoseconds and
confirms exact offline batch-one GRU equality. The later audit has its own code
hash. No model checkpoint or bulk run is intended for Git.

All 34 new synthetic replay tests passed; the complete suite passed 303 tests.
Checks cover oldest-to-newest order, frozen scaling, count ownership, invalid
bins, zero counts, identity/grid rejection, boundaries, label-value independence,
future-bin causality and offline/replay agreement. Initial fixture failures
were repaired by aligning its declared synthetic segments and exact grid edges;
production boundary comparisons were not relaxed. Lint, formatting, offline
lock consistency and locked environment check passed. No dependencies changed.

Actual experiment commands used the existing .venv interpreter. Reproduce from
the repository root with trusted earlier artifacts and a new run ID:

~~~powershell
uv run --locked python scripts/replay_validation.py phase6a-review
uv run --locked python scripts/audit_replay.py phase6a-review
uv run --locked pytest -p no:cacheprovider
uv run --locked ruff check .
uv run --locked ruff format --check .
uv lock --check --offline
~~~

Existing run directories and verification files are refused. Run tests before
benchmarking, without a competing assistant workload. Exact latency reproduction
across runs/platforms is not expected. Fresh-clone/cross-platform execution,
raw-event acquisition throughput and full-process cold start were not tested.
No disruptions, cursor integration or video were added. Stop after Phase 6A;
next increment tests predefined input disruptions before building the visual
cursor replay from verified predictions.
