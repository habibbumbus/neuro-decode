# Design decisions

## 2026-10-10 — Phase 2A timing and chronological splits

Use 20 ms bins, ten bins of history, zero additional lag, and zero forecast
horizon. Estimate completed-bin displacement rate using stored cursor positions
selected at or before each boundary. Fix maximum sample age and behavioral-gap
threshold at 6 ms, based on source-documented 4 ms sampling. Preserve source time
origin and channel/unit identities; hardware alignment remains unresolved.

Use nominal 70/15/15 fractions of elapsed shared coverage. Do not choose
boundaries from movement quality or future performance. Purge the maximum of
neural history and target-source lookback on each side of both boundaries,
currently 200 ms per side. Require independent full warm-up inside each retained
partition/continuous segment. Purging is a conservative support guard, not proof
of temporal independence.

Use half-open partition/segment ownership, including exclusion of the terminal
recording endpoint as an emission. Behavioral intervals >6 ms (+1 ns round-off
tolerance) split continuity. Never bridge gaps. The tolerance permits neither
future behavioral samples nor expanded partition ownership. Reset future model
state/history at session, partition, and segment starts.

Keep the reviewed Phase 1C interval [1288, 1298) s inside training. Save a small
source-bound specification now; exact eligible indices remain pending. No count
bins, velocity arrays, learned preprocessing, or model-state implementation are
included. See [methods.md](methods.md) for support formulas, endpoint conventions,
reproduction, rationale, and limitations.

## 2026-10-11 — Actual elapsed time for velocity targets

Use selected behavioral elapsed time b1−b0 as the velocity denominator, explicitly
superseding the earlier fixed nominal 20 ms choice. The neural grid remains
20 ms. Record both source times/ages and actual-minus-nominal separation.
Constant-velocity synthetic data with a 16 ms selected interval recover the
correct mm/s with this denominator; nominal division would bias the value.

Accept positive separation within nominal width ± (maximum sample age + 2 ns),
derived from bounded sample ages and grid representation. Never substitute or
clamp a rejected denominator. Continue past-only selection, shared grid edges,
partition/segment ownership and local warm-up. Source-support extent is unchanged;
the saved split boundaries and purge distances require no alteration.

Keep full count rows, invalid target NaNs, original bin IDs and explicit validity
reasons. Use one mask for paired counts/targets. Revalidate bin identities/content
through the existing binning implementation. Audit held-out validity counts only;
show source positions/velocities only from fixed early training rows.
No fitted transforms, histories, models or array persistence are included.


## 2026-10-11 — Histories and unique-bin training normalization

Use oldest-to-newest histories of ten original 20 ms channel-count bins, paired
with the final bin's existing velocity label. Keep full source IDs/timestamps
and reject gaps, invalid rows, segment changes and partition crossings.
Warm-up labels may be unavailable while their source counts remain valid.
Zero counts are legitimate observations; other target-quality exclusions
conservatively break histories, as do explicitly known invalid neural bins.

Fit per-channel population mean/standard deviation over unique valid original
training bins once, including warm-up source counts. Avoid overlap weighting
from repeated window entries. Use frozen statistics on all partitions; exact
zero training standard deviation uses scale 1 with the channel retained.
Keep raw counts separate for later Kalman/robustness use; do not scale velocity.

The fit is offline across training, so intentional refitting on changed later
training data can change statistics. Causality applies to raw histories and
application of a frozen scaler. Held-out values never fit/update it.
Save small ignored fit metadata with source/split/configuration/policy/code and
fit-row/count hashes. Persist no bulk arrays or models at this checkpoint.
See [methods](methods.md) for eligibility, formulas, examples and limitations.


## 2026-10-11 — Shared baseline validation and undefined R²

Keep Phase 2D eligible examples unchanged and score both trivial predictors on
the same validation timestamps/mask. Fit mean velocity only from eligible
training labels; held-out prediction APIs accept label-free metadata.
Zero/mean baselines need no feature scaler or scientific dependency addition.

Overall RMSE averages squared errors over examples and both axes before the
square root. Report per-axis R² and their unweighted mean. Constant axes or
fewer than two scored examples have explicit null R²; mean R² is null if either
axis is undefined. Reject nonfinite values and mismatched timestamps/masks rather
than silently repairing scores. The R² reference mean uses scoring labels only
inside metric calculation, never as a predictor input.

Open validation for development, while final test scoring stays closed.
Fix the plotted interval to the first eligible ten validation seconds before
viewing quality; refuse gaps instead of choosing a favorable replacement.
Save detailed ignored run evidence; commit only the small technical report.


## 2026-10-18 — Ridge fitting and validation selection

Reuse prepared histories, frozen unique-bin training normalization and shared
Phase 3A scoring. Map oldest-to-newest [10,96] histories to 960 features in the
existing C-order; fit two velocity outputs with an unpenalized intercept.

Use scikit-learn's SVD ridge solver and summed squared-error objective.
Declare lambda grid [0.1,1,10,100,1000,10000] before results; minimize overall
validation RMSE, with largest lambda within absolute 1e-9 mm/s of the minimum
as tie-break. Keep test scoring closed. Validation labels select a candidate
but never fit candidate weights or normalization.

Save explicit JSON parameters plus the exact frozen transform and feature/
source/split identity, with strict loading/context checks and no pickle.
Reuse the baseline plotting interval and exporter; preserve old artifacts.
Selected lambda=1000 beats both trivial validation baselines, with no detected
defect requiring a diagnostic increment. Continue next with the Kalman decoder.


## 2026-10-19 — Kalman parameter estimation and covariance policy

Use state [vx,vy], current-bin observations from all 96 channels, and the existing
frozen training-only count standardization. Keep Phase 2D row eligibility and
source identities. Fit A/a only on consecutive valid training grid pairs within
the same session/segment; fit H/d on matched valid training observations/states.
Use centered NumPy least squares, requiring two identifiable velocity axes.

Use population residual covariances (ddof=0). Retain full Q; use diagonal R as
the initial approximation and keep every channel. Record the full raw residual
R's singularity/correlation diagnostics. Apply the predeclared absolute/relative
variance floors only as documented in methods, reporting exact adjustments.
Do not tune floors from held-out performance or silently project A.

Use training velocity mean/population covariance for the future initial prior.
Reset this prior at each eligible run start, before its first observation,
with no extra transition or held-out behavioral initialization.
Save parameters/configuration/identities and diagnostics now. Filtering,
filter-state handling in code and validation comparison are Phase 4B.


## 2026-10-19 — Causal Kalman filtering and unchanged validation comparison

Load the unchanged Phase 4A parameters/transform and selected Phase 3B ridge.
Use numerical solves and Joseph covariance updates, with roundoff symmetrization
and no adaptive covariance tuning. Start/reset from the training prior before
the first eligible observation; no extra transition or held-out velocity input.
Skip invalid rows, reset after discontinuities, and preserve the original shared
nine-bin warm-up exclusions without additional Kalman settling exclusions.
A low-level missing-observation prediction-only step is tested but is not used
to carry state across invalid time in this comparison.

Score all four models on the original 2775 validation rows and first ten-second
plot interval. Kalman underperforms ridge and both baselines; preserve that result.
Independent posterior equations and identity checks reveal no specific correctness
defect. Keep model assumptions as possible limitations rather than silently
retuning Q/R or shifting signals. Next: the causal GRU; final test stays closed.


## 2026-10-19 — Bounded-window GRU components before recording training

The read-only Kalman artifact diagnosis found bias/excess spread and disclosed
conditioning, but no concrete correctness defect; keep the fitted model unchanged.
Implement one CPU float32, unidirectional GRU (H=32) plus a two-output linear head.
Reuse prepared histories/scaler/identities; initialize hidden state to zero for
each window, including later replay, rather than carrying state across windows.
Use seeded Adam/MSE with clipping, chronological validation and earliest best-RMSE
snapshot selection. Save explicit identity-bound tensors/metadata. Use uv's CPU
PyTorch index on Windows/Linux and PyPI fallback on macOS.
Phase 5A verifies synthetic mechanics only. Its overfit fixture deliberately
copies examples for validation wiring and makes no generalization claim.
Real-data candidate declarations, training and matched comparison are Phase 5B.

## 2026-11-17 — Bounded real-data GRU round and all-seed reporting

Declare learning rates [0.001,0.003] before recording training; retain the actual
Phase 5A H32/reset/CPU/optimizer/batching defaults. Use seed 2026 for setting
selection, minimum validation overall RMSE with lower rate within 1e-9 mm/s as
tie-break; retain earliest strict minimum epoch, patience 8, maximum 50 epochs.
Then repeat only the selected setting with seeds 2027/2028. Report all seeds
and mean/population standard deviation; use seed 2026 for the representative
trace instead of selecting a seed. Budget four fits / 200 epochs / 10,400
updates and a shared 20-minute deadline checked between epochs. No target
scaling, synthetic memorization settings, new architecture or extra candidates.

All fits completed within budget. lr=0.003 selected; all three seeds remain
worse than ridge on identical validation targets. Preserve the result. A
generalization gap, limited prediction spread and saved-head output bounds
explain some observed limitations without proving a single cause. Validation
still improves overall near the epoch cap, so do not claim convergence or a
proven sustained overfitting onset. Keep final test closed. Next increment
verifies offline/sequential replay and measures processing latency before video.

## 2026-11-17 — Sequential replay before disruption tests or video

Retain the exact saved models, training transforms, ten-bin histories and
Phase 2D quality/identity metadata. Reveal raw completed count bins one at a
time; reuse existing singleton feature adapters and model APIs. GRU starts at
zero per bounded window. Kalman updates only common eligible rows, carrying
its own estimated state within a continuous run and resetting to the training
prior after invalid time/boundaries, with no extra warm-up observations.

Declare combined absolute/relative equivalence tolerances before results and
gate timing on all-model agreement. Use CPU batch size one, 200 warm-ups and
one 2775-emission timed pass, retaining every outlier. Report checked model API
and completed-count-bin preparation-plus-model as separate boundaries; exclude
loading, event binning, disk, acquisition, plotting and playback sleep. The
first run overlapped tests, so preserve it and use a separate clean timing run
after tests, rather than selecting the fastest observations. Report first calls
without describing them as fresh-process cold starts.

Equivalence passed. Offline GRU batch-one predictions are exactly equal to
replay; small differences from saved batch-256 outputs reflect float32 arithmetic.
Warmed p95 is below 20 ms, but all models have maxima above it. Preserve this
limitation and make no hard real-time claim. Next increment tests predefined
input disruptions before producing cursor video from verified predictions.

## 2026-11-18 — Fixed disruption round with coverage-aware scoring

Declare ten validation-only conditions before inference: clean, nested
10/24/48-channel loss, two thinning retentions, two training-rate Poisson levels
and one/five-bin missing bursts. Hash-select physical channels from source and
IDs, retaining silent selections. Key count RNG by source/bin/channel identity;
apply raw modifications once per source packet and reuse across all models.

Represent missing explicitly as missing=True/counts=None/invalid source.
Retain Phase 6A's uniform skip/reset behavior for this round instead of adding
unvalidated imputation or continuous Kalman prediction-only carry. Predeclare
the union-of-missing/recovery exclusion mask for all conditions and pair each
available-row score with clean on those exact rows. Report coverage prominently:
no-emission intervals are not evidence of decoding accuracy. No tuning after
scores or extra candidates. All declared conditions completed; slight gains
under two disruptions are retained without claiming noise/removal is beneficial.
Stop after Phase 6B; next is visual cursor replay from verified predictions.

## 2026-11-19 — Audited reference integration for fixed validation video

Use the existing first500 validation prediction ends in [1550.200,1560.200),
without selecting a favorable segment or rerunning models. Audit original tracker
position differences, target labels and actual durations before plotting.
Forty behavioral intervals differ from neural-bin durations; exact original
tracker-position alignment is not verified. Display saved target velocity
integration with that explicit label, not purported original positions.

Anchor all paths once at relative(0,0) at 1550.180 s and integrate every selected
velocity over its actual neural-bin duration. Show500 endpoint frames at 50 fps,
using the same fixed physical scale for reference/ridge/GRU/Kalman and retaining
drift. Do not correct, smooth, shift or reset decoded positions. Reject gaps.
Show full-validation velocity scores separately from local integrated distance.
Export video/still and provenance under ignored runs/, inspect actual decoded
first/middle/last frames, and commit only source/config/tests/docs. No test-set
opening, model tuning, live-control implication or new processing-time claim.
