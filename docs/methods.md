# Causal timing, chronological evaluation and velocity alignment

## Recording and scope

The plan uses Indy session indy_20161005_06, Zenodo record 583331 v1, with the
SHA-256 in [the manifest](../data/manifest.json). Behavior spans 1288–1662 s
(374 s), with 4 ms sampling and no detected timestamp gaps. Global neural event
extrema are 1283.0023444054618–1666.9959174171197 s. Shared coverage is therefore
[1288, 1662). Event extrema do not prove uninterrupted acquisition; silence is
not an outage indicator. Physical synchronization and the finger/cursor
discrepancy remain unresolved in [the dataset card](dataset.md).

Phase 2A planned boundaries without count bins. Phase 2B now constructs causal
channel counts, and Phase 2C adds aligned velocity targets. Transforms, history
arrays and model predictions remain unimplemented. The saved specification's
model-window/scoring indices remain unmaterialized; target validity is explicit.

## Timing contract

[configs/base.toml](../configs/base.toml) sets Δ = 0.020 s, ten consecutive bins
of history (0.200 s), zero additional lag, and zero forecast horizon.
The fixed bin grid has origin 1288 s and ends origin + k × Δ, integer k ≥ 1.
Later construction must check numerical edges; never shift bins to align movement.

For an emission associated with bin end t:

| Quantity | Support or availability |
| --- | --- |
| One neural bin | Spike events in (t−Δ, t]; a right-edge event belongs here, not to the next bin. |
| Neural feature history | Ten bins covering (t−10Δ, t]; nothing after t. |
| Nominal target interval | Just-completed movement from t−Δ to t. |
| Actual target source samples | Latest stored behavioral timestamp b0 ≤ t−Δ and b1 ≤ t, each at most 6 ms old. |
| Conservative target source envelope | [t−Δ−0.006−0.000000001, t], including age-comparison tolerance. |
| Prediction availability | At or after t plus actual computation delay; latency remains unmeasured. |

Phase 2C computes (p(b1) − p(b0)) / (b1−b0) in mm/s, separately for x/y.
This explicitly supersedes Phase 2A's proposed fixed 0.020 s denominator.
The actual selected interval and its difference from nominal are recorded under
the validity policy below. It estimates average velocity over that past sampled
interval, associated with the just-completed neural bin; it is not forecasting.
Do not interpolate using later samples. Positions provide targets and scoring
references, never neural decoder inputs.

Select each behavioral sample using searchsorted(side="right") − 1.
The fixed maximum age is 6 ms, 1.5 times the source-documented 4 ms period.
This accommodates sampling phase while rejecting stale positions. The same
fixed 6 ms threshold detects behavioral discontinuities. A 1 ns floating-point
tolerance applies only to age/gap comparisons, never to allowing future samples,
expanding partition ownership, or repairing timestamps. Strict support comparisons
may conservatively reject a numerically marginal boundary. Exact eligible indices
and bin-edge handling remain pending construction.

### Concrete source-time example

At t = 1288.200 s, features may use spikes in (1288.000, 1288.200]; the final
bin is (1288.180, 1288.200]. On this recording the helper selected behavioral
indices 45 and 50, at exactly 1288.180 and 1288.200 s. The target uses
those two x/y positions divided by their actual elapsed seconds (approximately
0.020 s here), under the effective Phase 2C policy. Its conservative source envelope
begins at approximately 1288.174 s (1 ns earlier with the tolerance). Nothing after 1288.200 s is allowed. That Phase 2A check verified timestamp
selection only; the Phase 2C calculation and positions are documented below.

For a small timeline example, an emission at 10.200 s requires neural history
(10.000, 10.200] and positions associated with 10.180 and 10.200 s. If validation
starts at 10.100 s, that emission is invalid there because its history crosses
the boundary. It must wait for complete history collected inside validation.

## Chronological partitions

Training learns parameters; later validation selects settings; final test data
evaluate the frozen choices. Randomly mixing nearby windows can share support
between partitions. Leakage also includes fitting scaling or channel-selection
policies with validation/test data. Future learned preprocessing must fit on
training only.

Nominal 70/15/15 fractions refer to **elapsed shared-coverage time**, including
any gaps, not sample counts or future usable examples. Gaps do not move boundaries.
The longest conservative lookback is:

max(10 × 0.020, 0.020 + 0.006 + 0.000000001)
= max(0.200, 0.026000001) = 0.200 s.

Remove this duration on **each side** of both nominal boundaries: a 0.400 s
purge gap at each boundary. This is a deliberately conservative support-based
guard, not proof that temporal dependence disappears.

| Partition | Nominal interval (s) | Retained interval (s) | Purge discard |
| --- | --- | --- | --- |
| Training | [1288.000, 1549.800) | [1288.000, 1549.600) | 0.200 s |
| Validation | [1549.800, 1605.900) | [1550.000, 1605.700) | 0.400 s |
| Test | [1605.900, 1662.000) | [1606.100, 1662.000) | 0.200 s |

Retained elapsed durations are 261.6, 55.7, and 55.9 s. Purge discard totals
0.8 s; detected gaps discard no additional time here. Displayed values are rounded;
[the JSON specification](../data/splits/indy_20161005_06.json) stores authoritative
machine endpoints, including floating-point representation.

All partitions and continuous segments use [start, end): start belongs, end does
not. Prediction ends and all source samples must belong to the same interval.
This conservatively excludes an emission at terminal overlap endpoint 1662 s.
Raw data are unchanged. Spike bins separately use (start, end].

A **purge gap** removes source time around a boundary from every partition.
A **warm-up** collects history inside a retained partition/segment before an
emission is eligible. Warm-up events may contribute to the first complete
history; they are not another purge. Required warm-up is the maximum of neural
history and target-source lookback, 200 ms here. Complete-support lower bounds
are 1288.200, 1550.200, and approximately 1606.300 s. These are not guaranteed
eligible emissions: grid, source ages, and support checks must still pass.
Do not fabricate indices or assume usable-example proportions equal time fractions.

Every future window/target must use one retained partition and one continuous
segment. No cached history, target sample, or future model state may cross
boundaries. Reset recurrent/Kalman state at session, partition, and segment starts
using a later specified neutral or training-derived initialization; held-out
movement must not initialize the decoder. All models must use identical eligible
timestamps.

## Gaps and prior exploration

An adjacent behavioral interval >6 ms + 1 ns splits continuity. Exclude
[last sample before, first sample after), start a segment at the first sample
afterward, and require fresh warm-up. This conservatively excludes the pre-gap
terminal sample as an emission. Never bridge, sort, repair, or search for a
better-looking interval. Short fragments remain recorded but cannot form full
windows. For a conservative pre-construction feasibility check, each partition
must have a fragment longer than lookback plus one bin, ensuring temporal room
regardless of grid phase. This is not extra warm-up for every emission and does
not certify sample-age eligibility. A partition without that room fails clearly.
Known acquisition outages discovered later must likewise invalidate intersecting
support, split segments, and reset state before feature generation. Neural outages
cannot currently be detected from spike silence alone.

Phase 1C explored [1288, 1298) s. Its reviewed artifact JSON matches the source
SHA-256 and interval. It lies entirely inside retained training: no conflict.
The config/specification preserve that ledger; generation rejects a conflict.
No held-out movement patterns or correlations determined splits. Whole-file
schema/finite-value auditing is distinct from examining movement patterns to
choose favorable evaluation boundaries.

## Reproduction and limits

From the repository root after setup and verified acquisition:

~~~powershell
uv run --locked python scripts/plan_splits.py
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
~~~

The command verifies source SHA-256, reuses the existing loader, and saves source
identity, configuration fingerprint (UTF-8 text normalized to LF), coverage,
grid, endpoints, purge/gap/warm-up/reset rules, and the exploration ledger under
data/splits/. Identical reruns reproduce the same specification; an existing
different specification is refused for explicit review. The small JSON is
trackable; raw recordings and runs remain ignored.

Synthetic tests check ordering/disjointness, elapsed proportions, endpoints,
support, past-only selection, stale samples, gap resets, short timelines, and
invalid settings. Real-file planning checks source identity and timing metadata
separately. These checks do not establish biological lag, uninterrupted neural
acquisition, decoding quality, or correct future feature/target construction.
Eligible model-window/scoring indices remain null. The Phase 2B binning work
and current next increment are documented below.

## Causal spike-count construction — Phase 2B

The implemented binning function is
src/neurodecode/data/binning.py:bin_partition(recording, specification, config,
partition="train"). It reuses the loader's raw event vectors and TimingConfig.
It returns BinnedSpikes, with int64 counts [T, C], bin_start_s/bin_end_s [T],
ordered channel_ids [C], source_channel_indices [C], included unit_indices [C],
partition name, per-row segment_ids and integer grid_end_indices, and
segment edge-exclusion metadata. Counts are events per bin, not events/second,
normalized features, or single-neuron measurements.

### Exact grid and edge policy

Use the saved origin (1288 s here) and configured width (0.020 s). Compute each
edge independently as origin + integer_index × width in float64. Bin k has
start g(k−1) and end g(k), so adjacent bins share exactly the same stored edge.
Never generate edges with cumulative addition or independently subtract the
width from every end; those approaches can disagree at shared edges through
floating-point rounding. Small float differences between end−start and 0.020
are representation effects, not variable requested bin widths.

Estimate a generous range of integer indices, then retain only candidates with
start ≥ continuous-segment start and end < continuous-segment end. Compute bins
separately within each saved segment/partition on the same global grid.
Drop partial head/tail bins and any full bin ending exactly at the excluded
segment endpoint. Do not shorten bins, snap boundaries, or introduce a local
grid. This preserves Phase 2A's half-open ownership, including its authoritative
machine endpoints. Strict comparison can conservatively drop a marginal bin.

For each original unit vector, count:
number of events ≤ end minus number of events ≤ start.
Both searches use searchsorted(side="right"). Thus event inclusion is exactly
(start, end]. No event after an end, even one float step later, is moved backward
by an epsilon. There is no spike-time rounding or tolerance in event counting.
Each source-list entry is counted once in its bin. Source duplicate entries
remain separate entries; no within/across-unit deduplication is invented.

Hand example: events [1.000, 1.010, 1.020, 1.025] give counts 2 in
(1.000, 1.020] and 1 in (1.020, 1.040]. The shared-edge event 1.020 belongs to
the earlier bin only. The first boundary event 1.000 is excluded from these bins.

### Channels, units, gaps, and warm-up

Select all identifiers beginning with the verified "M1 " prefix if any exist,
and sort them lexically. Otherwise use all identifiers in lexical order and
report that explicit fallback. Selection is identity-based, never based on
full-recording activity, movement, correlation, or held-out performance.
This session yields M1 001–096, preserving quiet and all-empty channels.

Aggregate every available source slot once into its recording channel, including
explicit empty slots. Python indices 0–4 correspond to source U1–U5 here.
The [official dataset description](https://zenodo.org/records/583331) identifies
U1 as residual unsorted threshold crossings left after sorting other units.
It is not a copy of all sorted events. Preserve the included slot indices aligned
with each channel ID; U1 plus U2–U5 form channel activity, not one identified neuron.
No claim of stable neurons or cross-session unit identity is made.

Validate disjoint ordered retained partitions, ordered nonoverlapping segments,
declared coverage, and actual behavioral timestamp gaps against the fixed
6 ms + 1 ns policy. Refuse a stale segment that crosses a detected gap rather
than silently repairing it. Known invalid intervals and purge gaps have no
rows. Genuine valid zero-spike bins do have rows with zeros. A fragment too
short for a complete grid bin has an empty row range and exclusion metadata.
Never infer neural packet loss from spike silence. There is no packet-quality
mask in the currently verified schema; undiscovered acquisition gaps remain a
limitation.

Keep warm-up count bins: later eligible histories need their events. These count
rows are not yet model-ready windows or scoring indices. History warm-up,
behavioral sample-age checks, target alignment, and common eligibility remain
separate later operations. No velocity arrays or histories are constructed here.

### Reproduction, provenance, and cache policy

From the repository root:

~~~powershell
uv run --locked python scripts/bin_recording.py
~~~

This command processes **training only**. It verifies recording SHA-256 and
source/version/file identity against the manifest and saved specification,
verifies the configuration's UTF-8/LF fingerprint, and recomputes timestamp-only
split planning to check the saved contract before binning. It prints source,
split (canonical-JSON SHA-256), configuration, grid, channel/slot policy, shape,
segment exclusions, and a fixed early training example. Direct API callers must
supply the verified source-bound specification; the script performs the full
provenance checks.

There is **no processed-array persistence or cache reuse** in Phase 2B. Every
invocation reconstructs counts in memory from verified bytes. No bulk arrays are
written to Git or disk. If persistence is added later, files must be under ignored
data/processed/ and bind source checksum, split/configuration identity, effective
binning policy, and complete channel/unit mapping before any cached reuse;
that cache implementation is deferred.

### Actual training output and verification

Training counts have shape **[13079, 96]**: 13,079 chronological complete 20 ms
bins and 96 recording-channel columns. There is one valid continuous segment.
The first bin is (1288.000, 1288.020]; the last is
(1549.560, 1549.580]. The head exclusion is 0 s; the tail exclusion is approximately
0.020 s because a bin ending at 1549.600 would violate ownership.
Warm-up bins remain present.

For M1 001, the first three counts are 0, 2, 1. In its second bin,
(1288.020, 1288.040], U1 contains event 1288.033454282609 s and U3 contains
1288.0246070068824 s; U2/U4/U5 contribute zero. Their aggregate is 2.
The following bin contains U3 event 1288.0577023718893 s, giving 1.
This is a deterministic early example, not selection for apparent decoding quality.

Independent boolean inclusion masks over the original unit lists matched count
sums separately for every selected channel: **219,913 included training events**.
This count covers only retained complete bins, not the entire raw recording.
No real validation/test activity patterns or arrays were inspected.

Twenty-two new synthetic cases verify hand counts, exact/shared edges, duplicate
list entries, empty trains, zero rows, multiple units, deterministic channel
ordering and fallback, independent conservation (including disjoint segments),
causality under future-event mutation, one-float-step future events, integer-grid
construction, incomplete/excluded edges, gaps, ownership, and invalid inputs.
All 77 tests, including 55 retained checks, passed; lint and formatting passed.
These checks establish count extraction and tested boundary behavior, not
physiological alignment, neural continuity, decoding performance, or velocity
target correctness. These are Phase 2B results; Phase 2C is documented below.

## Velocity target alignment — Phase 2C, October 11, 2026

### Effective denominator and sampled support

Phase 2C explicitly supersedes Phase 2A's fixed nominal-width denominator.
For source samples at b0 and b1, compute:

v = (p(b1) − p(b0)) / (b1 − b0), independently for x/y, in mm/s.

Nominal neural bins remain 20 ms. Record actual elapsed_s = b1−b0,
elapsed_deviation_s = elapsed_s−0.020, and both source sample ages. The label
is average velocity over the actual selected sample interval [b0, b1], associated
with the just-completed neural bin; when ages differ, that interval differs from
the nominal bin. This is an approximation to movement over the nominal interval,
not noiseless instantaneous velocity or future forecasting.

Use the latest stored sample at or before each stored bin edge, with
searchsorted(side="right")−1. The left edge is the bin's shared grid edge g(k−1),
mathematically t−Δ, instead of independently subtracting Δ from a float end.
This follows Phase 2B's shared-edge rule and prevents one-rounding-step
disagreements between neural and behavioral boundaries. There is no future-sample
epsilon or interpolation. Ages must be nonnegative and ≤6 ms + 1 ns.

A pair is valid only if actual elapsed time is positive and
abs(elapsed_s−nominal_bin_s) ≤ max_behavior_age_s + 2 ns, currently 6.000002 ms.
The age difference bounds the departure from bin width; the first ns is the
existing sample-age tolerance, and the second accommodates nominal-grid
floating-point representation. The source samples and partition ownership are
never moved by these comparison tolerances. Reject a pair outside the bound;
do not clamp its denominator, replace it with 20 ms, or rescale velocity.
This policy was fixed before auditing any held-out outcomes. The source support
bound is unchanged, so saved purge distances and boundaries remain valid.

For intuition, displacement (2, −1) mm across selected timestamps 16 ms apart
gives (125, −62.5) mm/s. Dividing by nominal 20 ms would instead give
(100, −50) mm/s. Differences magnify position noise: a 0.1 mm displacement error
over 20 ms produces a 5 mm/s velocity error; shorter intervals amplify it more.
No smoothing or noise correction is implemented.

### Identity, validity, and partition rules

align_velocity_targets in src/neurodecode/data/targets.py returns AlignedTargets.
Its bins field retains the original BinnedSpikes object and all count rows,
channels, bin edges, global integer end indices, partition and segment IDs.
velocity_mm_s is [T, 2]; valid_mask is [T]; reasons records "valid" or the first
exclusion reason. source_indices, source_time_s, and sample_age_s are [T, 2],
ordered as left/right selections. elapsed_s and elapsed_deviation_s are [T].
Unavailable indices are −1; unavailable timestamps/ages and invalid velocities
are NaN. Invalid labels are never replaced by zero velocity. valid_rows identifies
original count rows and preserves their explicit bin identities.

Reuse bin_partition to validate supplied counts against the verified recording
and specification before alignment. Fail on mismatched shapes, row order, counts,
timestamps (even one float step), global/segment IDs, channel/unit mapping, or
segment metadata. Never trim, sort, silently reorder, or align solely by length.
This intentionally recomputes counts for validation; it is preparation logic,
not an inference-latency benchmark. Use one shared mask:

~~~python
targets = align_velocity_targets(recording, bins, specification, config)
count_rows = targets.bins.counts[targets.valid_mask]
velocity_rows = targets.velocity_mm_s[targets.valid_mask]
original_rows = targets.valid_rows
~~~

The masked example above is a paired single-bin view, not a history-building
input. Future history construction must use the full original bin sequence and
segment metadata; windowing filtered rows would stitch across exclusions and
lose warm-up source events.

Count rows include warm-up source data. Target eligibility additionally requires
the full configured support inside the same retained segment: ten consecutive
grid bins here, with their left history edge and conservative behavioral source
envelope not preceding segment start. This checks support/cardinality without
constructing a history array. Prefix targets are marked warmup; every new segment
starts fresh. Future history construction must respect the same original IDs
and mask, including any unavailable data-quality rows.

Malformed/nonfinite, duplicate or nonmonotonic behavioral timestamps and mismatched
position dimensions are structural errors: fail clearly before searching.
Existing loader rejection of nonfinite raw positions remains unchanged.
For a directly supplied Recording with a nonfinite selected position, invalidate
that target row; unrelated future position values cannot change an earlier label.
Finite selected positions whose difference/division overflows also invalidate
the row rather than emitting infinity.

The first row reason follows this order:

1. missing_past_sample: no sample exists at/before a boundary.
2. cross_segment_sample: a selected sample belongs outside the same retained segment.
3. stale_sample: a selected sample's age violates the fixed limit.
4. nonpositive_elapsed: the two selected sample times do not provide positive elapsed time.
5. elapsed_deviation: elapsed time departs too far from the nominal bin width.
6. warmup: incomplete local neural/behavioral support.
7. nonfinite_position: a selected source coordinate is not finite.
8. nonfinite_velocity: the arithmetic result is not finite.

A stale saved segment crossing an actual behavioral gap fails clearly and requires
replanning/rebinning under the existing >6 ms + 1 ns policy. Properly planned
gaps have no count rows and restart warm-up. Source samples cannot cross partition
boundaries or purge gaps. There is no interpolation, sorting, repair, smoothing,
normalization, learned channel exclusion, or behavioral model input.

### Reproduction and actual results

~~~powershell
uv run --locked python scripts/bin_recording.py --targets
~~~

The existing command retains its no-flag Phase 2B training-count behavior.
The new flag reuses source SHA-256/configuration/split verification, derives
targets independently in each partition, and prints only validity/timestamp
quality counts for validation/test. Examples are the first three valid training
rows in fixed identity order. All arrays remain in memory; no cache is saved
or reused. Held-out target preparation is a deterministic quality audit, not
model scoring or movement-pattern inspection; no parameters or boundaries were
tuned from those partitions.

| Partition | Count rows retained | Valid aligned targets | Excluded reason |
| --- | --- | --- | --- |
| Training | 13079 | 13070 | 9 warmup |
| Validation | 2784 | 2775 | 9 warmup |
| Test | 2793 | 2784 | 9 warmup |

Actual selected elapsed times depart from nominal by >1 ns in 22 valid training
rows, 222 validation rows, and 222 test rows. These are timestamp-quality counts,
not movement statistics. Every deviation is retained, and all pairs satisfy
the predeclared age/separation bounds. Do not silently force them to 20 ms.

The first valid training row is original count row 9, global end index 10,
segment 0, neural bin (1288.180, 1288.200]. Its first three channel counts are
[1, 1, 0]. The selected source positions are:

| Source time (s) | Cursor x (mm) | Cursor y (mm) |
| --- | --- | --- |
| 1288.180 | −7.882167580583823 | 30.560668915346305 |
| 1288.200 | −7.945765364028281 | 30.651556978673803 |

Displacement is (−0.06359778344445832, 0.09088806332749755) mm.
Actual elapsed time is 0.01999999999998181 s, differing from nominal by
−1.81903103690928e−14 s. Dividing displacement by that elapsed time yields
(−3.179889172225808, 4.54440316637901) mm/s. Both sample ages are zero here.
The negative x value means movement toward decreasing x, not negative speed.

Twenty-four new synthetic target tests cover constant velocity and units,
irregular sampling/actual denominator, exact/future boundaries, future-value
mutation (including nonfinite future positions), stale samples, gap resets/refusal,
purge ownership, nonpositive separation, nonfinite positions/arithmetic,
malformed clocks/shapes, identity/content mismatch and empty fragments.
All 101 tests passed, along with lint, formatting and lock consistency.
This verifies implemented target arithmetic and tested causal pairing, not
physiological synchronization, noiseless movement, model quality or inference
latency. Next increment: Phase 2D, fit count normalization on training data only.


## Causal histories and training-only transforms — Phase 2D, October 11, 2026

### Window order, eligibility and provenance

Reuse AlignedTargets and its original BinnedSpikes; do not parse or independently
realign source data. build_history_windows in src/neurodecode/data/features.py
uses configured L=10, Δ=0.020 s, the unchanged global grid origin 1288 s, and
the saved partition/segment contract. For original target row r, use count rows
r−L+1 through r, oldest to newest. The target belongs to the final bin.
Return integer X_raw [N, L, C] and y [N, 2] in mm/s. C=96 here, in the established
M1 001 through M1 096 order, with every source unit slot still aggregated within
its channel. An aggregate channel is not a single neuron.

Retain original source_rows [N,L], source_bin_end_indices [N,L], every source
bin's start/end timestamps [N,L], target_rows [N], target_bin_end_indices [N],
target_time_s [N], segment_ids [N], channel order, contributing unit slots,
partition and source/split/configuration/target-policy identity. The first and
last source identities are columns 0 and L−1; target identity equals the last.
Do not window a previously filtered count matrix or infer pairing by length.

Require all L original source rows to be valid, have consecutive global integer
grid IDs and exactly shared edges, and belong to the same retained partition
and continuous segment. Require the final target to be valid. Validate full row
lengths, finite valid labels, target mask/reason agreement, exact grid timestamps,
channel/unit mapping lengths, and ordered segment ownership. Fail on mismatch;
never sort, trim, repair, interpolate or shift source data.

A warmup target exclusion represents an unavailable label with valid source
counts. Such prefix counts can supply the first complete history; otherwise
warm-up would accidentally be imposed twice. Other target-quality exclusions
conservatively invalidate that row for subsequent neural histories too.
An optional explicit boolean bin_valid_mask can additionally mark known
missing/invalid observations. An all-zero count row is valid and does not mark
a gap. Properly planned discontinuities have no rows, and must not be bridged.

Drop incomplete prefix windows in each segment. After an invalid or missing
bin, require L fresh consecutive valid bins; do not stitch across the excluded
row. Summary rejection reasons are warmup (incomplete prefix or upstream
warm-up exclusion), invalid_target, discontinuous_history, and
invalid_history_bin, in that precedence. Each excluded target row receives one
reason; invalid_windows excludes the separate warm-up count. Missing source
time has no candidate row and is not counted as a fabricated zero row.

For an exploratory hand example with L=3 and channels A/B:
rows [0,1], [2,3], [4,5] produce one window
[[0,1], [2,3], [4,5]], paired with the final row's velocity.
flatten_features uses C-order reshape: [0,1,2,3,4,5], oldest lag first with
channels varying fastest. unflatten_features explicitly checks L and C and
reverses it. Actual [N,10,96] becomes [N,960]; no ridge model is trained.

### Normalization fit, application and replay metadata

Fit per-channel statistics over unique valid training source-count rows exactly
once, including valid warm-up bins. Do not fit over repeated entries from
overlapping windows, which would weight middle bins more than edge bins.
The same source-validity policy applies to the fit. No held-out activity or labels
select channels, fit statistics, tune rules or change order.

For M valid original training bins and channel c:

- μ_c = sum_i count[i,c] / M.
- σ_c = sqrt(sum_i (count[i,c]−μ_c)^2 / M), population standard deviation (ddof=0).
- s_c = σ_c when σ_c > 0, otherwise 1.
- z[n,l,c] = (X_raw[n,l,c]−μ_c) / s_c.

The same channel transform applies at all lags. Means/scales are in events per
20 ms bin; normalized features are dimensionless. Do not normalize y, which
remains mm/s. Constant/silent training channels stay in their original positions.
Their training normalized values are zero; later nonconstant activity uses the
same frozen mean and scale 1 rather than being discarded.

fit_count_scaler rejects a non-training partition. FrozenCountScaler stores
read-only means, scales and constant-channel flags. transform returns a new
floating array without overwriting X_raw, source counts, targets or fitted
statistics. Reject source/split/configuration/target-policy, channel/unit order
or history-dimension mismatch when applying it. Raw counts remain available
for the later Kalman observation path and robustness work.

This is an offline fit over the whole designated training partition. Changing
later observations within training and deliberately refitting can legitimately
change statistics. That differs from deployment causality: once fitted, incoming
future counts and held-out values cannot update the frozen scaler or an earlier
history. The tests check causal raw windows and frozen transformations separately;
they do not claim an offline refit is causal within its training interval.
Fitting on validation/test would leak their activity distribution into learned
preprocessing, even without reading their labels.

The reproducible command saves runs/phase2d/count_scaler.json, ignored by Git.
Metadata retain all fitted means/scales, constant flags, ordered channel IDs and
unit slots, L, source SHA-256, canonical split SHA-256, LF configuration SHA-256,
target-policy SHA-256, feature implementation SHA-256, fit basis, formulas,
units and flattening order. Two additional SHA-256 values identify included
training (segment ID, global bin ID) pairs and original fitted counts, encoded
as row-major little-endian int64 bytes. Future replay must preserve these
identities and conventions and use the frozen training statistics.

No bulk processed arrays or window cache are saved/reused. The command always
recomputes from the verified recording and unchanged split using existing
components. An identical metadata rerun is accepted; a differing file at the
output path raises an error instead of overwriting existing work. Preserve and
review the old output before intentionally choosing a new fit.

### Reproduction, real example and limits

~~~powershell
uv run --locked python scripts/bin_recording.py --features
~~~

| Partition | Valid source bins | Retained windows | Warm-up exclusions | Other invalid windows | X shape | y shape |
| --- | --- | --- | --- | --- | --- | --- |
| Training | 13079 | 13070 | 9 | 0 | [13070,10,96] | [13070,2] |
| Validation | 2784 | 2775 | 9 | 0 | [2775,10,96] | [2775,2] |
| Test | 2793 | 2784 | 9 | 0 | [2784,10,96] | [2784,2] |

Raw and normalized X share the same shapes. Flattened shapes are [13070,960],
[2775,960], [2784,960]. Held-out outputs are counts/shapes only; examples come
from the first three eligible training target identities. No held-out movement
patterns, activity summaries, scores or tuned rules are reported.

First training example: partition train, segment 0, source bin IDs 1–10,
target row 9 and end ID 10. Ten source bin ends are
[1288.020,1288.040,1288.060,1288.080,1288.100,1288.120,1288.140,1288.160,
1288.180,1288.200] s. Each begins at its preceding shared edge, starting at
1288.000 s. The raw feature shape is [10,96], covering (1288.000,1288.200].
Its aligned target is [−3.179889172225808,4.54440316637901] mm/s,
using the Phase 2C selected positions at 1288.180 and 1288.200 s.

M1 001 counts in that history are [0,2,1,2,1,1,2,2,1,1].
Its training mean is 0.12890893799220124 and population scale is
0.36460280279149737. The oldest feature is therefore
(0−0.12890893799220124)/0.36460280279149737 =
−0.35355992056352736. A negative normalized value means below the training
mean; it does not mean a negative event count. Eleven channels are constant
in training and retained: M1 009, 047, 055, 058, 061, 068, 071, 078, 079, 082, 096.

Twenty-four new synthetic checks cover hand-calculated order/final-target pairing,
reversible flattening, silence versus missing/invalid bins, gaps/partition
ownership, future-count mutation, frozen held-out application, unique-bin fit,
constant channels, metadata, malformed identity/shapes and empty windows.
All 125 tests passed, including 101 retained tests; Ruff lint/format and lock
consistency passed. The real command ran successfully and reproduced identical
metadata on rerun. These checks establish tested preparation behavior and actual
tensor eligibility, not decoding performance, physiological timing, independent
samples, noise-free targets, or latency. Overlapping windows remain correlated.

No models, target scaling, feature smoothing or evaluation metrics were added.
Next: Phase 3A, establish zero-velocity and training-mean baselines and define
evaluation metrics.


## Trivial predictors and shared validation evaluation — Phase 3A, October 11, 2026

### Predictor inputs and fixed example protocol

Reuse the existing bin_partition → align_velocity_targets → build_history_windows
path unchanged for train and validation. The source/configuration/split checks
from scripts/bin_recording.py are extracted into load_verified_inputs so the
baseline entry point uses the same verification code. There is no second parser,
alignment rule or eligibility calculation. The split specification and its
historical pending-index field remain unchanged; Phase 2D constructs actual IDs.

Training/validation prepared tensors remain [13070,10,96]/[2775,10,96], paired
with [13070,2]/[2775,2] mm/s labels. Baselines do not use neural values or fit
feature transforms. Preserve channel/unit order and source/split/configuration/
target-policy identity even though these trivial predictors ignore features.

Zero velocity emits [0,0] mm/s for every eligible timestamp.
TrainingMeanVelocity.fit accepts the training HistoryWindows only and takes the
unweighted two-axis mean over all eligible training targets once per target.
Its read-only frozen mean is broadcast at validation timestamps. Validation
predict accepts an ExampleIndex containing timestamps, global bin IDs,
segment IDs, scoring mask, partition and preparation/channel/unit context,
with no labels or positions. Changing validation target values cannot refit or
change predictions. Generate both prediction arrays before accessing validation
labels for scoring. No held-out velocity is used as a predictor input.

ExampleIndex and VelocitySeries in src/neurodecode/evaluation.py make identities
explicit. Validate [N,2] velocities, finite [N] timestamps, strictly increasing
timestamps/global bin IDs, nonnegative ordered segment IDs, channel/unit context
and boolean scoring mask. Require exact prediction/target timestamps, IDs,
segment IDs, partition, preparation identity, channel order and scoring masks.
Never trim, sort, interpolate, silently intersect row lists or round timestamps.
Prepared target identity must equal the final source bin's identity/time.

All Phase 2D eligible validation rows are scored, with an all-true mask after
preparation. Any future explicit quality mask must be identical across compared
models; masks cannot depend on residual magnitude or model success.
All input velocities must be finite, including unscored rows. Reject nonfinite
inputs and arithmetic overflow rather than silently dropping observations.
Zero scored rows are an error.

### Metric definitions and undefined values

For K scored targets y[i,a] and predictions p[i,a], a in {x,y}, in mm/s:

- e[i,a] = p[i,a]−y[i,a].
- RMSE_a = sqrt(sum_i e[i,a]^2 / K), in mm/s.
- Overall RMSE = sqrt(sum_i sum_a e[i,a]^2 / (2K)), in mm/s.
  Equivalently sqrt((RMSE_x^2+RMSE_y^2)/2), not the arithmetic mean of axis RMSE.
- R²_a = 1−sum_i e[i,a]^2 / sum_i (y[i,a]−mean_i y[i,a])^2.
- Mean R² = (R²_x+R²_y)/2, unweighted; it is dimensionless.

The scoring-set mean in the R² denominator is an evaluation reference, not a
predictor input. R²=1 indicates perfect predictions for nonconstant targets;
R²=0 matches the scoring-set mean reference's squared error. Negative R² means
worse squared error than that reference, not a forbidden score.

A constant target axis has undefined R², even for a perfect prediction. Return
None/JSON null and status constant_target; do not fill 1 or 0. Fewer than two
scored examples gives null with insufficient_scored_examples. If either axis
is undefined, mean R² is also null; never silently average only the defined axis.
RMSE remains defined for one finite example and for constant targets.

Hand example: targets [[0,0],[2,4]], zero predictions.
Axis SSE is [4,16], RMSE is [sqrt(2),sqrt(8)], overall RMSE=sqrt(5).
Target-axis means [1,2] yield SST [2,8], so both R² values are −1.
Perfect predictions for nonconstant axes give RMSE=0 and R²=1.

### Validation plot, exports and limits

~~~powershell
uv run --locked python scripts/evaluate_baselines.py phase3a-review
~~~

Choose a new run ID under ignored runs/. Existing directories are preserved
with an error. Save run.json (effective settings, source/full split, configuration
and code hashes, channel/unit order, mean fit, eligibility, Git commit/dirty state,
environment, plotting rule and whole-run duration), metrics.json, timestamped
predictions.csv and validation_velocity.png. The CSV includes actual targets,
both predictions, exact end IDs/timestamps, partition/segment, scoring mask and
first/last history IDs. No code/data credentials or absolute recording paths
are written. Duration covers preparation/scoring/export, not inference latency.

The fixed plot selects [first eligible validation timestamp, +10 s), before
inspecting label values, currently [1550.200,1560.200) s with 500 rows.
Reject missing grid IDs, segment changes, masked rows or insufficient coverage
in that interval; do not seek a better-looking alternative. Both axes share
the same time offset. Plot unsmoothed derived velocity and constant predictions
with mm/s labels; flat lines represent values that never change. No trajectory
integration, interpolation, neural smoothing or signal shifts are added.

Validation is now available for model/metric development. This command never
constructs test targets/windows, predicts test, displays test or scores test.
The common loader necessarily reads the acquired recording and verifies global
coverage metadata as before. Previous deterministic test preparation audits
are documented; final test performance has not been opened. Freeze the later
recipe before a separately recorded final test-opening event.

The [Phase 3A validation report](../reports/phase3a_baselines.md) records measured
scores only after source/configuration/split verification and export auditing.
All 2775 validation targets span 1550.200–1605.680 s; 13070 training targets fit
the mean. Predictions are fixed [0,0] and
[−0.1479870893879215,0.1386043376635596] mm/s. Overall validation RMSE is
72.81748587747391 / 72.8176536401862 mm/s for zero / training mean respectively.
Both mean R² values are slightly negative, near zero; these are validation
references, not final test results or evidence of neural decoding.

Twenty-eight new synthetic tests cover known metrics, perfect/negative/undefined
R², shared masks, nonfinite/overflow failures, exact shapes/timestamps/IDs/context,
training-only frozen means, final-bin identity and value-independent plot
selection with gap refusal. All 153 tests passed, including 125 retained checks,
plus lint, formatting, lock consistency and the unchanged training count command.
Exported scores were independently recalculated from CSV; a second run produced
identical predictions/metrics. The actual figure was visually inspected.
No additional dependency, learned decoder, ridge fit, CI or demo was added.
Next: Phase 3B, fit and validate ridge regression on the same eligible examples.


## Ridge decoder — Phase 3B, October 18, 2026

### Dimensions, objective and training-only transforms

Use the same prepared partitions, targets, timestamps/masks and 96-channel/unit
order as Phase 2D and the same evaluation functions as Phase 3A.
prepare_partition is a thin shared composition of the existing bin_partition,
align_velocity_targets and build_history_windows functions. It adds no parser,
timing definition or eligibility rule. Both baseline and ridge commands use it.

Each count history is [L,C]=[10,96], oldest-to-newest. Apply the frozen Phase 2D
per-channel transform at every lag and call the existing flatten_features:
C-order [960], oldest lag first, with channels varying fastest.
Feature d=l*C+c identifies zero-based lag l (oldest=0) and channel c.
Training X is [13070,960], Y [13070,2]; validation X [2775,960], Y [2775,2].
Keep all silent/constant channels and raw counts. Do not normalize Y.

Fit W [960,2] and intercept b [2] to minimize:

||Y − XW − 1bᵀ||²_F + lambda ||W||²_F.

The loss is a sum, not an average over training rows; lambda's numerical scale
depends on this convention and normalization. W contains one contribution weight
per feature and output axis. b supplies a constant offset in mm/s and is not
penalized. For intuition, centering X/Y yields
(XᵀX + lambda I)W = XᵀY; the implementation never explicitly inverts this matrix.

Use scikit-learn 1.7.2 Ridge(alpha=lambda, solver="svd", fit_intercept=True),
not a custom fitting solver. Its dense SVD solver handles correlated histories
and zero columns; intercept centering/reconstruction leaves b unpenalized.
Confirmed the installed library's intercept reconstruction and tested constant
targets with very large lambda: weights shrink to zero, intercept retains the
two constant targets. See [Ridge documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Ridge.html).

Fit normalization on unique valid training bins only, exactly as Phase 2D.
If the existing phase2d/count_scaler.json is available, restore its core frozen
metadata and verify exact identity against a fresh training-only fit. Refuse
any disagreement. If absent in a new checkout, recompute via the same existing
fit_count_scaler rule. Save the complete verified transform with the model.
No validation statistics update channel order, means or scales.

RidgeInputs contains normalized flat features and label-free ExampleIndex,
history length, feature-order convention and full transform fingerprint.
Training fit additionally receives matching training VelocitySeries. Reject
non-training fit, extra training masks, nonfinite data, mismatched shapes/IDs/
context and nonpositive/nonfinite lambda. Predict receives no target values.
A validation-label change cannot alter a particular candidate's weights or
normalization. It can legitimately select a different lambda and thus a
different final candidate; fitting leakage and validation selection differ.

### Predeclared grid, primary metric and test gate

configs/ridge.toml fixes lambda=[0.1,1,10,100,1000,10000] before real candidate
results are inspected. Every candidate fits only the eligible training rows.
Use the unchanged overall validation RMSE as primary metric, minimizing it.
Take the largest lambda whose score is within absolute 1e-9 mm/s of the minimum.
Do not substitute mean R² as the selection metric, extend the grid after seeing
results or use test data for selection.

Score the selected ridge model plus zero/training-mean predictors with the same
evaluate_velocity function and exact validation index/mask. The baseline exporter
is shared, retaining original baseline columns and adding ridge vx/vy columns.
Undefined-axis/nonfinite/error policies from Phase 3A are unchanged.

Only training and validation targets/windows are constructed by the ridge
command. The common loader still reads the recording and global coverage
metadata as before. No test model selection, targets/windows, predictions,
plots or scoring are added. Selection consumes validation labels, so selected
validation performance is development evidence; final test scoring stays closed.

### Persistence, reproducible command and actual validation

~~~powershell
uv sync --locked
uv run --locked python scripts/train_ridge.py phase3b-review
~~~

Use a new directory under ignored runs/. Save ridge.json with W, b, lambda,
solver/intercept convention, feature/output order/units, full channel/unit
mapping and frozen transform, fit count, fitting library/version and transform
hash. FrozenCountScaler.from_metadata restores the existing fit convention
without refitting; it validates finite means, positive scales, constant-channel
scale 1, dimensions and exact policy metadata. RidgeDecoder.load validates
schema, parameter dimensions/values, ordering/units and transform hash; an
optional expected scaler must match exactly. Inference also rejects incompatible
channel/unit order, source/split/configuration/policy identity, L or transform.
Files are created exclusively; existing runs/model artifacts are preserved.

The command verifies exact validation prediction equality after saving/loading.
Predictions use XW+b with the extracted library-fit parameters; synthetic tests
also compare this path with sklearn's own predict method.

run.json records source/full split, base and ridge configuration identity,
full normalization and provenance, channel/unit/feature order, every candidate's
validation metrics, selected lambda, selected training diagnostics, validation
scores/first example, plotting rule, package versions/native thread settings,
Git commit/dirty state, implementation and artifact SHA-256 values and whole-run
duration. Duration includes preparation, six fits, selection and export; it is
not inference latency. metrics.json and timestamped predictions.csv retain
shared scoring evidence. Model parameters and all generated artifacts stay in
ignored runs/, with no processed cache or test-model output.

Use the existing first-ten-second validation plotting rule unchanged:
[1550.200,1560.200) s, 500 samples. The shared plot now supports ridge alongside
the two constant baselines. Reject gaps/invalid intervals rather than searching
for a favorable clip. No smoothing, signal shift or trajectory integration.

The real grid selected lambda=1000, an interior grid value. Validation overall
RMSE is 52.571379922589024 mm/s, per-axis [56.84421168030996,47.9190522900287].
R² is [0.47454563800409655,0.48460628954752893], mean 0.47957596377581274.
Both baseline metrics and every original baseline CSV column/row match Phase 3A.
All three score the same 2775 target timestamps from 1550.200 to 1605.680 s.
See the [ridge validation report](../reports/phase3b_ridge.md) for the full grid,
comparison, provenance and first prediction. The first ridge prediction has
larger errors than zero, illustrating that aggregate improvement is not a
guarantee at every timestamp.

All 183 tests pass, including 30 new ridge checks and 153 retained checks.
Tests cover known synthetic linear recovery, exact oldest/channel order,
unpenalized intercept, save/load equality, immutable/rejected transform/channel
contexts, validation-label independence of fitting, shrinkage, selection/ties
and malformed artifacts. Lint/format/lock checks pass; a second real run produces
identical model, prediction CSV and metric JSON. Exported metric recalculation
matches, and the actual figure was visually inspected. New dependencies are
only scikit-learn and its tool-resolved requirements.

Ridge beats both trivial baselines, so no failure-triggered preparation change
was made. Splits, timing/targets, channel policy and normalization remain fixed.
Overlapping histories are correlated; validation selection can overfit with
repeated experimentation, and this one-session result has no final test,
physiological alignment, cross-session or live-control guarantee.
Next checkpoint: the Kalman decoder.


## Kalman parameter estimation — Phase 4A, October 19, 2026

### State, observation and timing

Use two-dimensional state s_t=[vx,vy] in mm/s. It represents the existing
completed-bin derived velocity, with the actual selected behavioral-sample
denominator from Phase 2C. State times remain the 20 ms global bin-end grid;
behavioral source intervals can differ slightly from the nominal bin width.
No target redefinition, forecast horizon, additional lag or timing shift is added.

The observation z_t has C=96 channel values from the current bin (t−20 ms,t].
Reuse the existing frozen training per-channel count standardization.
current_bin_observations calls FrozenCountScaler.transform on the existing
HistoryWindows and selects its final bin, preserving all channel/unit order
and exact timestamps. It does not read movement labels. Raw counts are preserved.
Its output is [N,96], rather than the ridge [N,960] history representation.
The same Phase 2D history/quality eligibility determines available fitting rows;
prefix warm-up or unavailable targets do not become fabricated zero states.

Use the column-vector models:

s_t = A s_(t−1) + a + w_t, w_t modeled with mean zero and covariance Q.
z_t = H s_t + d + epsilon_t, epsilon_t modeled with mean zero and covariance R.

A is [2,2], a [2], H [96,2], d [96], Q [2,2], R [96,96].
A maps previous velocity into expected current velocity. H maps velocity into
expected neural activity; later filtering will use measured activity to infer
velocity. a is in mm/s; d is in standardized-count units. Q is in (mm/s)^2,
R in standardized-count squared units. H coefficients therefore have units of
standardized count per mm/s. These are channel observations, not single neurons.

For intuition, A=0.9I and a=0 turn previous velocity [10,−4] into [9,−3.6].
Q expresses uncertainty left after that movement prediction; R expresses
uncertainty in observations left after accounting for velocity. They are
covariances, not velocity estimates or learned neural features.

### Exact row and transition eligibility

The command reuses load_verified_inputs and prepare_partition for training
only, with the same source checksum, split/configuration checks, target
construction, validity and channel policies. Fit-count normalization uses the
same unique valid training bins as before. Restore the existing Phase 2D
metadata when available and verify its core fingerprint against a fresh
training-only fit; refuse disagreement. No validation/test transform fitting.

CurrentBinObservations retains label-free ExampleIndex, exact original bin
starts/ends/global IDs, row session identities, origin/width and transform hash.
The matching VelocitySeries is supplied only to training parameter estimation.
Reject shape/nonfinite/context/normalization/timestamp/mask mismatches using
the existing identity checks; both inputs must describe the same rows.
Reject non-training partitions. Do not silently intersect, trim or sort.

Fit transitions only between adjacent original rows when:

- Both rows' shared validity/scoring mask is true.
- Their global bin IDs differ by exactly one.
- Previous bin end equals current bin start on the unchanged exact grid.
- Session and continuous-segment identities match.

Never first filter rows and then treat remaining neighbors as consecutive.
Missing grid bins, invalid rows, purge/split boundaries and session/segment
changes break the chain. Every observation must match origin+k*width and its
saved left edge exactly. The transition represents one 20 ms grid step only;
it is not rescaled to bridge missing time. Nonfinite values fail, even outside
an explicit mask, consistently with the existing prepared/scoring contract.

Real preparation provides 13070 eligible training states/observations and
13069 eligible transitions, with no further adjacency exclusion. First pair
global end IDs [10,11], last [13078,13079]. Training target times are
1288.200–1549.580 s. The original nine prefix warm-up target exclusions remain.

### Regression and residual covariance estimation

Fit A/a using previous/current eligible training velocities. Fit H/d using all
valid aligned training state/observation rows. Center predictor and response
columns, call NumPy linalg.lstsq(rcond=None), then reconstruct the intercept.
This uses a library least-squares solver without explicit matrix inversion.
The fitted row-oriented coefficient is transposed into the stated A/H
column-state convention. At least four observations/transitions and rank-two
centered velocity predictors are required; insufficient or rank-deficient
state information fails clearly rather than discarding a velocity dimension.
See [NumPy least squares](https://numpy.org/doc/2.4/reference/generated/numpy.linalg.lstsq.html).

Compute residuals w=s_current−(A*s_previous+a) and
epsilon=z−(H*s+d). Estimate centered population covariances E_centered.T @
E_centered / M (ddof=0), where M is the corresponding valid fit-row count.
This is a fitted residual estimate with no degrees-of-freedom correction.
Symmetrize for roundoff. Retain raw covariance eigenvalue/rank diagnostics.

Use fixed configs/kalman.toml rules, without validation/test tuning:

- For full Q and initial-state covariance, floor =
  max(1e-6 (mm/s)^2,1e-8*mean(diagonal)). Add max(0,floor−minimum eigenvalue)
  to each diagonal. Report the exact added jitter; do not project A.
- For R, use diagonal raw residual variances and floor each at
  max(1e-8,1e-8*mean(raw variances)), in standardized-count squared units.
  Keep all channels. This deliberately omits cross-channel residual covariance.
- Diagnose the full raw R too, reporting singularity, discarded off-diagonal
  norm and which channels were floored; a zero/silent channel is not missing data.
- Reject nonfinite covariances, asymmetry or materially negative eigenvalues.
  Roundoff tolerance is 32*machine_epsilon*dimension*max(1,max_abs_eigenvalue).
  Report numerical rank with that threshold and undefined/null condition number
  for singular matrices. Final covariances must be strictly positive definite.

The real Q eigenvalues are [135.692915,338.118672] (mm/s)^2, with no jitter.
Raw full R has rank 85/96 and minimum eigenvalue −4.23e-16, within tolerance.
Diagonal R retains all 96 channels; 11 constant channels receive variance 1e-8.
Final R eigenvalues span [1e-8,1.0006791874]; condition number is about 1e8.
This conditioning is disclosed and needs numerical filter checks in Phase 4B.
Q/R do not quantify pure biological or hardware noise: noisy labels, linear
model mismatch and unmodeled activity contribute to the fitted residuals.

### Training-derived prior and saved evidence

Save initial mean m0 as the eligible training velocity mean, and P0 as its
population covariance with the same full-state floor policy.
Here m0=[−0.1479870893879215,0.1386043376635596] mm/s.
P0 eigenvalues [4716.954100,7137.032657] (mm/s)^2 need no jitter.

The initial-state policy for Phase 4B is a prior at the first eligible bin
before its observation, with no extra transition at that first bin.
Reuse m0/P0 at every session/partition/segment start or eligibility
discontinuity. Do not carry state across excluded time or initialize from
held-out movement. Later temporal memory will differ from ridge's bounded
ten-bin history while comparison timestamps/masks remain common.

Fitting uses training movement labels to learn dynamics and the neural
observation relationship. Held-out filtering will accept neural observations
and its own previous estimate; held-out movement labels will be scoring
references only. Training-label-conditioned regressions/residual diagnostics
are not held-out decoding predictions.

~~~powershell
uv run --locked python scripts/fit_kalman.py phase4a-review
~~~

Choose a new ignored run directory. Save kalman_parameters.json with A/a/H/d,
Q/R, m0/P0, state/observation units, full frozen transform/channel/source/split
identity, fitting configuration, fit-pair hashes, diagnostics and initialization/
reset policy. diagnostics.json and run.json record source/full split, exact
configuration/code/artifact hashes, environment, commit/dirty status and
whole fitting-run duration. Existing runs are refused without overwrite.
No observation arrays, residual series, validation/test outputs or filter
state are persisted. Duration is not inference latency.

All 210 tests pass, including 27 new synthetic checks. They cover known
linear-dynamical parameter recovery, current-bin versus history selection,
gaps/sessions/invalid rows, rejection of held-out partitions and mismatched
identities, raw held-out mutation invariance of training fits/normalization,
silent channels, singular/invalid covariances, full-state floors and prior/
artifact policies. Lint/format/lock checks pass. Repeated real fitting gives
identical parameter and diagnostic files. See the [fit report](../reports/phase4a_kalman_fit.md).

No filtering loop, validation/test Kalman scores, parameter search, new
dependency, neural preprocessing, alternate split or target definition was
added. Linear stationary/Gaussian, diagonal-observation-noise and one-step
state assumptions remain approximations. Next: Phase 4B causal predict/update
filtering and validation comparison against ridge and both trivial baselines.


## Causal Kalman filtering — Phase 4B (2026-10-19)

Reuse the exact Phase 4A A [2,2], a [2], H [96,2], d [96], Q [2,2],
R [96,96], training mean/covariance prior and frozen observation scaler.
No refit, covariance tuning, channel deletion, new lag or alternative data path.
The observation z_t is the standardized count vector for the current completed
20 ms bin, extracted from the final source bin of the existing prepared window.
All raw histories remain available and unchanged.

### Prediction, innovation and update

For a contiguous eligible bin after the first update, use column-state equations:

- Predicted velocity m_minus = A @ m_previous + a.
- Predicted covariance P_minus = A @ P_previous @ A.T + Q.
- Innovation r = z_t - (H @ m_minus + d), shape [96].
- Innovation covariance S = H @ P_minus @ H.T + R, shape [96,96].
- Cross covariance B = P_minus @ H.T, shape [2,96].
- Gain K = solve(S, B.T).T, shape [2,96].
- Updated velocity m = m_minus + K @ r, shape [2], in mm/s.
- Joseph covariance P = (I-KH) @ P_minus @ (I-KH).T + K @ R @ K.T.

Use [NumPy's numerical linear solve](https://numpy.org/doc/2.4/reference/generated/numpy.linalg.solve.html)
without constructing an inverse. Validate finite states/observations and symmetric
positive-semidefinite covariances with the established roundoff tolerance.
Symmetrize P_minus, S and P for roundoff; do not clip eigenvalues, add adaptive
jitter, change fitted floors or silently repair a failed solve. The Joseph form
avoids the cancellation in a direct covariance subtraction. Covariance describes
the fitted model's uncertainty, not a demonstrated calibration guarantee.

One-axis illustration, using an identity observation model only for teaching:
previous velocity 8 mm/s, covariance 3; A=1,a=0,Q=1 predicts 8 with variance 4.
Observation 10 with R=1 gives innovation 2, gain 4/(4+1)=0.8, updated velocity
9.6 and variance 0.8. Real observations are 96 standardized channel counts,
not direct velocity measurements. Q adds uncertainty to motion prediction;
R describes observation residual uncertainty and influences neural correction.

### Initialization, timing, warm-up and missingness

Each call filters exactly one ordered partition and starts fresh. At its first
eligible bin, use the training prior m0/P0 directly and perform the measurement
update, with no additional A/Q transition. Reuse this prior after session or
segment changes, missing grid IDs, nonshared edges or explicitly invalid rows.
No state crosses partitions, purge gaps or discontinuities. Reject unordered
rows and grid/channel/unit/source/split/transform mismatches; never sort or trim.

Keep Phase 2D's nine prefix warm-up exclusions for a ten-bin history; begin
Kalman filtering at the first common eligible row, without processing its nine
unscored history bins as extra Kalman updates. Add zero Kalman-specific settling
exclusions. Score the reset bin too. After a discontinuity the prepared windows
already require complete support; the filter additionally resets at the next
eligible row. Kalman can remember earlier eligible estimates within the segment,
whereas ridge uses bounded ten-bin histories; identical score times do not imply
identical temporal memory.

The current observation is used only after its bin closes. Prediction timestamps
are bin ends, indicating earliest availability before unmeasured computation
delay, not a wall-clock latency measurement or future movement forecast.
Held-out velocity/position values never initialize, correct or enter filtering.
Preparation's fixed label-quality metadata still determines shared eligibility;
movement values are accessed for scoring after all predictions are generated.

A genuine zero raw count vector is valid and must be standardized normally;
it is neither a missing packet nor necessarily zero after standardization.
Under this checkpoint's batch contract, an explicit invalid row produces no
emitted estimate and breaks the chain. Its finite training-prior placeholders
are masked and labeled not_emitted_invalid, never counted as decoded values.
Missing rows on the grid also force a reset at the next eligible observation.
Malformed nonfinite arrays fail rather than becoming implicit missingness.
The lower-level kalman_step(..., observation=None) implements and tests a
prediction-only mathematical step, but this comparison does not propagate that
state across missing/invalid time. A future robustness packet-loss experiment
needs its own explicit policy rather than quietly changing this contract.

### Frozen loading, comparison and evidence

KalmanParameters.load checks the existing JSON schema, units, training-only
prior/reset policy, covariance validity and full transform hash/identity.
No pickle, refit or behavioral initialization is used. Save/load predictions
and covariances are bit-identical. The evaluation command checks the Phase 4A
artifact hash against its run metadata, and requires its source/full split,
timing/configuration and training mean to agree with the loaded context.
Load the already selected Phase 3B ridge artifact with the same scaler;
no lambda selection is repeated. The training-mean baseline reuses m0, whose
fit population is the same eligible training targets.

~~~powershell
uv run --locked python scripts/evaluate_kalman.py phase4b-review
~~~

Default model runs are runs/phase4a-kalman-fit and runs/phase3b-ridge;
--kalman-run and --ridge-run may point to equivalent saved local artifacts.
Choose a new output run ID; existing directories are refused. Detailed evidence
is in runs/phase4b-kalman-audited: model copy, timestamped four-model predictions,
filter reset/status CSV, indexed posterior covariance NPZ, metrics and run JSON.
Metadata includes model/transform/source/split/channel/configuration identities,
first two complete step traces, package versions, CPU/thread policy, Git commit
and dirty state, implementation/artifact hashes. Runtime is whole-run duration,
not inference latency. All these generated outputs remain ignored.

Score all 2775 original eligible validation rows, 1550.200–1605.680 s inclusive,
using unchanged Phase 3A per-axis/overall RMSE and R² functions and common mask.
Require every original Phase 3B target, zero/mean/ridge prediction, timestamp,
segment, partition and mask to match the saved CSV exactly. Keep the fixed first
10 s plot [1550.200,1560.200), 500 samples, no smoothing or interval search.
Test targets/windows, predictions, plots and metrics are not constructed.
The common source loader still reads the recording/global metadata as before.

Kalman validation overall RMSE is 102.838507 mm/s, mean R² -1.102563;
ridge remains 52.571380 and 0.479576. Both baselines are about 72.8175 mm/s.
Kalman underperforms all three; no post-result change was made to Q/R,
splits, lag, channels, transform, eligibility or warm-up. Full axis scores and
provenance are in the [Phase 4B report](../reports/phase4b_kalman.md).

Numerical audit independently solves the equivalent two-state posterior precision
equations for every emitted bin. Maximum mean difference is 1.13687e-13 mm/s,
maximum covariance difference 1.81899e-12. All posterior covariances are symmetric
and positive definite; minimum eigenvalue 1324.911074 (mm/s)^2. Maximum relative
S*K.T solve residual is 5.39388e-16. Eleven silent-channel H rows and gains are
zero, so their small R floors do not provide movement evidence.
The fixed plot shows missed reversals and sustained y bias: validation mean
predicted vy=-28.738074 versus label mean 0.010377 mm/s. These observations and
synthetic tests reveal no specific correctness defect. Stationary linear dynamics,
current-bin linear/Gaussian observations, temporally independent residuals and
diagonal R remain possible model limitations, not proven causes of the poor score.
No assumption was tuned away using validation, and no final-test claim is made.
Next increment: a small causal GRU, retaining the common preparation/evaluation.
