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
