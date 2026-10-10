# Causal timing and chronological evaluation — Phase 2A

## Recording and scope

The plan uses Indy session indy_20161005_06, Zenodo record 583331 v1, with the
SHA-256 in [the manifest](../data/manifest.json). Behavior spans 1288–1662 s
(374 s), with 4 ms sampling and no detected timestamp gaps. Global neural event
extrema are 1283.0023444054618–1666.9959174171197 s. Shared coverage is therefore
[1288, 1662). Event extrema do not prove uninterrupted acquisition; silence is
not an outage indicator. Physical synchronization and the finger/cursor
discrepancy remain unresolved in [the dataset card](dataset.md).

This checkpoint creates no count bins, velocity arrays, transforms, eligible
feature indices, or model predictions.

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

Future velocity construction will compute (p(b1) − p(b0)) / 0.020 in mm/s,
separately for x/y. This defines completed-bin displacement rate using sampled
positions associated with the boundaries. When samples precede boundaries, it
approximates continuous movement over the nominal interval; retain actual source
times and ages. Use the declared bin width as denominator, not an undocumented
varying sample separation. Do not interpolate using later samples. This is
completed-interval estimation, not future forecasting. Positions provide targets
and scoring references, never neural decoder inputs.

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
indices 45 and 50, at exactly 1288.180 and 1288.200 s. A future target would use
those two x/y positions divided by 0.020 s. Its conservative source envelope
begins at approximately 1288.174 s (1 ns earlier with the tolerance). Nothing after 1288.200 s is allowed. This checked timestamp
selection only; no velocity value was calculated.

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
Eligible indices remain null. Next: Phase 2B causal spike-count bins.
