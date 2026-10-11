# Phase 3A validation baselines

Measured on one recorded Indy session, not final test performance.
Both trivial predictors ignore neural activity. The purpose is a reproducible
reference for later neural decoders on the same eligible timestamps.

## Provenance and protocol

- Source: Zenodo record 583331, version v1, session indy_20161005_06.
- Recording SHA-256: 4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac.
- Canonical split SHA-256: 3e54457065c7bf9d71fb2eef4a5bf7710d7d498fbf84bf1995d59ad11107ee13.
- Configuration UTF-8/LF SHA-256: 2ca6b9a7a83b9fa12bf206aca2521cea60cd8903ed50f77e008b7db87c3c59f1.
- Run: runs/phase3a-baselines; repeat: runs/phase3a-repeat (ignored local outputs).
- Base Git commit: 249ce682ed4543b6db9c3df3187c265bbe532f2f; implementation was
  uncommitted/dirty. These scores are not a result from that commit alone.
  run.json records hashes of the actual implementation files and dirty state.
- Environment: Python 3.11.17, NumPy 2.4.6, h5py 3.16.0, Matplotlib 3.11.2,
  Windows, CPU. CPU identifier was unavailable; no inference latency was measured.
- Existing Phase 2D preparation: ten 20 ms bins, 96 ordered M1 channels, original
  raw counts and actual-elapsed-time velocity labels, unchanged split and masks.
- Mean fit: 13070 eligible training velocity targets, unweighted once per target.
  Frozen prediction [-0.1479870893879215, 0.1386043376635596] mm/s.
- Validation: all 2775 eligible examples; target-end timestamps 1550.200 through
  1605.680 s inclusive, span 55.480 s. Nine warm-up targets were already excluded
  by preparation; no score-based exclusions or additional mask changes.
- No test windows, predictions, plots or metrics were constructed by this run.
  The original loader reads the acquired recording, as in earlier checkpoints;
  this is not a claim that the test interval's bytes were never loaded.

## Validation comparison

| Predictor | RMSE x (mm/s) | RMSE y (mm/s) | Overall RMSE (mm/s) | R² x | R² y | Mean R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero velocity | 78.418547 | 66.748063 | 72.817486 | −2.097390e−7 | −2.416840e−8 | −1.169537e−7 |
| Training mean | 78.418755 | 66.748186 | 72.817654 | −5.499573e−6 | −3.690502e−6 | −4.595038e−6 |

Overall RMSE is sqrt(mean squared error across both axes and all scored examples).
R² uses each scoring axis's own mean only as its reference denominator; that mean
is not provided to either predictor. Unweighted mean R² averages the two axis
scores. Both axes are nonconstant here. Full precision is in metrics.json.

Both predictors have R² slightly below zero. The training mean is marginally
worse on this validation interval than zero. Neither tracks movement variation.
These references do not establish neural decoding ability, human performance,
live control, final test accuracy, statistical significance or latency.

## One timestamped example

At validation bin end 1550.200 s (global bin ID 13110), the actual derived target
is [14.162167438059216, 3.7882881059650577] mm/s.

| Predictor | Predicted [vx,vy] (mm/s) | Error: prediction − target (mm/s) |
| --- | --- | --- |
| Zero velocity | [0,0] | [−14.162167438059216,−3.7882881059650577] |
| Training mean | [−0.1479870893879215,0.1386043376635596] | [−14.310154527447137,−3.649683768301498] |

A single signed error is not RMSE; RMSE squares and aggregates all scored errors.

## Plot and reproduction

The saved validation_velocity.png uses the first eligible timestamp plus exactly
ten seconds: [1550.200,1560.200) s, 500 examples, with a shared time offset.
The interval rule was fixed before viewing movement or prediction quality.
Two panels show x/y derived velocity in mm/s and both constant predictions.
Constant values form flat lines; the two lines nearly overlap near zero.
No smoothing, interpolation, signal shifts or trajectory integration are used.
The generated figure was visually reviewed for labels, units and shared timing.

Run from the repository root with a new run directory name:

~~~powershell
uv run --locked python scripts/evaluate_baselines.py phase3a-review
~~~

The directory receives run.json, metrics.json, predictions.csv and
validation_velocity.png. Existing directories are refused without overwriting.
Run metadata include source/split/configuration/policy/code identities, effective
timing/channel order, environment, frozen training mean, eligibility, plotting
rule and whole-run duration. Duration includes preparation/export and is not
an inference-latency measurement. All generated artifacts stay outside Git.

The measured CSV was independently audited for 2775 ordered finite validation
rows, fixed predictions, scoring masks and target/source IDs. Recalculation
from exported predictions reproduced the recorded scores. Repeated runs produced
identical CSV and metrics. No parameters were tuned from these results.

See [methods](../docs/methods.md) for exact metric and undefined-value policies.
Next increment: Phase 3B, ridge regression on these same examples.
