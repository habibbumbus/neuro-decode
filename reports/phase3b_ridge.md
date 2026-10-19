# Phase 3B ridge validation

Ridge improves on both trivial predictors on the development validation set.
These results do not describe final test performance. No test targets/windows,
predictions, scores or plots were produced in this increment.

## Provenance and fixed protocol

- Session: indy_20161005_06, Zenodo record 583331, version v1.
- Source SHA-256: 4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac.
- Canonical split SHA-256: 3e54457065c7bf9d71fb2eef4a5bf7710d7d498fbf84bf1995d59ad11107ee13.
- Base timing configuration UTF-8/LF SHA-256:
  2ca6b9a7a83b9fa12bf206aca2521cea60cd8903ed50f77e008b7db87c3c59f1.
- Measured run: runs/phase3b-ridge; repeat: runs/phase3b-ridge-repeat.
  Both directories are ignored. The original baseline artifacts were preserved.
- Base commit: cd71d47; new code/configuration were dirty and uncommitted.
  run.json records actual implementation hashes, dependency versions and dirty
  status; these are not results from the base commit alone.
- Python 3.11.17, NumPy 2.4.6, scikit-learn 1.7.2, SciPy 1.17.1,
  h5py 3.16.0, Matplotlib 3.11.2, threadpoolctl 3.7.0, Windows CPU.
  Native thread pools were limited to one. No inference latency was measured.
- Training windows [13070,10,96], flattened to [13070,960]; targets [13070,2].
  Validation windows [2775,10,96], flattened to [2775,960]; targets [2775,2].
- Features are oldest-to-newest, with channels varying fastest in C-order.
  All 96 M1 channel identities and unit mappings remain unchanged.
- Restored Phase 2D channel normalization and verified it exactly against the
  existing unique-bin training-only fit (13079 source bins, including warm-up).
  The transform SHA-256 is
  603aa299f0a1d75625acea41a18b9ecf56c58e4e0515ec8322ea1aa34abe68dd.
- Eligible validation identity SHA-256:
  30446619fcf5a2d49ada8bff99ac8616247a255e13a68a7c4b1836171c9875c1.
  Every original baseline CSV column/row matches the Phase 3A export exactly.
- All 2775 eligible validation rows are scored by the unchanged Phase 3A
  metrics/mask policy. End timestamps 1550.200–1605.680 s inclusive;
  timestamp span 55.480 s. No model-specific row exclusion was introduced.

## Training and lambda selection

Use sklearn Ridge(alpha=lambda, solver="svd", fit_intercept=True). Weights are
penalized; the intercept is not. Objective is summed squared error plus
lambda times squared weight norm, not mean squared error plus the same penalty.
See the [versioned Ridge documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Ridge.html)
and [project methods](../docs/methods.md).

The grid [0.1,1,10,100,1000,10000] was declared in configs/ridge.toml before
viewing candidate results. Fit every candidate only on training. Minimize
validation overall RMSE; candidates within an absolute 1e-9 mm/s of the minimum
tie, and the largest lambda wins. No grid extension or test-based tuning occurred.

| Lambda | Validation overall RMSE (mm/s) |
| ---: | ---: |
| 0.1 | 52.886882 |
| 1 | 52.886503 |
| 10 | 52.882726 |
| 100 | 52.846123 |
| 1000 | 52.571380 |
| 10000 | 52.807856 |

Selected lambda: 1000. All candidate full axis scores are retained in run.json.
The selected training overall RMSE is 50.301849 mm/s, mean R² 0.578469;
these training values are not validation estimates.

## Validation comparison

| Predictor | RMSE x (mm/s) | RMSE y (mm/s) | Overall RMSE (mm/s) | R² x | R² y | Mean R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero velocity | 78.418547 | 66.748063 | 72.817486 | −2.097390e−7 | −2.416840e−8 | −1.169537e−7 |
| Training mean | 78.418755 | 66.748186 | 72.817654 | −5.499573e−6 | −3.690502e−6 | −4.595038e−6 |
| Ridge, lambda=1000 | 56.844212 | 47.919052 | 52.571380 | 0.474546 | 0.484606 | 0.479576 |

Both baseline scores reproduce their existing values exactly. Ridge reduces
overall validation RMSE versus both references. The six-way selection uses
validation labels, so the selected validation score is a development estimate,
not an untouched estimate or a final test result. No uncertainty/significance
claim follows from these correlated, overlapping examples in one session.

## First prediction and fixed plot

Use the first eligible validation example, end timestamp 1550.200 s, ID 13110.
Its [10,96] count history is normalized with training statistics and flattened
to [960]. Weights [960,2] and intercept
[-0.10485421436834126,0.12171317024650069] mm/s produce:

- Predicted [vx,vy]: [31.39011385437117,12.002288083723482] mm/s.
- Actual derived velocity: [14.162167438059216,3.7882881059650577] mm/s.
- Prediction minus actual: [17.22794641631195,8.213999977758423] mm/s.

The oldest M1 001 feature is raw count 0, normalized to -0.35355992056352736.
Its two weights [1.5557951627671744,1.66753712506479] contribute
[-0.5500668141610823,-0.58957429347464] mm/s. Sum all 960 contributions and add
the intercept to obtain the output pair. RMSE aggregates squared errors over
all examples/axes; this single pair is not an aggregate score.

This particular example has larger errors than zero prediction, despite better
aggregate ridge performance. It was retained by chronological identity rather
than replaced with a favorable example.

validation_velocity.png uses the identical baseline interval rule:
[first eligible validation timestamp, +10 s), currently [1550.200,1560.200) s,
500 rows. Actual derived velocity is blue, ridge red; zero and training mean
are near-zero flat lines. Visual review confirmed timing, labels and units.
Ridge follows some variations but misses some peaks and produces errors during
other intervals. No smoothing, signal shifts, integration or interval search
was used.

## Reproduction and evidence

From the repository root, after installing the updated lockfile:

~~~powershell
uv sync --locked
uv run --locked python scripts/train_ridge.py phase3b-review
~~~

Use a new run directory name. Artifacts are ridge.json (explicit parameters plus
full frozen transform), run.json, metrics.json, predictions.csv and
validation_velocity.png. Existing directories/files are refused without overwrite.
JSON model loading validates schema/order, shapes, units, transform hashes and
optional expected transform/channel identity. No pickle model is needed.

All 183 tests passed, including 30 new synthetic checks. The known synthetic
linear mapping is separate from real-data evidence. Exact model save/load
validation prediction equality passed. A second real run produced identical
model JSON, prediction CSV and metrics JSON. Independent recalculation from
exported CSV reproduced the scores. Re-running the baseline entry point after
shared-helper changes preserved its original metrics.

Ridge beats both baselines and the checks reveal no current preparation/model
defect requiring a diagnostic increment. Next checkpoint: the Kalman decoder.
Physiological synchronization, out-of-session robustness, final test quality,
runtime latency and replay remain unresolved or unmeasured.
