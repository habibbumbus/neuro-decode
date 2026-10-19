# Phase 4B: causal Kalman validation comparison

Kalman performs worse than selected ridge and both trivial baselines on the
shared development validation set. Test scoring remains closed. No parameter
or split was changed after this result.

## Provenance and fixed protocol

- Session indy_20161005_06, Zenodo 583331 v1; source SHA-256:
  4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac.
- Canonical split SHA-256:
  3e54457065c7bf9d71fb2eef4a5bf7710d7d498fbf84bf1995d59ad11107ee13.
- Target policy SHA-256:
  2e86a91a0f1735f1ff3f5d5e2d0ce2781cca25ecf979e2d99e720652df90189b.
- Observation-normalization SHA-256:
  603aa299f0a1d75625acea41a18b9ecf56c58e4e0515ec8322ea1aa34abe68dd.
- Fitted Kalman artifact runs/phase4a-kalman-fit/kalman_parameters.json,
  SHA-256 5b907ca94f805a7d244c903b9fd26470271a3fb723100dcd59dbb05297884507.
  Loaded unchanged A/a/H/d/Q/R and training prior, with all 96 M1 channels.
- Selected ridge runs/phase3b-ridge/ridge.json, lambda=1000,
  SHA-256 43455b85f99928b785296890d5fa316820d8e9f071d77883f6743efe1bcf36ee.
  No selection/refit in this phase. Baseline training mean equals saved m0.
- Measured run runs/phase4b-kalman-audited. Initial verification run
  runs/phase4b-kalman-validation has identical predictions, metrics, plot,
  parameter copy, filter status and covariance artifact bytes. The audited run
  additionally saves independent-equation and aggregate bias diagnostics.
- Base commit 4e64fe7647e66db96e0b7a1ce3b7607a51834910; implementation was
  dirty/uncommitted. run.json contains code hashes, package versions, Git status,
  effective configuration and artifact hashes. Results are not from that commit alone.
- Python 3.11.17, NumPy 2.4.6; exact remaining package versions are in run.json.
  Windows CPU, native thread pools limited to one. Whole-run duration is not
  inference latency; no latency benchmark is claimed.
- Shared preparation and scoring are unchanged. All 2775 validation targets,
  zero/mean/ridge predictions, timestamps, segments, partition and masks match
  the recorded Phase 3B CSV exactly. Eligible identity SHA-256:
  30446619fcf5a2d49ada8bff99ac8616247a255e13a68a7c4b1836171c9875c1.
- All scored target ends are 1550.200–1605.680 s inclusive (55.480 s span).
  Nine common history warm-up bins are excluded before this population.
  One training-prior reset at the first eligible row; no additional Kalman
  settling exclusion. The reset row is scored too.

## Matched validation scores

RMSE units are mm/s. Overall RMSE is the square root of mean squared error
across both axes and all scored rows. Mean R² is the unweighted axis mean.
Small negative baseline R² values round to zero here; exact values remain in JSON.

| Predictor | N | RMSE x | RMSE y | Overall RMSE | R² x | R² y | Mean R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero velocity | 2775 | 78.418547 | 66.748063 | 72.817486 | -0.000000 | -0.000000 | -0.000000 |
| Training mean | 2775 | 78.418755 | 66.748186 | 72.817654 | -0.000005 | -0.000004 | -0.000005 |
| Ridge, λ=1000 | 2775 | 56.844212 | 47.919052 | 52.571380 | 0.474546 | 0.484606 | 0.479576 |
| Kalman | 2775 | 93.653913 | 111.267523 | 102.838507 | -0.426311 | -1.778814 | -1.102563 |

Kalman is not entitled to beat ridge. This poor validation score is retained.
The comparison uses validation for development; it is not final test performance
or an independent replication across sessions. Correlated time bins do not
provide independent trial-level uncertainty estimates.

## One real ordinary predict/update step

Use the second chronological eligible bin, ID 13111, interval
(1550.200,1550.220] s. The first bin used the training prior directly, yielding
posterior [-0.8077172671,-6.1653490615] mm/s. For this second bin:

- A @ previous + a predicts [-0.6040586967,-6.1077634040] mm/s.
- z has 96 frozen-standardized current-bin channel counts. H @ prediction + d
  predicts those observations; the innovation is measured minus expected.
- First three innovations for M1 001/002/003 are
  [-0.3456668617,-0.5396176663,-0.3541071292]. Remaining entries are saved too.
- K is [2,96]. Its first three columns are approximately
  [[0.87000023,-3.85554874,5.00225756],
   [2.87773137,0.70531899,2.78288271]].
- The product of the FULL gain and 96-entry innovation is
  [-28.0620903006,-33.9006676669] mm/s.
- Updated velocity is [-28.6661489974,-40.0084310709] mm/s.
- Joseph covariance is approximately
  [[5355.22033995,-1040.20726145],[-1040.20726145,3636.47032498]] (mm/s)^2.
- Scoring-only actual velocity is [5.2603945883,-6.1748941911] mm/s.
  This is a poor prediction; that label did not participate in any step above.

The run saves full predicted covariances, z/expected z, all innovations and gain
entries for the first two predetermined rows. No favorable-example selection.

## Numerical and model diagnosis

All updates produce finite velocity and symmetric positive-definite posterior
covariance. Minimum posterior eigenvalue is 1324.911074 (mm/s)^2, maximum stored
asymmetry zero, maximum relative innovation-system solve residual 5.39388e-16.
An independently evaluated two-state posterior-precision formula agrees across
all 2775 rows: maximum mean difference 1.13687e-13 mm/s, covariance difference
1.81899e-12. This checks algebra/numerics, not scientific adequacy.

The parameter export round trip gives exactly identical predictions/covariances;
its bytes match the Phase 4A input artifact. Channel order and normalization
are checked before inference. Eleven silent channels have zero H rows and zero
gain entries in the saved traces; their small R floors do not inject evidence.

The fixed plot is runs/phase4b-kalman-audited/validation_velocity.png:
first eligible interval [1550.200,1560.200), 500 samples, same interval as earlier
models, no smoothing or display shift. It was visually inspected. Both axes have
mm/s labels, matching time offsets, distinct predictor lines and a validation
label. Zero/training-mean lines nearly overlap. Kalman misses reversals and has
sustained negative y bias; the full-validation mean vy is -28.738074 mm/s versus
0.010377 for recorded-derived labels. Ridge mean vy is -1.957722 mm/s.

Focused synthetic tests and the numerical/identity audits reveal no specific
correctness defect. Plausible limitations include stationary linear velocity
dynamics, a current-bin linear/Gaussian neural relationship, temporal residual
independence, and diagonal R ignoring correlations already measured in Phase 4A.
They are hypotheses rather than established explanations. No Q/R/floor tuning,
channel removal, signal shifting, extra warm-up or favorable interval was used.

## Reproduction and scope

~~~powershell
uv run --locked python scripts/evaluate_kalman.py phase4b-review
uv run --locked pytest tests/test_kalman_filter.py tests/test_kalman.py
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
~~~

A new run ID preserves previous outputs. The command uses the documented frozen
model paths by default; optional --kalman-run/--ridge-run change local artifact
locations, not the source/split/transform contract. Recreate missing training
artifacts with the existing fit/train commands, without opening test scoring.

Focused tests cover hand-computed updates, a synthetic dynamical system,
nonsquare H, uncertainty reduction, finite symmetric covariance, prediction-only
missingness arithmetic, gap/session/segment/partition resets, future-observation
causality, label-free prepared-window inference, strict identities and persistence.
See [progress](../docs/progress.md) for actual command outcomes.
No test partition targets/windows, predictions, plots or scores were generated;
the common loader still reads source/global metadata as previously documented.
No GRU, replay, trajectory integration, robustness sweep or latency benchmark.
Next increment: the causal GRU; underperformance alone is not a demonstrated
implementation error requiring a corrective checkpoint.


## Read-only follow-up before Phase 5A (2026-10-19)

Read only existing Phase 4A training fit/prior and Phase 4B validation artifacts;
verified saved artifact hashes before reading their CSV/NPZ/traces. No recording
reprocessing, test access, refit or parameter tuning. Bias is mean(prediction
minus target); spread is population standard deviation (ddof=0), in mm/s over
all 2775 common validation examples.

| Validation series | Bias x | Bias y | Spread x | Spread y |
| --- | ---: | ---: | ---: | ---: |
| zero | -0.035914 | -0.010377 | 0.000000 | 0.000000 |
| mean | -0.183901 | 0.128228 | 0.000000 | 0.000000 |
| ridge | 4.948505 | -1.968098 | 54.473187 | 51.032669 |
| kalman | -1.991768 | -28.748450 | 88.849973 | 106.632138 |

Validation label mean is [0.035914,0.010377], spread [78.418539,66.748062].
Training label mean from the saved prior is [-0.147987,0.138604], spread
[81.918416,71.717221]; its covariance needed no jitter. These are training label
statistics. Training Kalman prediction bias/spread are not saved and remain
unavailable; no training scores were manufactured. Constant predictor spread
has only floating-point residue.

Q condition is 2.491793, prior covariance 1.513060, posterior covariance
conditions 1.703811–2.466351. R condition is 1.00068e8; first two innovation
system conditions are 1.23647e8/1.18728e8. For diagnosis only, the 85 nonzero-H
rows have R condition 1.031401 and innovation-block conditions 1.273965/1.223279.
The eleven zero-H rows have zero gain and floored R. Actual filtering retains
all 96 channels. Large full-system conditioning is worth disclosing but is not
by itself evidence for the cause of the poor score.

First/second gain Frobenius norms are 37.872807/33.031266; maximum absolute
entries 9.089302/8.285794. Row norms [26.294132,27.257441] and
[23.583738,23.127296] convert standardized neural residuals to mm/s; gains above
one are not inherently invalid. Innovation products give corrections
[-0.659730,-6.303953] and [-28.062090,-33.900668] mm/s. Only two full gain traces
are saved, so no all-gain distribution is claimed. The existing all-row solve
residual and independent posterior audit remain the stronger numerical checks.

Large y bias and excessive spread are observed. Linear stationary dynamics,
diagonal/cross-time noise assumptions and current-bin observation mismatch remain
hypotheses, not proven causes. No concrete correctness defect was found;
proceed with the synthetic-only GRU implementation without changing Kalman.
