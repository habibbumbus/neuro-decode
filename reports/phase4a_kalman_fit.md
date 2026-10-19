# Phase 4A — training-only Kalman parameter diagnostics

This checkpoint estimates a two-dimensional velocity transition model and a
neural observation model. It does not implement filtering, produce decoded
velocities, or measure Kalman validation/test performance.

## Provenance and fitting rows

- Recording: indy_20161005_06, Zenodo record 583331, version v1.
- Source SHA-256: 4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac.
- Canonical split SHA-256: 3e54457065c7bf9d71fb2eef4a5bf7710d7d498fbf84bf1995d59ad11107ee13.
- Base configuration SHA-256:
  2ca6b9a7a83b9fa12bf206aca2521cea60cd8903ed50f77e008b7db87c3c59f1.
- Normalization fingerprint:
  603aa299f0a1d75625acea41a18b9ecf56c58e4e0515ec8322ea1aa34abe68dd.
- Measured run: runs/phase4a-kalman-fit; repeat: runs/phase4a-kalman-repeat.
- Base commit c78d413c84b2ebc86beb563a01da9b396bf35796; new implementation
  was uncommitted/dirty. run.json records actual code hashes and dirty status.
- Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1, Windows CPU, native threads limited
  to one. No dependency was added and no inference latency was measured.
- Existing preparation yielded 13070 eligible training observation/state rows,
  target ends 1288.200–1549.580 s. Existing nine warm-up target exclusions remain.
- 13069 consecutive transition pairs, first global IDs [10,11], last
  [13078,13079]. No adjacency was excluded within these already eligible rows.
- Neural observations are only the current bin's 96 standardized channel counts,
  using the existing Phase 2D transform fitted on 13079 unique training bins.
  The ten-bin history contract continues to determine eligible rows; earlier
  history bins are not concatenated into the Kalman observation.

## Models and dimensions

State s_t=[vx,vy] is completed-bin derived velocity in mm/s. Observation z_t has
96 standardized channel counts. Column-vector conventions are:

s_t = A s_(t−1) + a + w_t, with covariance Q for w_t.
z_t = H s_t + d + epsilon_t, with covariance R for epsilon_t.

| Parameter | Shape | Role |
| --- | --- | --- |
| A | [2,2] | Previous velocity to expected current velocity |
| a | [2] | Transition intercept, mm/s |
| H | [96,2] | Current velocity to expected neural observation |
| d | [96] | Neural observation intercept |
| Q | [2,2] | Transition residual covariance, (mm/s)^2 |
| R | [96,96] | Neural residual covariance, standardized count squared |
| Initial mean | [2] | Training velocity mean |
| Initial covariance | [2,2] | Training velocity population covariance |

The fitted transition is approximately:

~~~text
A = [[ 0.97372105, -0.02813544],
     [ 0.02892290,  0.98708227]]
a = [0.00896779, 0.00130489] mm/s

Q = [[294.48813862, -83.23653226],
     [-83.23653226, 179.32344868]] (mm/s)^2
~~~

A's spectral radius is 0.9807938374. It was reported without clipping/projecting
the fitted dynamics. Both centered regression predictor matrices have rank two.
All parameter arrays are finite.

## Covariance diagnostics and regularization

Centered population residual covariance uses denominator M, with no
degrees-of-freedom correction. Full raw covariance is recorded for diagnostics.
For Q and the initial covariance, set floor=max(1e-6,1e-8*mean(diagonal)),
in (mm/s)^2, and add max(0,floor−minimum eigenvalue) to each diagonal.
Both required zero added jitter in this recording.

For R, retain the raw residual variances on a diagonal, explicitly ignoring
cross-channel residual covariances. Floor each variance at
max(1e-8,1e-8*mean(raw variances)); here this is 1e-8.

| Matrix | Raw numerical rank | Final eigenvalue range | Applied adjustment |
| --- | ---: | --- | --- |
| Q | 2/2 | [135.692915,338.118672] | No jitter |
| R | Raw full covariance 85/96 | [1e-8,1.0006791874] | Diagonal assumption; 11 variances floored |
| Initial covariance | 2/2 | [4716.954100,7137.032657] | No jitter |

All final covariances are finite, symmetric (max absolute asymmetry zero) and
positive definite. Raw full R has minimum eigenvalue −4.23e-16, within the
reported roundoff tolerance 4.98e-12, and is singular. Its condition number is
reported as undefined/null rather than a finite substitute.
Final diagonal R has condition number about 1.00068e8; it is positive definite
but poorly scaled. Numerical predict/update checks remain necessary in Phase 4B.

Floored channels, all retained: M1 009, 047, 055, 058, 061, 068, 071, 078,
079, 082, 096. Their fitted H rows and d entries are exactly zero.
Ignoring off-diagonal residual covariance is a modeling assumption, not just
a numerical repair; its discarded Frobenius norm is 6.8843393535.

## Saved initial-state policy

Use the eligible training velocity mean:

[-0.1479870893879215,0.1386043376635596] mm/s.

The corresponding population covariance, requiring no jitter, is:

~~~text
[[6710.62694022, -922.01599566],
 [-922.01599566, 5143.35981698]] (mm/s)^2
~~~

The saved future-filter policy treats this as the prior at the first eligible
bin before observing that bin, without an extra transition. Reuse it at session,
partition, continuous-segment and eligibility discontinuities. Do not initialize
from validation/test velocity or propagate state across excluded time.
No such filtering/reset execution is implemented in Phase 4A.

## Reproduction, evidence and limits

~~~powershell
uv run --locked python scripts/fit_kalman.py phase4a-review
~~~

Use a new directory name. The command saves kalman_parameters.json,
diagnostics.json and run.json under ignored runs/. Existing outputs are refused
without overwrite. These contain matrices, transform/channel/source/split
identity, effective configuration, fit-row/pair hashes, training prior policy,
covariance adjustments and code/environment/Git provenance. No raw recordings,
observation arrays, residual time series or predictions are saved.

All 210 tests passed, including 27 new tests for synthetic linear dynamics,
correct current-bin extraction, boundary/mask handling, held-out-value isolation,
finite covariances, singular/constant cases, invalid structures and persistence.
Repeated fitting produced identical parameter/diagnostic JSON. The existing-run
refusal was exercised and preserved the original run.

Training movement labels are used to estimate A/H and their residuals.
These label-conditioned fit diagnostics are not decoded-velocity scores.
Q includes unmodeled dynamics and velocity-label noise; R includes neural/model
residuals conditional on noisy labels, not pure electrode noise.
Stationary linear/Gaussian and diagonal-neural-noise assumptions are approximations
for spike counts. No validation or test targets/windows, predictions, plots or
scores were constructed by this command; the common loader still reads the
recording and global metadata as in earlier checkpoints.

Next: Phase 4B, implement and verify causal Kalman predict/update filtering,
then compare validation results on the established common examples.
