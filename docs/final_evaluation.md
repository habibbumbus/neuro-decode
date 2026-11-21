# Frozen final evaluation — Phase 7A

This specification is written before test examples, predictions, figures or
metrics are opened. Phase 7A implements and rehearses the command on validation
and synthetic data only. The machine-readable companion is
configs/final_evaluation.json. A separate, explicit test-mode invocation is a
later checkpoint; committing this specification does not itself open test.

## Held-out recording and partition

Session indy_20161005_06, Zenodo583331 v1, source SHA-256
4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac.
Reuse data/splits/indy_20161005_06.json unchanged; canonical split SHA-256
3e54457065c7bf9d71fb2eef4a5bf7710d7d498fbf84bf1995d59ad11107ee13.
Nominal test [1605.9,1662.0) s; authoritative retained interval
**[1606.1000000000001,1662.0) s**, after the established purge.
Train retained [1288.0,1549.6); validation [1550.0,1605.7). Never refit on
train+validation, revise boundaries or borrow support across partitions.

Test count/quality eligibility has not been measured in Phase 7A. The scoring
population is every original eligible Phase 2D test window, without model-specific
subsampling. A timestamp-only grid calculation gives at most2784 complete-history
windows, not a measured test sample count. Report the actual count and every
data-quality/warm-up exclusion only when authorized test evaluation occurs.

## Frozen preprocessing and targets

- Right-closed spike bins (start,end], width20 ms, origin1288 s, integer grid IDs;
  zero additional lag and forecast horizon. Estimate completed-bin velocity.
- All96 physical M1 channels, lexical order M1 001 through M1 096, retaining all
  previously aggregated unit slots including unsorted slot0. Exact channel/unit
  mapping is frozen in the JSON and model/scaler artifacts, never reselected.
- Ten consecutive bins (200 ms), oldest first, channels fastest for ridge's
  C-order960 features. GRU input [1,10,96], CPU float32, H32, one unidirectional
  layer plus two-output linear head;12,546 parameters. Ridge/Kalman float64.
- Reuse the frozen training-only count transform (raw−mean)/scale, unique valid
  training bins including warm-up; population std, exact zero std becomes1.
  Fit count13079, normalization SHA-256
  603aa299f0a1d75625acea41a18b9ecf56c58e4e0515ec8322ea1aa34abe68dd.
  No target scaling, refit, channel removal, smoothing or time shift.
- Target [vx,vy] mm/s is the original tracker-position difference divided by
  actual selected behavioral elapsed time, not a forced20 ms denominator.
  Select latest sample <= each stored bin edge; age<=6 ms+1 ns, positive elapsed,
  elapsed deviation<=6.000002 ms, both samples/support in the same segment/partition.
  Preserve existing target-policy hash
  2e86a91a0f1735f1ff3f5d5e2d0ce2781cca25ecf979e2d99e720652df90189b.
- Keep existing quality/gap rules: behavior gaps>6 ms+1 ns break segments; missing,
  stale, nonfinite and discontinuous support is excluded identically before
  every model. Warm-up counts remain inputs but are not extra scored rows.
  No score-informed exclusion, imputation or perturbation is added.

## Models and exact artifacts

| Model | Frozen source | SHA-256 |
| --- | --- | --- |
| Zero velocity | src/neurodecode/models/baselines.py, bound by implementation hash | No learned parameters; exact [0,0] mm/s |
| Training mean | runs/phase3a-baselines/run.json, training_mean_mm_s | 6ed540be6c1e22520bb82d678f82fae77a12cd5a7998586d15b04035e0e58f1d |
| Ridge lambda1000 | runs/phase3b-ridge/ridge.json | 43455b85f99928b785296890d5fa316820d8e9f071d77883f6743efe1bcf36ee |
| Kalman | runs/phase4a-kalman-fit/kalman_parameters.json | 5b907ca94f805a7d244c903b9fd26470271a3fb723100dcd59dbb05297884507 |
| GRU representative seed2026, lr0.003, epoch50 | runs/phase5b-gru-round1/lr-0.003-seed-2026/gru.pt | 9e5098ceb41fa1cf77c080786be06497396b6022d24e74cd269006179a97d512 |

Training mean is [-0.1479870893879215,0.1386043376635596] mm/s, fitted on13070
eligible training targets. Require exact agreement with ridge metadata and the
Kalman training prior. Do not estimate it from validation/test. Kalman A/a/H/d/Q/R,
variance floors, prior and reset policy remain unchanged. Trusted local loading
only, expected frozen scaler required; no optimizer or training calls.

GRU seeds2027/2028 remain **validation-only development results** in Phase 5B.
Their hashes are recorded for reporting provenance, but they will not be loaded,
tested, ensembled, averaged into a test score or selected after test performance.
The only final GRU is the already chosen representative2026. No parameter
selection, calibration, new robustness condition or latency benchmark on test.

## Inference, common rows and independent checks

Fresh state at partition/session/segment starts. Ridge and GRU rebuild the same
ten-bin history/scaler; GRU starts each bounded window at zero. Kalman skips
the shared history-only warm-up, updates the training prior at the first common
eligible row without an extra transition, then predicts/updates consecutively.
Gaps/invalid sources clear history and deactivate state. Resume only after full
valid history and reset Kalman to training statistics. No held-out movement
initializes/corrects any model; genuine zero counts are observations.

Primary output uses sequential predictions. Generate an independent offline
pass with the same frozen interfaces (GRU batch_size=1), compare every exact
row/source ID, timestamp, partition and segment with replay, and compare all
velocities. Ridge/Kalman atol1e-10 mm/s, rtol1e-12; GRU atol1e-4, rtol1e-6,
abs(diff)<=atol+rtol*abs(offline). Zero/mean must agree exactly. In Phase 7A,
rehearsal also compares validation with established Phase 5B/6A CSV predictions.
There are no pre-existing test predictions to compare before future access.

All five models score exactly the same prepared eligible rows, with no extra
Kalman settling period or baseline-only rows. If a model fails to emit a common
eligible row, stop: do not silently mask it for that model or fill it with zeros.
Known source/target-quality rejection and ten-bin recovery are fixed preparation
rules, recorded before inference. Report exclusion reason counts and emitted/
eligible coverage; no synthetic packet loss is applied in this clean evaluation.

Primary metric: overall RMSE=sqrt(mean squared error across all scored rows
and both axes), mm/s. Secondary: axis RMSE, axis R² and their unweighted mean.
R² uses the scoring labels' own mean only in its denominator, not as a predictor
input. Constant axes/insufficient rows yield explicit null R² with status, not
zero or a fabricated value. No significance/confidence interval from overlapping
bins or seed reuse. Independently recompute all metric values from exported CSV,
with tolerance1e-10, and verify exact row identities/provenance. Check parameter
digests before/after prediction. No train/validation/test support crossing,
feature normalization fitting, optimizer updates or held-out label inputs.

## Exact presentation outputs

1. metrics.csv and metrics.md: fixed model order zero,mean,ridge,kalman,gru;
   counts, time bounds, both axis/overall RMSE, both axis/mean R² and null statuses.
2. predictions.csv: timestamps, source-window IDs, target values, all five
   sequential and offline outputs on identical rows; no adaptive selection.
3. exclusions.csv and integrity.json: every input-bin status, fixed rejection
   reasons, row/support/equivalence checks, frozen state and independent metrics.
4. velocity.png: actual target versus all five predictors on the fixed first
   ten-second clip. metrics.png: full-partition RMSE and mean-R² bars, no CIs.
5. cursor_replay.mp4 and still.png: reference target-velocity integration plus
   ridge, representative GRU and Kalman, shared physical axes and single relative
   zero anchor, no correction/smoothing, identical Phase 6C integration semantics.
   1600×900,50 fps,500 frames,H.264 CRF20 medium, yuv420p; still frame250.
6. protocol.json, access_event.json, run.json and verification.json: frozen
   spec/model/code/source/split/scaler/environment identities, event/mode,
   artifacts/hashes and explicit success/failure status. Everything under ignored
   runs/. Validation rehearsal is visibly labeled validation, never final test.

Timestamp-only clip rule: take the tenth complete grid-bin end in the first
retained test segment, based solely on saved boundaries/grid, and use [t0,t0+10).
It resolves **before data access** to **[1606.320,1616.320) s**, IDs15916–16415.
The grid-derived pre-clip anchor is1606.300 s. If this predetermined clip lacks
500 contiguous eligible rows, stop presentation/evaluation and retain failure;
do not choose a later/better-looking interval. No trajectory is inspected to
select bounds. Original tracker alignment is audited only after authorization;
reference always remains labeled tracker-derived reference velocity integration.

## Fail-closed execution and freeze boundary

Stop before presenting scores if any source, split/config, code, environment,
model/scaler/channel/unit hash or context differs; fitting provenance/mean/prior
disagrees; support crosses a partition or uses future samples; identities,
model state, row count, numerical finiteness, offline/replay outputs or independent
metrics fail; or the fixed clip cannot be produced. Constant-axis null R² is a
documented metric limitation, not a reason to change targets or reject RMSE.
Keep an explicit failure log; no best-effort model dropping or automatic refit.

The command defaults to no data opening: explicit operations preflight,
validation, test. Test additionally requires the exact current freeze SHA-256,
the same passing validation rehearsal evidence, and committed freeze plus
implementation bytes. Missing/incorrect authorization is rejected before the
recording loader or partition preparation. A separate authorized invocation
records a test-access event before constructing test examples. No such invocation
is run in Phase 7A. Existing recording loading for a validation rehearsal still
reads the acquired file/global metadata; only validation examples are constructed.

Any real defect must be documented and corrected before test opening, with
affected validation checks rerun and a revised freeze. If an integrity failure
is first found after access, stop, disclose exposure and amend transparently;
do not tune until a result improves or describe a reopened test as untouched.
A later within-session segment is one chronological generalization check, not
evidence about new sessions, humans, live BCI control or independent trials.

## Frozen identity and completed rehearsal

Canonical JSON SHA-256 (sorted keys, compact separators; independent of CRLF):
`e69c7f3181c479fdf0df076d256e7beb1cc7197c758deee7e8fab805c23832fc`.
The JSON binds implementation/configuration files by LF-normalized SHA-256,
models/reference predictions by raw file hashes, and source/split metadata by
canonical hashes. The zero baseline implementation hash is
`8e1509fbf6ac0d7536bc09565af9619d5d17af0910334e8cfe9d2f85bf0e34df`.
Changing bound code, dependencies, models or scientific rules requires a revised
freeze and another documented validation rehearsal before test access.

Phase 6C was verified at main commit
`f94484225d9170ecf2dddb1365779e150dd24745`, equal to the remote branch at inspection.
Its report, source/artifact hashes, independent integration audit, video probe
and decoded first/middle/last frames were checked; its 13 focused tests passed.
Tracked work initially was clean. The unrelated untracked `%SystemDrive%/`
directory is preserved. No Git staging, commit or push was performed.

The frozen command completed validation only under
`runs/phase7a-validation-dry-run/`: all 2,775 common rows, all five models,
fixed validation clip and media, successful independent artifact-only audit.
Saved `run.json` SHA-256:
`54398f6f4a63e39845d7bde43ce51e32eca1ed872e7d3c022cff5a3aaeb4c34b`.
Maximum absolute sequential/offline differences [mm/s]: zero 0, training mean 0,
ridge 3.126388037344441e-13, Kalman 0, GRU 0 (batch size one).
Established Phase 5B/6A prediction comparisons also passed; parameters were
unchanged. Validation overall RMSE [mm/s]: zero 72.817486, training mean
72.817654, ridge 52.571380, Kalman 102.838507, GRU 54.791919.
These are rehearsal/development scores, not final test results.

All 346 tests passed, including 14 new final-evaluation tests covering synthetic
support/alignment/leakage, label-independent prediction, frozen parameters,
independent metrics, authorization-before-loading and artifact corruption.
Ruff lint/format, offline lock consistency and locked environment checks passed.
No scientific defect, new dependency or unresolved implementation blocker was
found. The validation rehearsal rendered the fixed clip and its labels were
inspected. No test operation was invoked; no test examples, predictions, plots
or metrics were generated. Test eligibility/count remains unknown.

## Exact commands and review boundary

Run from the repository root with the existing locked environment. Read-only
preflight verifies artifacts, code and environment without loading the recording:

```powershell
uv run --locked python scripts/evaluate_final.py preflight
```

The following validation-only commands were completed in Phase 7A:

```powershell
uv run --locked python scripts/evaluate_final.py validation --run-id phase7a-validation-dry-run
uv run --locked python scripts/audit_final.py phase7a-validation-dry-run
```

Outputs are listed above and retained under ignored `runs/`. Existing output
directories and verification files are never overwritten. Do not delete the
rehearsal to rerun it: a changed implementation needs a new documented freeze
and run identity. Repeat its independent read-only audit without rewriting files:

```powershell
uv run --locked python -c "import sys; from pathlib import Path; sys.path.insert(0, 'scripts'); from audit_final import audit; print(audit(Path('runs/phase7a-validation-dry-run')))"
uv run --locked python -m pytest -p no:cacheprovider --tb=short
uv run --locked ruff check .
uv run --locked ruff format --check .
uv lock --check --offline
uv sync --locked --offline --check
```

**Future authorized evaluation only — NOT RUN in Phase 7A.** After review and a
focused commit containing the frozen implementation/configuration, a separate
instruction to open test is required. The prescribed future commands are:

```powershell
uv run --locked python scripts/evaluate_final.py test --run-id phase7b-final-test --open-test-sha256 e69c7f3181c479fdf0df076d256e7beb1cc7197c758deee7e8fab805c23832fc
uv run --locked python scripts/audit_final.py phase7b-final-test
```

Expected future location: `runs/phase7b-final-test/`, containing the exact tables,
predictions, figures, video, manifests and audit specified above. A failed check
stops execution and retains a failure log; incomplete outputs are not a passed
evaluation. Review any failure before further test access. The successful
validation audit is required by the future gate, so preserve local ignored
artifacts. No test output directory exists at the Phase 7A checkpoint.

A later segment of one recording estimates performance under that recording's
conditions. It cannot establish between-session/person generalization, clinical
usefulness, live closed-loop control or a hard real-time guarantee. Selecting a
good-looking test clip after inspecting predictions would favor the presentation
even if the full-partition score were unchanged. Use the frozen timestamp rule.
This document and its commit do not themselves authorize test access.
