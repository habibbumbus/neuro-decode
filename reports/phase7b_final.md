# Phase 7B: frozen held-out test evaluation

The single authorized frozen test evaluation completed on 2026-11-23, after
the committed UTF-8 gate repair. All 2,784 common eligible rows and the independent
audit passed. Ridge has the lowest predeclared overall RMSE (60.421673 mm/s),
GRU seed 2026 scores 67.360708 mm/s, and Kalman scores 134.059526 mm/s, worse
than both trivial baselines. No model, exclusion, metric or clip was changed.
This is offline decoding from one recorded nonhuman-primate session and one
held-out time block; it does not establish transfer to another session.

## Complete test results

All five models score the same 2,784 rows, bin IDs 15916–18699, target-end times
1606.320–1661.980 s inclusive (55.660 s timestamp span). RMSE is mm/s; R² is
dimensionless. Overall RMSE averages squared error over all rows and both axes;
mean R² is the unweighted axis mean. All R² values are defined. Full-precision
values are retained in metrics.csv/run.json; near-zero baseline R² is negative.

| Model | N | RMSE x | RMSE y | Overall RMSE | R² x | R² y | Mean R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero velocity | 2784 | 80.540636 | 85.667370 | 83.143527 | -0.000027 | -0.000026 | -0.000027 |
| Training mean | 2784 | 80.539999 | 85.668194 | 83.143644 | -0.000011 | -0.000046 | -0.000029 |
| Ridge, lambda 1000 | 2784 | 65.332476 | 55.074718 | 60.421673 | 0.341979 | 0.586681 | 0.464330 |
| Frozen Kalman | 2784 | 99.887504 | 161.140931 | 134.059526 | -0.538169 | -2.538281 | -1.538225 |
| Representative GRU, seed 2026 | 2784 | 69.875487 | 64.748331 | 67.360708 | 0.247283 | 0.428735 | 0.338009 |

There are 2,793 complete source-count bins in one retained continuous segment;
nine initial history-only bins are excluded from scoring by the original rule.
There are zero additional invalid-window exclusions and no model-specific
exclusions. Every model emits all 2,784 eligible predictions: 100% eligible-row
coverage. Existing grid/retained-boundary trimming excludes approximately 20 ms
at either end before these source bins; it was not modified to improve scores.
exclusions.csv and quality_summary.json retain reasons and exact durations.

## Comparison with clean development validation

Validation is the preserved revised Phase 7A rehearsal, all 2,775 eligible rows
at 1550.200–1605.680 s. It is not the disruption-masked Phase 6B population.
Each partition uses common rows across models; the two partitions naturally
contain different target movements. No model is selected or refitted here.

| Model | Validation overall RMSE | Test overall RMSE | Validation mean R² | Test mean R² |
| --- | ---: | ---: | ---: | ---: |
| Zero velocity | 72.817486 | 83.143527 | -0.000000117 | -0.000027 |
| Training mean | 72.817654 | 83.143644 | -0.000004595 | -0.000029 |
| Ridge | 52.571380 | 60.421673 | 0.479576 | 0.464330 |
| Kalman | 102.838507 | 134.059526 | -1.102563 | -1.538225 |
| GRU 2026 | 54.791919 | 67.360708 | 0.433754 | 0.338009 |

Every overall RMSE is higher on this test block. The zero predictor's higher
error also shows that target movement magnitudes differ; absolute RMSE change
alone is not a controlled diagnosis of model deterioration. Ridge retains the
lowest primary metric. GRU improves on the trivial predictors but does not beat
ridge. Kalman remains poor on both axes; negative R² means worse squared error
than each scoring axis's own mean reference, not an input supplied to the model.
This run does not uniquely identify the cause of degradation or establish that
it represents electrode instability. No extra Kalman settling exclusion,
calibration, new GRU epoch/seed, ensemble or alternate metric was introduced.
Seeds 2027/2028 remain validation-only development results, as frozen.

## Freeze, provenance and opening event

HEAD and a fresh GitHub main check both returned repair commit
`7d9c50cf52844f7ff2b0b05b65f402f629983a94`. All 29 committed file identities,
exact committed freeze, local model/scaler/reference/environment preflight and
the revised validation artifact audit passed before access. Comparison with
original Phase 7A confirmed only the documented encoding repair/provenance/
validation-evidence identity changes; all scientific fields remained identical.
Tracked tree/index were clean at execution. run.json correctly records dirty=true
because the unrelated untracked `%SystemDrive%/` directory remains present;
there were no tracked source/configuration changes. No agent Git mutation.

Canonical frozen protocol SHA-256:
`61941d4f1ed6226b2af3a80c5a79c0c88f5f374186ff5f6be047ce75094e6f28`.
The test-access event was recorded at **2026-11-23T18:25:42.958569+00:00**, before
source loading and explicit test-partition construction. The test command ran
once and exited 0; the independent audit exited 0. No failures or corrective
test run. Earlier whole-recording/global metadata loading during development
was already disclosed; this is the first test example/prediction/scoring run,
not a claim that the recording's test-interval bytes were never read previously.

- Session indy_20161005_06, Zenodo 583331 v1; source SHA-256
  4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac.
- Canonical split SHA-256
  3e54457065c7bf9d71fb2eef4a5bf7710d7d498fbf84bf1995d59ad11107ee13;
  retained test [1606.1000000000001,1662.0) s, unchanged chronological purge.
- Frozen training scaler SHA-256
  603aa299f0a1d75625acea41a18b9ecf56c58e4e0515ec8322ea1aa34abe68dd;
  13,079 unique training bins, all 96 ordered M1 channels, ten-bin history.
- Target-policy SHA-256
  2e86a91a0f1735f1ff3f5d5e2d0ce2781cca25ecf979e2d99e720652df90189b;
  completed 20 ms neural bins and actual-behavior-elapsed velocity targets.

| Artifact | SHA-256 |
| --- | --- |
| runs/phase3a-baselines/run.json, frozen mean from 13,070 training targets | 6ed540be6c1e22520bb82d678f82fae77a12cd5a7998586d15b04035e0e58f1d |
| runs/phase3b-ridge/ridge.json, lambda 1000 | 43455b85f99928b785296890d5fa316820d8e9f071d77883f6743efe1bcf36ee |
| runs/phase4a-kalman-fit/kalman_parameters.json | 5b907ca94f805a7d244c903b9fd26470271a3fb723100dcd59dbb05297884507 |
| runs/phase5b-gru-round1/lr-0.003-seed-2026/gru.pt, epoch 50, 12,546 parameters | 9e5098ceb41fa1cf77c080786be06497396b6022d24e74cd269006179a97d512 |

Zero has no learned parameters; its bound implementation and the complete
channel/unit map, transforms, code hashes and versions are in protocol.json.
Python 3.11.17, NumPy 2.4.6, PyTorch 2.14.1+cpu; all remaining frozen packages and
FFmpeg/ffprobe identities passed preflight. GRU CPU float32, ridge/Kalman float64,
one torch/native thread. No model loading outside trusted project runs/.

## Independent audit and reset checks

| Predictor | Max absolute offline/replay difference [mm/s] |
| --- | ---: |
| Zero | 0 |
| Training mean | 0 |
| Ridge | 3.410605131648481e-13 |
| Kalman | 0 |
| GRU 2026, offline batch size one | 0 |

All differences pass the unchanged atol/rtol rules. Every emitted timestamp,
partition, segment and ten-bin source identity matches the offline interface.
Each decoder has one initial reset at bin 15916; no later gaps in this block.
GRU starts every bounded window from zero hidden state; Kalman starts from its
training prior, then carries only estimated state. No held-out movement values
initialize or correct a decoder. Frozen parameters and scaler digests match
before/after prediction. No normalization fitting, optimizer updates or training.

The artifact-only audit verifies saved hashes and exact row/support identities
and recomputes every model's axis/overall RMSE and axis/mean R² independently
using Python fsum from predictions.csv, absolute metric tolerance 1e-10. It loads
no recording or model and performs no second inference run. All checks passed.
run.json SHA-256 recorded by verification.json:
`87c10c06a2635a3b506b693cf2d024c3cd39ecadf971a521ecea32f803c94bc2`.
All 353 repository tests passed in 31.14 seconds; Ruff lint/format and offline
lock/environment checks passed. No dependency, implementation or freeze change.

## Predetermined figures, video and reference limitation

Only the specified metrics.png, velocity.png, cursor_replay.mp4 and still.png
were generated, plus specified inspection frames. The clip is unchanged
**[1606.320,1616.320) s**, IDs 15916–16415, 500 eligible consecutive endpoints.
Reference/ridge/GRU/Kalman start once at relative zero at 1606.300 s. Integrate
each velocity over its actual neural-bin duration; no smoothing, shifts, snapping,
periodic resets or future reference corrections. No better-looking clip was
selected after seeing results, and poor Kalman performance remains visible.

The alignment audit reconstructs target velocities exactly, but 40/500 selected
behavior intervals last 16/24 ms rather than 20 ms. Integrated targets differ
from original sampled tracker coordinates by up to 1.163070053 mm per coordinate.
The reference therefore remains labeled **tracker-derived reference velocity
integration**, not original tracker positions.

The MP4 is 1600×900, 500 frames at 50 fps, 10 seconds, with the frozen H.264
settings. Decoded frames 0/250/499 were inspected: correct offline intracortical/
frozen-held-out/not-live labels, times 1606.320/1611.320/1616.300 s and elapsed
0/5/9.98 s, unchanged common physical axes x[-379.247263,871.266709],
y[-237.548640,1012.965332] mm. Large Kalman drift is unhidden; other trajectories
appear small on that required common scale. A visually closer local integrated
path does not supersede the full-block velocity metric. Playback is not a new
latency measurement or a live-control demonstration.

## Outputs, exact commands and stop

All generated artifacts remain ignored under runs/phase7b-final-test/: complete
predictions.csv, metrics.csv/md, exclusions.csv, quality_summary.json,
integrity.json, protocol.json, access_event.json, run.json, verification.json,
the two plots, MP4, still and inspection frames. Pre-access evidence, command
output and execution_summary.json are under runs/phase7b-opening-preflight/.
Full command-output hashes: test-command.log
`395488a0fd4f5a000dd28d59d731474aa89e9e30646aa4c0417ecc076ac923f2`;
audit-command.log
`a04c29cb0c67bbca978bb8bbc3c96077f04fc933f0f521acb4e7f996dc501f21`.

The exact frozen commands were executed once with output capture:

```powershell
uv run --locked python scripts/evaluate_final.py test --run-id phase7b-final-test --open-test-sha256 61941d4f1ed6226b2af3a80c5a79c0c88f5f374186ff5f6be047ce75094e6f28 2>&1 | Tee-Object -FilePath runs/phase7b-opening-preflight/test-command.log
uv run --locked python scripts/audit_final.py phase7b-final-test 2>&1 | Tee-Object -FilePath runs/phase7b-opening-preflight/audit-command.log
```

The shell propagated a nonzero native exit code if present. These are recorded
reproduction commands, not instructions to rerun or delete evidence. Existing
output directories/verification files are refused. Later review can repeat the
artifact-only audit without rewriting verification or reopening the recording:

```powershell
.venv/Scripts/python.exe -c "import sys; from pathlib import Path; sys.path.insert(0,'scripts'); from audit_final import audit; print(audit(Path('runs/phase7b-final-test')))"
```

Repository checks used the existing interpreter and local uv cache:

```powershell
.venv/Scripts/python.exe -m pytest -p no:cacheprovider --tb=short
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check .
uv --cache-dir .uv-cache lock --check --offline
uv --cache-dir .uv-cache sync --locked --offline --check
git -c safe.directory=C:/Users/disar/projects/neuro-decode diff --check
```

No unresolved execution/audit blocker. No retuning, retraining, new seeds,
disruptions, extra test condition, alternate clip or corrective repeat. Test is
now opened; any later test-informed scientific changes would be exploratory and
cannot reuse this block as an untouched final estimate. This one-session,
one-block result cannot establish cross-session/participant transfer, human or
clinical usefulness, live closed-loop control or a hard real-time guarantee.
Overlapping 200 ms windows and correlated time bins are not independent trials;
no significance/confidence interval is claimed. Tracker-derived finite-difference
labels and quality flags also limit interpretation. Phase 7B contains the single
frozen evaluation described above; no additional experiment is implied.

## Historical pre-access blocker (preserved)

On 2026-11-23, the required committed-implementation integrity gate failed.
**The frozen test command was not invoked. No test examples, predictions,
figures, cursor replay or metrics were produced.** There are no held-out results
to report or compare with validation. This is a blocker report, not a completed
final evaluation. The [Phase 7A specification](../docs/final_evaluation.md) remains
unchanged.

## Verified starting state

Read CODEX.md, the freeze specification and relevant progress, methods,
decisions and model/replay reports. No applicable AGENTS.md was found. Branch
main HEAD and a fresh read-only GitHub main check both returned
`e64dc2a545f3f1c26afa4459386aa2a13c58377f` (Phase 7A). Tracked files/index were
clean. The unrelated untracked `%SystemDrive%/` directory was preserved.
CODEX.md, recordings, checkpoints and runs remain ignored/untracked. No agent
staging, commit or push.

The unchanged canonical freeze SHA-256 is
`e69c7f3181c479fdf0df076d256e7beb1cc7197c758deee7e8fab805c23832fc`.
Local preflight passed: implementation/configuration hashes, canonical
source/split metadata, model/reference artifact hashes, saved scaler/model
context, Python/packages and export tool versions. Preflight called no recording
loader. The retained test boundary [1606.1000000000001,1662.0) s, all five models,
common-row rules, primary overall RMSE in mm/s, secondary metrics and fixed
[1606.320,1616.320) s clip were not changed.

## Failure and byte-level diagnosis

The next pre-access check, `require_committed_freeze`, raised:

```text
ValueError: Frozen implementation is not committed at HEAD: scripts/evaluate_baselines.py
```

This is a false mismatch from text decoding, not evidence of changed scientific
code. The gate reads `git show` using `subprocess.run(..., text=True)` without an
explicit encoding. This Windows Python process reports locale encoding cp1252.
Git emits UTF-8 source; default decoding changes non-ASCII characters before the
gate re-encodes and hashes the string. An independent byte-level comparison found:

| Representation | SHA-256 |
| --- | --- |
| Frozen expected UTF-8/LF | 2e92954c953b46d259bb313cc53910275330ab13cbd38c14df4e9b1371237349 |
| Local explicit UTF-8/LF | 2e92954c953b46d259bb313cc53910275330ab13cbd38c14df4e9b1371237349 |
| Committed bytes decoded explicitly as UTF-8/LF | 2e92954c953b46d259bb313cc53910275330ab13cbd38c14df4e9b1371237349 |
| Committed source via gate's default text decoding | 52bd557f25ba2bb02911962cb37f6a95fa26e1116494af27df7fa3c4e10308d9 |

Explicitly decoded committed content equals local content exactly. No source,
weights, split, freeze, exclusion policy or environment setting was changed to
bypass the failure. In particular, test mode was not retried under a different
encoding environment. The gate stopped before reaching its validation-evidence
check; the intended follow-on validation artifact audit in that Python command
also did not execute. The existing Phase 7A audit remains preserved, but this
round did not reach a fresh audit or test inference.

Saved diagnostic evidence:
`runs/phase7b-preaccess-integrity-20261123/failure.json`, recorded at
2026-11-23T17:56:26.169665+00:00. This is a pre-access diagnostic, not the test
opening event. `runs/phase7b-final-test/` does not exist.

## Commands and checks

These commands ran before test access:

```powershell
git -c safe.directory=C:/Users/disar/projects/neuro-decode status --short
git -c safe.directory=C:/Users/disar/projects/neuro-decode log -5 --oneline
git -c safe.directory=C:/Users/disar/projects/neuro-decode branch -vv
git -c safe.directory=C:/Users/disar/projects/neuro-decode ls-remote --heads origin main
.venv/Scripts/python.exe scripts/evaluate_final.py preflight
.venv/Scripts/python.exe -c "import sys,json; from pathlib import Path; sys.path.insert(0,'scripts'); from evaluate_final import require_committed_freeze,canonical_sha; from audit_final import audit; f=json.loads(Path('configs/final_evaluation.json').read_text()); require_committed_freeze(f); print('COMMITTED FREEZE AND VALIDATION EVIDENCE: PASS'); print(audit(Path('runs/phase7a-validation-dry-run'))); print('FROZEN CLIPS',f['clips']); print('FROZEN SOURCE',f['source'])"
```

The last command stopped at the committed-code gate, before its scheduled
artifact-only validation audit and success prints. The initial sandboxed remote lookup failed
to connect; the approved read-only retry succeeded. No scientific command ran
through that networking failure.

All 346 existing tests passed in 34.59 seconds, including the 14 final-evaluation
tests. Ruff lint/format, git diff whitespace, offline lock consistency and locked
environment checks passed; no dependencies changed. Exact check commands:

```powershell
.venv/Scripts/python.exe -m pytest -p no:cacheprovider --tb=short
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check .
git -c safe.directory=C:/Users/disar/projects/neuro-decode diff --check
uv --cache-dir .uv-cache lock --check --offline
uv --cache-dir .uv-cache sync --locked --offline --check
```

These existing tests do not catch the observed default-encoding defect; passing
synthetic tests cannot override the actual gate failure. No new regression test
was added in this stopped checkpoint. No test metrics or plots were inspected
and no test replay was available for visual inspection. No new model/scientific
implementation was made in this blocked checkpoint.

## Required next step and scope

Stop under the frozen failure policy. A separately reviewed repair should use
explicit UTF-8 decoding for committed source (or hash normalized bytes), add a
non-ASCII/Windows-encoding regression test, update the bound implementation hash
and freeze identity, and rerun affected validation/rehearsal checks with a new
documented evidence identity. Preserve the old freeze/rehearsal and this failure
trail. Review/commit/push the revised freeze before requesting test execution.
Do not silently work around this gate or reuse the old digest with changed code.

The intended measurement remains frozen decoding accuracy on one session and
one later held-out time block. Since this checkpoint stopped before access,
there is no new estimate yet. Even successful future held-out results would not
establish cross-session/person generalization, clinical benefit, live brain
control or a hard real-time guarantee. Validation remains development evidence;
the preselected clip must not be replaced because its future output looks poor.
