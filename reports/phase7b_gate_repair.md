# Pre-test repair: committed-code UTF-8 integrity gate

This repair addresses the [committed Phase 7B blocker](phase7b_final.md), not a
held-out evaluation. Test examples, predictions, plots and metrics remain
unopened. Original Phase 7A commit e64dc2a and blocker commit
`c7638acd31d182aefff9888de13e04ac0bf5a586` are preserved. Branch main began with
clean tracked files/index and the unrelated untracked `%SystemDrive%/` directory;
that work was preserved. No applicable AGENTS.md was found. No agent staging,
commit, push or history rewrite.

## Exact defect and repair

The old gate called `git show HEAD:<path>` with `text=True`, then re-encoded the
returned string before SHA-256. On this Windows process, subprocess text uses
cp1252, while the Git blobs contain UTF-8. Non-ASCII source therefore changed
before hashing even though HEAD/local/frozen source identities were identical.
For scripts/evaluate_baselines.py the expected/explicit UTF-8 hash is
`2e92954c953b46d259bb313cc53910275330ab13cbd38c14df4e9b1371237349`;
the old default-decoded hash was
`52bd557f25ba2bb02911962cb37f6a95fa26e1116494af27df7fa3c4e10308d9`.

`scripts/evaluate_final.py:read_committed_utf8` now captures binary stdout from
`git show HEAD:<path>`, decodes strict UTF-8, normalizes CRLF/lone CR to LF and
returns the text. Hash its UTF-8 encoding, exactly matching local preflight's
`Path.read_text(encoding="utf-8")` universal-newline behavior. No whitespace
stripping, Unicode normalization, BOM removal, replacement/ignored characters,
working-tree substitution or hash relaxation. Bad UTF-8 and Git errors propagate.
The freeze JSON uses the same reader; it retains exact canonical equality.
`verify_committed_implementation` checks every declared file against HEAD and is
called by the original `require_committed_freeze` before its evidence gate.

An all-file historical audit accepted all 29 authentic committed implementation
identities, the original HEAD freeze and original validation evidence. It also
ran the independent original validation artifact audit successfully without a
recording loader. Explicit simulation of old cp1252 decoding found four affected
files: scripts/evaluate_baselines.py, scripts/evaluate_final.py,
src/neurodecode/evaluation.py and src/neurodecode/plotting.py. The shared repair
handles all of them. Other text-mode subprocesses report Git commit/status or
ASCII encoder/probe metadata; they do not obtain/hash committed source blobs.
They were inspected and left outside this focused source-identity repair.

Seven new regression cases run on every platform: non-ASCII UTF-8 (`R² — café`)
under LF/CRLF/CR, genuine changed Git content, invalid UTF-8, Git read failure,
and non-ASCII freeze JSON with exact-identity rejection. The encoding cases
explicitly reproduce the wrong old cp1252 hash. Mock Git stdout is bytes and
asserts binary capture, so a text-mode regression cannot merely pass on a UTF-8
development host. The complete historical HEAD check supplies real Git evidence
in addition to these synthetic fixtures. No tests construct real test examples.

## Explicit freeze revision before data access

Parent canonical freeze:
`e69c7f3181c479fdf0df076d256e7beb1cc7197c758deee7e8fab805c23832fc`.
Revised canonical freeze:
`61941d4f1ed6226b2af3a80c5a79c0c88f5f374186ff5f6be047ce75094e6f28`.
Only one bound file hash changes: scripts/evaluate_final.py,
`3baae9fdc072a7bc2c0a54cfb598ccb330620f20ee6f94918ef2ee1bb27d5088`
to `bcd83243753708010c2a3a0102fc85eee78acae2006643d3a452675dc9db7632`.

The other changed JSON fields are phase description, explicit parent/repair
provenance and validation evidence ID `phase7a-utf8-gate-validation`. All other
fields were compared and remain identical: scientific preprocessing, all model/
source/split/scaler/environment identities, eligibility/metrics/tolerances,
resets, test run ID and timestamp-selected clip/export settings. No optimizer,
model selection, target/exclusion change or alternate clip. The revised runner
hash necessarily changes the freeze's own digest; using the old token would
misidentify the executable recipe. The [specification amendment](../docs/final_evaluation.md)
documents this revision before the validation-only rehearsal.

Old/new gate behavior is deliberately distinct: authentic historical HEAD now
passes, genuinely altered content fails, and the revised **uncommitted** freeze
still fails with `Current freeze is not committed exactly at HEAD`. Local
preflight is not a substitute for that test gate. After the user's commit the
same committed gate must accept the revised HEAD plus its audited validation
evidence before a later authorized test execution. No test-mode command was
called to demonstrate any of these checks.

## Reproduction and evidence

From the repository root, the completed preflight/committed-identity checks use:

```powershell
.venv/Scripts/python.exe scripts/evaluate_final.py preflight
.venv/Scripts/python.exe -c "import sys,json; sys.path.insert(0,'scripts'); from evaluate_final import read_committed_utf8,require_committed_freeze; require_committed_freeze(json.loads(read_committed_utf8('configs/final_evaluation.json'))); print('committed identities and validation evidence passed')"
.venv/Scripts/python.exe -u scripts/evaluate_final.py validation --run-id phase7a-utf8-gate-validation
.venv/Scripts/python.exe scripts/audit_final.py phase7a-utf8-gate-validation
```

Pre-commit historical checks are saved under ignored
`runs/phase7b-gate-repair-preflight/`: committed_identity_audit.json,
freeze_revision.json and uncommitted_gate.json. The original failure JSON and
original Phase 7A rehearsal remain untouched. New validation output remains
under ignored `runs/phase7a-utf8-gate-validation/`, with protocol, access event,
source/model/code identities, predictions, metrics, integrity/quality/exclusions,
fixed validation figures/video/still and independent verification. Existing
directories are not overwritten; these commands are recorded history, not an
instruction to replace preserved outputs.

Repository checks:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_committed_integrity.py tests/test_final_evaluation.py -p no:cacheprovider --tb=short
.venv/Scripts/python.exe -m pytest -p no:cacheprovider --tb=short
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check .
uv --cache-dir .uv-cache lock --check --offline
uv --cache-dir .uv-cache sync --locked --offline --check
git -c safe.directory=C:/Users/disar/projects/neuro-decode diff --check
```

Focused 21 tests passed; full suite 353 passed in 30.89 seconds (346 retained,
seven new). Ruff lint/format and offline lock/environment checks passed; no
dependency changes. Local revised preflight calls no recording loader. Real
validation preparation retains the established whole-recording/global-metadata
loader, but only validation windows/targets are constructed. No test examples
or metrics are constructed by this repair.

The test partition remains unopened and cannot be described as evaluated. No
held-out claims, new latency/robustness study or cross-session conclusions.
Stop for repair review/commit and later committed-gate verification; a separate
test-opening instruction is required.

## Validation-only numerical verification

The revised recipe retains all 2,775 validation rows, target-end timestamps
1550.200 through 1605.680 s. All five predictors cover 100% of the common eligible
population; nine initial history-only bins remain excluded by the same rule.
No new quality rejection. The table is development validation, not test:

| Model | RMSE x | RMSE y | Overall RMSE | R² x | R² y | Mean R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero | 78.418547 | 66.748063 | 72.817486 | -0.000000 | -0.000000 | -0.000000 |
| Training mean | 78.418755 | 66.748186 | 72.817654 | -0.000005 | -0.000004 | -0.000005 |
| Ridge | 56.844212 | 47.919052 | 52.571380 | 0.474546 | 0.484606 | 0.479576 |
| Kalman | 93.653913 | 111.267523 | 102.838507 | -0.426311 | -1.778814 | -1.102563 |
| GRU 2026 | 58.990834 | 50.243310 | 54.791919 | 0.434111 | 0.433397 | 0.433754 |

RMSE units mm/s. Sequential/offline maximum absolute differences [mm/s] remain
zero 0, mean 0, ridge 3.126388037344441e-13, Kalman 0, GRU 0. Frozen parameters
remain unchanged and established Phase 5B/6A predictions pass comparison.
New predictions.csv, metrics.csv and metrics.md are byte-identical to original
Phase 7A output. Predictions SHA-256:
`4eb4b5309586102548bd810d726bd8e81ca5ea23f4d17aa4ff93dc990ba5718a`;
metrics.csv SHA-256:
`6c24f2b818032e5fd1b4a3cdda4846b9edb59e50a6a62e6d4abc16a6775e8e6e`.
The encoding repair changes provenance verification, not numerical inference.

The complete validation command and independent artifact-only audit passed.
New run.json SHA-256:
`8adb942e7f0d6a00884bf0029b8b2dce636ba2cf3e04744b283a885bd65377cf`.
The audit recomputes all five models' metrics from exported CSV with Python
fsum (absolute tolerance 1e-10), verifies exact source/behavior row identities,
partition/segment support, unchanged parameters and all saved artifact hashes.
`runs/phase7b-gate-repair-preflight/validation_comparison.json` independently
records exact old/new CSV and metric equality and preserved failure evidence.

The unchanged fixed validation clip [1550.200,1560.200) s exported successfully:
500 frames, 50 fps, 10 seconds, 1600×900. Decoded first/middle/last MP4 frames
were visually inspected: readable validation/offline/reference caveats, elapsed
times 0/5/9.98 s, fixed common physical axes and unhidden Kalman drift. No interval
or presentation setting was adjusted. The independent audit passed after video
completion, and runs/phase7b-final-test/ remains absent.
