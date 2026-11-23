# Phase 7B: blocked before held-out test access

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
