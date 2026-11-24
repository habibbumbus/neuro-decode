# Setup, review and reproduction

[Project overview](../README.md) · [Frozen specification](final_evaluation.md) · [Final results](../reports/phase7b_final.md)

There are three different tasks: install/test the public source; independently rebuild development validation artifacts; or reproduce/audit the exact historical frozen result. The last task needs the original trusted ignored artifacts. New training runs are not substitutes for their hashes. Phase 7C performed no retraining or test inference.

## Setup

Use Python 3.11.17 for the recorded environment. The walkthrough used uv 0.12.23. Install uv using its [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/); on Windows with the Python launcher, the equivalent pinned installation is:

```powershell
py -m pip install --user uv==0.12.23
```

From a new clone, these commands are shared by PowerShell and POSIX shells:

```text
git clone https://github.com/habibbumbus/neuro-decode.git
cd neuro-decode
uv python install 3.11.17
uv sync --locked --python 3.11.17
uv run --locked python -I -c "import sys, neurodecode; print(sys.executable); print(neurodecode.__file__)"
uv run --locked python -m pytest -p no:cacheprovider
uv run --locked ruff check .
uv run --locked ruff format --check .
uv lock --check
```

`--locked` preserves the committed resolution; it does not remove the need to download missing packages or build requirements. `--offline` requires a populated cache. The package import must point to this clone's `src/neurodecode/`, not another checkout. The lock selects CPU PyTorch on Windows/Linux; macOS uses PyPI. Only the Windows walkthrough below was executed. Cross-platform setup is a proposed recipe, not a verified byte-identical scientific reproduction.

`pyproject.toml` declares dependencies and the editable package, `.python-version` gives the supported Python line, and `uv.lock` fixes resolved packages. Environments/caches are ignored. Activation is optional when using `uv run`; editor interpreters are `.venv/Scripts/python.exe` on Windows and `.venv/bin/python` on POSIX.

After a successful sync, cache-restricted read-only checks can use:

```text
uv run --locked --no-sync ruff check --no-cache .
uv run --locked --no-sync ruff format --check --no-cache .
uv run --locked --no-sync python -m pytest -p no:cacheprovider
```

An optional local cache is `uv --cache-dir .uv-cache ...`; no cache contents are committed. No CI workflow is currently provided. Passing local tests is not evidence that a hosted CI run occurred.

### Windows pytest temporary-directory permissions

If test setup raises `PermissionError` while scanning the shared
`%TEMP%/pytest-of-<user>` directory, this is a temporary-directory access failure,
not a completed scientific test. Keep the original error record and select a
**new, absent** directory under ignored `runs/` for the synthetic suite:

```powershell
if (Test-Path -LiteralPath runs/pytest-review-unique) { throw 'Choose a new absent directory' }
New-Item -ItemType Directory -Force runs | Out-Null
uv run --locked python -m pytest -p no:cacheprovider --basetemp=runs/pytest-review-unique
```

pytest may clear its chosen base directory, so do not point `--basetemp` at
existing runs, data or the repository root. This changes only fixture storage,
not models, metrics, the scientific freeze or test-partition evaluation. The
[Phase 7D audit](../reports/phase7d_audit.md) records the observed default-path
failure and workspace-temp recheck; no user temp-folder permissions were changed.

## Public raw input

Download only `indy_20161005_06.mat` from the version-specific [Zenodo v1 file](https://zenodo.org/records/583331/files/indy_20161005_06.mat?download=1), place it at `data/raw/indy_20161005_06.mat`, and verify it against [data/manifest.json](../data/manifest.json). Do not overwrite an existing recording. The SHA-256 must be:

```text
4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac
```

PowerShell: `Get-FileHash -Algorithm SHA256 data/raw/indy_20161005_06.mat`.
POSIX: `sha256sum data/raw/indy_20161005_06.mat` (or `shasum -a 256` on macOS).
The [dataset card](dataset.md) gives byte size, source attribution, CC BY 4.0 license and source-field/physical-unit caveats. The dataset is public, not an access-controlled clinical recording. The repository's code license is still undecided.

## Rebuild validation artifacts

This is the historical development recipe for a separate reproduction, **not executed in Phase 7C**. It fits models and requires verified raw data. Use a separate checkout with absent output directories; keep the shown names because later scripts use them. Preserve existing runs. These commands do not predict or score test, but the legacy `--features` audit constructs aggregate histories/targets for all partitions, including test. It is not a validation-only pre-access command.

```text
uv run --locked python scripts/inspect_recording.py data/raw/indy_20161005_06.mat
uv run --locked python scripts/bin_recording.py --features
uv run --locked python scripts/evaluate_baselines.py phase3a-baselines
uv run --locked python scripts/train_ridge.py phase3b-ridge
uv run --locked python scripts/fit_kalman.py phase4a-kalman-fit
uv run --locked python scripts/evaluate_kalman.py phase4b-kalman-audited
uv run --locked python scripts/train_gru.py phase5b-gru-round1
uv run --locked python scripts/audit_gru.py phase5b-gru-round1
uv run --locked python scripts/replay_validation.py phase6a-validation-timing
uv run --locked python scripts/audit_replay.py phase6a-validation-timing
```

Use the committed split; `scripts/plan_splits.py` is the historical planner, not a required step to overwrite it. `bin_recording.py --features` generates `runs/phase2d/count_scaler.json`. Baseline evaluation supplies the training mean. Ridge fits/selects its declared validation grid; Kalman fits training parameters then evaluates on the same validation rows. The GRU script uses `configs/gru_experiment.toml` and produces the selected representative plus the declared validation-only seed repeats. Replay uses that exact run directory and audits its inputs.

These scripts produce fresh JSON/checkpoints/predictions with their own timestamps, code/environment/provenance. Refit results may differ; even numerically equal models need not have identical serialized bytes. **They do not recreate the exact hash-frozen artifact bundle automatically.** Do not update the freeze hashes just to make a rebuilt artifact pass. [Phase 5B](../reports/phase5b_gru.md), [Phase 6A](../reports/phase6a_replay.md) and the [methods](methods.md) document component scientific contracts and limitations. Processing latency is machine-specific.

## Restore the exact frozen bundle

No downloadable release of the original models/evidence is currently published. Obtain the original trusted files from the project author/original checkout, preserve their relative paths, and verify them against `configs/final_evaluation.json` and the saved run manifests. A fresh clone alone is insufficient; arbitrary external PyTorch checkpoints are not trusted inputs.

| Restore under repository root | Required content/use |
| --- | --- |
| `runs/phase3a-baselines/` | Original `run.json` with frozen training mean and identity |
| `runs/phase3b-ridge/` | Original `ridge.json`, lambda 1000, frozen scaler/context |
| `runs/phase4a-kalman-fit/` | Original `kalman_parameters.json`, dynamics/observation matrices and training prior |
| `runs/phase5b-gru-round1/` | Original representative seed-2026 checkpoint and five-model `predictions.csv`; restore the whole folder for its component audits |
| `runs/phase6a-validation-timing/` | Original sequential validation `predictions.csv` and evidence |
| `runs/phase7a-utf8-gate-validation/` | Revised passing rehearsal `run.json`, `verification.json` and every artifact listed in its manifest |
| `runs/phase7b-final-test/` | Already-completed final predictions, metrics, protocol, manifests, video and verification for read-only review |

The raw recording is needed for new inference but not artifact-only auditing. All model/reference hashes, Python/package versions and the specific FFmpeg/ffprobe build are part of the freeze. Python 3.11 compatibility alone is insufficient for exact preflight. Source/config identities remain bound to the committed implementation; documentation-only commits do not alter those 29 files.

### Safe checks with restored evidence

Local preflight verifies implementation/config/model/environment identities without loading the recording:

```text
uv run --locked python scripts/evaluate_final.py preflight
```

Verify the committed freeze, all 29 declared HEAD file identities and passing validation evidence, also without loading the recording:

```text
uv run --locked python -c "import sys,json; sys.path.insert(0,'scripts'); from evaluate_final import read_committed_utf8,require_committed_freeze; require_committed_freeze(json.loads(read_committed_utf8('configs/final_evaluation.json'))); print('committed freeze and validation evidence passed')"
```

Read-only re-audit of restored validation and final evidence:

```text
uv run --locked python -c "import sys; from pathlib import Path; sys.path.insert(0,'scripts'); from audit_final import audit; print(audit(Path('runs/phase7a-utf8-gate-validation'))); print(audit(Path('runs/phase7b-final-test')))"
```

This verifies hashes, source/timestamp/partition support and independently recomputes all metrics from saved CSV. It makes no model predictions. The CLI auditor writes a new verification file exclusively, so use the read-only function for an existing run.

### Generate an absent validation rehearsal from exact inputs

Only when the exact frozen model/reference bundle is restored, export tools match, raw data is verified, and the fixed validation output directory does not already exist:

```text
uv run --locked python scripts/evaluate_final.py validation --run-id phase7a-utf8-gate-validation
uv run --locked python scripts/audit_final.py phase7a-utf8-gate-validation
```

This prepares validation only, checks established Phase 5B/6A predictions, and produces the prescribed validation exports/evidence. The source loader reads the acquired recording/global metadata; no test examples or test inference are constructed by this mode. Existing directories are protected, not disposable caches.

### Historical final-evaluation command

The following was already executed once for the reported result. It is included for archival reproduction in a separate environment with the original inputs, not a recommendation to run another test experiment or delete existing evidence. No test inference was run in Phase 7C.

```text
uv run --locked python scripts/evaluate_final.py test --run-id phase7b-final-test --open-test-sha256 61941d4f1ed6226b2af3a80c5a79c0c88f5f374186ff5f6be047ce75094e6f28
uv run --locked python scripts/audit_final.py phase7b-final-test
```

The test gate requires committed exact identities and an audited matching validation rehearsal. Outputs include timestamped predictions, metrics/exclusions, frozen protocol, opening event, code/source/model identities and the fixed figures/video/still. Failures stop evaluation; no model dropping, refitting, score-informed exclusions or replacement clip. The [final report](../reports/phase7b_final.md) records the command-output logs and all measured results. The original test block is now used; future test-informed changes need an explicitly exploratory label and new untouched data for another final claim.

## Phase 7C documentation walkthrough

A local fresh clone of committed Phase 7B (`aaf1f8251d1ff98bbf698d66618575c5943d0df4`) was created under ignored `runs/phase7c-doc-walkthrough/clone/`, without raw data, original `runs/`, checkpoints or private local notes. The rewritten documentation/preview is reviewed separately before commit.

| Walkthrough check | Observed outcome |
| --- | --- |
| Local clone of committed source | Passed; no raw file, original model/evidence runs or local-only brief |
| New Python 3.11.17 environment, offline sync | Initially blocked: CPU PyTorch wheel absent from cache |
| Locked installation with network access | Packages installed in the separate clone environment; no changes to the original environment |
| Isolated import outside implicit source paths | Passed; interpreter and NeuroDecode package both resolve to the clone; NumPy 2.4.6 / torch 2.14.1+cpu |
| Locked installed-environment check | Passed, 34 packages checked, no changes required |
| Full synthetic tests in fresh clone | 353 passed in 52.35 seconds, no recording/models required |
| Final-evaluation `preflight` in clone | Expected failure at absent `runs/phase3b-ridge/ridge.json`; no recording loader or test inference |
| Local documentation links/anchors, result table, preview hash | Passed; evidence is `runs/phase7c-doc-walkthrough/document_checks.json` |
| Rendered README visual inspection | Blocked: Computer Use could not confidently verify the current Windows browser URL; a local HTML preview was generated but not visually certified |
| Actual-run demo still inspection | Passed visually; copied without transformation and hash matched to saved Phase 7B manifest |

Commands executed for the clean-clone checks (from the original root for cloning, then from the clone):

```powershell
git -c safe.directory=C:/Users/disar/projects/neuro-decode clone --no-hardlinks --local C:/Users/disar/projects/neuro-decode runs/phase7c-doc-walkthrough/clone
```

```text
uv sync --locked --offline --python C:/Users/disar/AppData/Roaming/uv/python/cpython-3.11-windows-x86_64-none/python.exe
uv sync --locked --python C:/Users/disar/AppData/Roaming/uv/python/cpython-3.11-windows-x86_64-none/python.exe
.venv/Scripts/python.exe -I -c "import sys,neurodecode,torch,numpy; print(sys.executable); print(neurodecode.__file__); print(torch.__version__,numpy.__version__)"
uv --cache-dir C:/Users/disar/projects/neuro-decode/.uv-cache sync --locked --offline --check
.venv/Scripts/python.exe -m pytest -p no:cacheprovider --tb=short
.venv/Scripts/python.exe scripts/evaluate_final.py preflight
```

The absolute Python/cache paths above record this machine's actual walkthrough, not portable setup requirements; use the version-based setup commands at the top for a new machine. The native/browser preview tool was stopped by its URL-confidence policy; no alternate UI route was used afterward. HTML generation alone is not a passed rendered-README inspection. External documentation/download URLs were retained from authoritative existing records but were not all HTTP-rechecked in this pass. A manual rendered-README review remains outstanding.

The walkthrough is a Windows check on this machine. It is not a fresh-host or macOS/Linux verification, and it does not download the dataset, restore private/local artifacts, retrain or execute test. A successful setup cannot erase the unpublished-artifact dependency of the exact scientific result.
