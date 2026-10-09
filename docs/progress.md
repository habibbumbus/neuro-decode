# Project status

## Package and tooling setup — October 4, 2026

The package and development environment are configured. No dataset has been
acquired and no decoder, training pipeline, scientific result, or latency
measurement is available yet.

### Implemented

- Installable `src/neurodecode` package with no work on import.
- Python 3.11 compatibility target, selected by `.python-version`.
- uv dependency management with generated `uv.lock` and an ignored `.venv`.
- pytest installation smoke test and Ruff lint/format configuration.
- Explicit source-distribution file selection and package-only wheel contents.
- Git exclusions for environments, secrets, recordings, and generated artifacts;
  provenance files and curated reports remain eligible for tracking.

### Verification

Verified on Windows 11 Home, 64-bit, Intel Core i9-13900H (14 cores / 20 logical
processors), 31.6 GiB RAM, with CPython 3.11.17 and uv 0.12.23:

- Locked environment sync and lockfile consistency checks passed.
- Package import and one pytest smoke test passed. The test imports the installed
  package in an isolated Python subprocess outside the checkout.
- Ruff lint and format checks passed.
- Source archive and wheel builds succeeded. Archive inspection confirmed package
  files are present and local files, environments, recordings, and runs are absent.
- Git whitespace and artifact-exclusion checks passed.

Resolved development dependencies include pytest 9.1.1 and Ruff 0.16.10.
macOS/Linux setup instructions are documented but have not been executed here.
Scientific dependency compatibility and decoding correctness remain unverified.

### Next milestone

Audit one public recording's source, licensing, checksums, schema, units, channel
metadata, and timing before implementing data preparation. Dataset selection is
provisional. Scientific dependencies and synthetic-data CI will be added with
components that need them. Project code licensing is undecided.

## Phase 0A: project definition — October 9, 2026

This entry supersedes the earlier next-milestone recommendation; the current
next increment is the Phase 0B review described below.

### Verified repository state

- Actual branch at this checkpoint: `setup/project-skeleton`.
- Last verified commit: `fb54cd3` (`initialize project tooling`), following
  initial commit `33184ad`.
- Origin is `https://github.com/habibbumbus/neuro-decode.git`. Local
  `origin/main` points to `33184ad`; the current branch is one commit ahead
  of that local reference and has no configured upstream. No fetch or remote
  inspection was performed, so current server state and prior push status are
  unverified.
- On entry, `README.md`, this progress file, and `pyproject.toml` already
  contained unstaged edits. The index was empty. Existing edits were preserved;
  `pyproject.toml` was not changed by this checkpoint.
- Tracked files confirm an existing package, lockfile, exclusions, and installation
  smoke test. Setup checks listed in the October 4 entry are historical recorded
  outcomes; they were not rerun today.
- No applicable `AGENTS.md` was found. There was no project brief or decision
  log. A separate decision log is not needed for this documentation slice.

### Work and review

Updated `README.md` with the experiment, planned evaluation, replay semantics,
and current status while retaining setup instructions. Added
[project_brief.md](project_brief.md) covering inputs, labels, velocity outputs,
chronological evaluation, planned models, visual replay, and limitations.
Appended this entry to preserve the earlier progress record.

Reviewed all three documents for agreement about implemented versus planned
work, held-out label use, timing, and replay claims. Inspected Git status,
branches, remotes, tracked files, recent history, and staged/unstaged diffs.
Verified the private local instruction file is untracked and excluded through
Git's local exclude file. Documentation whitespace review passed.

The shell tool could not start; local inspection used the JavaScript runtime.
Git initially rejected the runtime account's ownership mismatch; read-only
commands used a repository-specific, per-command safe-directory setting, with
no persistent configuration changes.

No dependencies were installed, data downloaded, models implemented, or tests
added/run during this documentation checkpoint. Scientific correctness and
current environment operability were not assessed. Documentation changes remain
uncommitted; no staging, commit, push, merge, or branch switch was performed.

### Concepts and understanding checks

Explained the tracker-controlled original cursor, concurrent intracortical spike
recordings, neural decoder inputs, tracker-derived labels, and velocity output.
Illustrated average velocity with a 2 mm change over 20 ms giving 100 mm/s.
Explained working tree, staging/index, local commit, and push to a remote.
The brief documents why chronological splits and training-only preprocessing
are planned and why visual replay must disclose integration and initialization.

Understanding check update: demonstrated that only staged file changes enter a
commit and that pushing publishes local committed history to the remote. After
clarification, correctly identified tracker-derived velocity as the reference
for held-out scoring rather than a decoder input.

Explained held-out evaluation, spike-count features versus raw signals, velocity
versus position, and label leakage versus overfitting. An independent explanation
of precise decoder inputs/outputs and leakage versus overfitting remains pending.
No broader mastery of Git, spike biology, array/schema handling, or model
mathematics is claimed by this entry.

### Decisions, open questions, and next increment

Preserve the existing package/tooling work and complete the missing public
project definition before scientific implementation. Keep current code licensing
and exact dataset/session selection open. Provisional timing and split settings
will need validation; no dataset-specific units or schema have been verified.

After the understanding check and user Git checkpoint, the next increment is a
Phase 0B review of the existing minimal environment/package setup against the
current plan. Verify what already works before adding anything; this entry does
not authorize new installation or data acquisition.

## Phase 0B: minimal package and environment — October 9, 2026

### Repository state and scope

- Actual branch: `main`, retained for direct checkpoint commits/pushes.
- Phase 0A is present in committed history: `0081f72` records progress and
  `3d9853e` adds the project overview/brief. Existing package/tooling was
  committed earlier as `fb54cd3`.
- `main` and local `origin/main` both point to `3d9853e`, recording the
  prior push. No new remote query/fetch was performed.
- Only an existing package-description edit in `pyproject.toml` was unstaged
  on entry. That edit was preserved. The index was empty.
- Read the local guidance and progress; no applicable `AGENTS.md` was found.
  The private instruction file remains excluded locally and untracked.

### Implemented versus retained

The minimal package, Hatchling build configuration, Python 3.11 compatibility,
`src/neurodecode/__init__.py`, `.python-version`, generated `uv.lock`, and
project-local `.venv` already existed and were retained. Runtime dependencies
remain empty. Existing pytest/Ruff development dependencies and the installation
smoke test were preserved; no scientific dependencies or new tests were added.

Updated `README.md` to explain the environment, package metadata, lockfile,
source layout, editable installation, and working uv executable commands. Added
`.uv-cache/` to shared `.gitignore` for the local verification cache. Existing
exclusions cover environments, Python/tool caches, secrets, raw/processed
recordings, runs, and model artifacts without excluding curated JSON/CSV/PNG.

The existing lockfile was validated using uv and was not hand-edited or
regenerated unnecessarily. Rebuilt the editable local package through
`uv sync --locked --cache-dir .uv-cache`. No dataset, model, UI, future module,
or CI was added.

### Environment and verification

Observed Windows NT build 26200, x64, and Windows PowerShell 5.1.26100.9549.
The runtime account's launcher exposes Python 3.14 with pip 25.2; the project
uses uv 0.12.23 and CPython 3.11.17 in `.venv`. An exact Windows edition query
through CIM was denied; the kernel build and shell version were obtained by
other read-only checks. Launcher inventory reflects this runtime account only.

Checks executed:

- `uv --version` confirmed 0.12.23. `py -m uv --version` failed because
  the launcher-selected interpreter could not find that module; the executable
  works and is now used in setup instructions.
- `uv lock --check --offline --cache-dir .uv-cache` passed.
- Initial offline sync failed because Hatchling was absent from this cache;
  the restricted online attempt was blocked by socket permissions. Approved
  execution then built and installed the local editable package successfully.
- `uv sync --locked --offline --check --cache-dir .uv-cache` passed and
  reported no environment changes needed after installation.
- `uv run --locked --offline --cache-dir .uv-cache python -I -c "import sys, neurodecode; print(sys.version); print(sys.executable); print(neurodecode.__file__)"`
  passed with the project environment's Python 3.11.17 and package source path.
- Direct isolated import using that environment's Python from a temporary
  directory passed; installed metadata confirmed an editable installation.
- The existing `tests/test_package.py` check passed (one test). Pytest emitted
  a cache-write permission warning under this runtime account; the import check
  succeeded. No cache deletion or permission changes were made.
- Git whitespace and exclusion checks passed; lockfile bytes were unchanged.
- Final status also showed an untracked `%SystemDrive%/` directory containing
  Windows cache database files. It was inspected without modification and is
  outside the intended staging list.

These checks establish local environment synchronization and package discovery,
including outside the checkout without manual path manipulation. They do not
establish data integrity, model correctness, scientific results, or latency.
A fresh checkout/macOS/Linux installation was not tested. Lint, formatting,
archive builds, and CI were not run/added during this increment.

### Concepts, pending checks, and next increment

Explained virtual environments, project/build metadata, dependency constraints
versus locked versions, the source layout, and editable installation. Demonstrated
that the project environment isolates dependencies and that an import check does
not establish decoder correctness. Clarified that importing verifies package
discovery/loading rather than every dependency behavior. After clarification,
correctly identified pyproject.toml as the place to declare dependency
requirements; uv.lock records exact resolved versions and is updated by uv.
An independent explanation of source layout and editable installation remains
pending; no broader packaging mastery is claimed.

Current Phase 0B edits are uncommitted. No staging, commit, push, merge, or branch
switch was performed by the assistant. The last verified relevant commit/push
state remains `3d9853e` on `main` and local `origin/main`.

After the understanding check and user Git checkpoint, the next increment is
Phase 0C: review the existing tooling and reproducible setup checks, adding only
what is useful and missing. CI, scientific dependency compatibility, source
verification, data acquisition, and model implementation remain pending.
