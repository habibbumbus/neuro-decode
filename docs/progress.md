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
