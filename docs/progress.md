# Progress

## Phase 0: package and tooling setup — October 4, 2026

Implementation and setup checks are complete. Understanding checkpoint and user
commit are pending. No dataset, model, training, or performance measurement is
present.

### Scope and decisions

- Create an installable `src/neurodecode` package with no work on import.
- Target Python 3.11 only for now; broader version support is unverified.
- Use uv and a generated lockfile as the project's dependency-resolution method.
- Add pytest and Ruff now; defer scientific dependencies until needed.
- Explicitly include package/project files in source distributions; restrict
  wheels to the package.
- Keep data, generated artifacts, secrets, and environments out of Git.
- Defer CI until scientific components have meaningful synthetic-data tests.
- Project code license is pending the author's choice.

### Initial inspection

- Repository root and origin match this project; branch is
  `setup/project-skeleton`, based on initial commit `33184ad`.
- Working tree and index were clean before setup edits; only README was tracked.
- No applicable `AGENTS.md` files were found.
- Local implementation brief is excluded through `.git/info/exclude` and is
  untracked. Local exclusions must be re-established in another checkout.
- Windows 11 Home, 64-bit; Intel Core i9-13900H, 14 cores / 20 logical processors;
  31.6 GiB RAM. CPU is the planned benchmark device; no accelerator was assessed.
- Existing Python installations: 3.12, 3.13, and 3.14. Python 3.11 and uv were
  initially unavailable.

### Verification

Checks actually run on Windows:

- Installed uv 0.12.23 into the existing launcher Python's user packages and
  installed uv-managed CPython 3.11.17. Existing Python versions were retained.
- `py -m uv sync` generated `uv.lock` and created the ignored `.venv`.
  A subsequent `py -m uv sync --locked` succeeded without changing the lockfile.
- `py -m uv run --locked python --version`: Python 3.11.17.
- Package import command printed `neurodecode`.
- `py -m uv run --locked pytest`: one test passed. The test imports the installed
  package in an isolated subprocess outside the checkout.
- Ruff lint and format checks passed; resolved versions are pytest 9.1.1 and
  Ruff 0.16.10, recorded with their dependencies in `uv.lock`.
- `py -m uv build` produced a source archive and a wheel under ignored `dist/`.
  Archive inspection confirmed the package is present and the local brief,
  environments, secrets, recordings, and generated runs are absent.
- Git exclusion checks confirmed the local brief is excluded and untracked;
  environments, recording paths, runs, checkpoints, and `.env` are ignored.
  A manifest JSON, curated PNG, and report CSV remain eligible for tracking.
- `git diff --check` passed. No staging, commits, pushes, or merges were performed.

These checks establish basic installation/tooling only. macOS/Linux commands are
documented but have not been executed here. Scientific dependency compatibility,
data integrity, causality, decoding performance, and inference latency are
unverified. No installation or check was blocked.

### Learning checkpoint

Understanding has not been assessed. Pending concepts: working tree versus index
versus commit; local exclusions versus tracking; package/environment smoke checks
versus scientific validation.

Next step after teach-back and user Git checkpoint: Phase 1 data-source audit,
provenance, and one verified recording. Dataset selection remains provisional.
