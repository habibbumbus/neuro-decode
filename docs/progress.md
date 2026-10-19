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

## Phase 0C: development and setup checks — October 9, 2026

### Verified state and retained implementation

- Branch: `main`. Phase 0B progress is committed as `8285520`; setup changes
  are committed as `98011df` (`init python package and environment`).
- `main` and local `origin/main` both point to `98011df`, recording the
  prior push. No new remote fetch/query was performed.
- The index and tracked working files were clean on entry. The previously
  observed untracked `%SystemDrive%/` Windows cache directory was preserved.
- Read all current local guidance and progress. No applicable `AGENTS.md` or
  decision log was present. The private guidance remains locally excluded and
  untracked.
- Existing Ruff configuration targets Python 3.11, uses an 88-character line
  length, and enables `E4`, `E7`, `E9`, `F`, and `I` rules. The existing
  lockfile selects Ruff 0.16.10 and pytest 9.1.1. No new dependency, configuration,
  test, or scientific source code was needed.

### Changes and checks executed

Updated `README.md` with normal development commands, cache-free variants,
check meanings, and an optional PowerShell temporary-environment verification.
Appended this progress entry. Existing package/source/configuration and lockfile
were retained. No CI, data download, model, CLI, or future scaffolding was added.

Verification used Windows x64 build 26200, CPython 3.11.17, and uv 0.12.23:

1. Created a new temporary directory with a separate `venv`. Set
   `UV_PROJECT_ENVIRONMENT` only for the installation process. Attempted
   `uv --cache-dir .uv-cache sync --locked --offline --python .venv/Scripts/python.exe`;
   this could not install locked Ruff because its wheel was absent from the cache.
2. Approved online execution of
   `uv sync --locked --cache-dir .uv-cache --python .venv/Scripts/python.exe`
   with the temporary environment selected succeeded, installing all eight
   locked packages. No working-environment sync was performed.
3. From a separate directory outside the checkout, ran the temporary
   environment's Python with `-I -c` to import NeuroDecode and print interpreter,
   module, and editable-install metadata. It passed: Python came from the temporary
   environment, NeuroDecode came from `src/neurodecode/__init__.py`, and metadata
   recorded an editable install. No manual `sys.path` or `PYTHONPATH` changes
   were made.
4. `uv --cache-dir .uv-cache --offline lock --check` passed.
5. `uv --cache-dir .uv-cache --offline run --locked --no-sync ruff check --no-cache .`
   passed with no lint findings.
6. `uv --cache-dir .uv-cache --offline run --locked --no-sync ruff format --check --no-cache .`
   passed with no formatting changes required.
7. `uv --cache-dir .uv-cache --offline run --locked --no-sync pytest -p no:cacheprovider tests/test_package.py`
   passed: one existing installation test, no warnings. Disabling test caching
   avoids the runtime account's previously observed cache-write restriction.
8. Git whitespace review passed. Lockfile, package/configuration, and test bytes
   were unchanged. The index remains untouched.

The unified shell tool failed before starting a command; local inspection and
checks used the JavaScript runtime, with approved shell execution for the
network-enabled temporary installation. Read-only Git inspection used a
per-command repository-specific safe-directory setting, without persistent
configuration changes. The temporary environment remains outside the repository
for inspection; the existing project `.venv` was not reinstalled or replaced.

### Meaning, limitations, and learning status

The installation smoke test confirms basic package loading from outside the
checkout. Ruff lint checks declared source rules; format checks style. Lock
validation and temporary installation verify that this machine can construct a
new environment using the current project metadata and selected dependencies.
They do not establish scientific behavior, data integrity, decoding quality,
full cross-platform reproducibility, or inference latency. A fresh clone and
macOS/Linux installation were not tested; offline setup requires cached artifacts.

Explained smoke testing, linting, formatting, and reproducible environment setup.
Demonstrated the distinction between a basic import smoke test and static lint
checks for issues such as unused imports and undefined names. Identified the
outside-checkout check as supporting portability; clarified that its specific
evidence is installed package discovery without relying on the current directory,
not portability to another machine or OS. Existing pytest is retained for its
installation test; no placeholder assertions or scientific tests were added.
CI remains deferred until meaningful component behavior exists.

Current README/progress changes are uncommitted. The last verified commit and
recorded push remain `98011df` on `main` and local `origin/main`. The assistant
performed no staging, commit, push, merge, or branch switch.

After the understanding check and user Git checkpoint, the next increment is
**Phase 1A: verify the dataset source and acquire one recording**. Resolve exact
source/version, file, license/attribution, size, and checksum, then verify the
acquired bytes and record provenance. No acquisition is authorized by this
Phase 0C checkpoint.

## Phase 1A: source verification and one recording — October 9, 2026

### Verified checkpoint state

Branch is `main`. Phase 0C is committed as `93f1abb` (progress) and
`aac6d3a` (development/setup documentation). Local `main` and `origin/main`
both point to `aac6d3a`, recording the prior push; no new fetch/remote query
was performed. Tracked working files and the index were clean on entry. Preserved
the existing untracked `%SystemDrive%/` Windows cache directory.

Read the complete local guidance, progress, README, and relevant setup files.
No applicable `AGENTS.md` was found. The private guidance remains excluded
through Git's local exclude file and untracked. Project Python 3.11.17 still
imports NeuroDecode, and uv 0.12.23 is available. Existing Phase 0C test/lint
outcomes were read rather than rerun for this acquisition/documentation slice.

### Acquired and verified

Retained the requested [Zenodo record 583331](https://zenodo.org/records/583331),
explicitly labeled v1 on its page, rather than following its newer-version link.
The [official API](https://zenodo.org/api/records/583331) confirmed the candidate
filename, 83,957,398 bytes, published MD5, creators, publication date, and
CC BY 4.0 license. The API's explicit version string was absent; version identity
was checked on the record page.

No existing recording was present. Downloaded only `indy_20161005_06.mat` into
ignored `data/raw/`, using the version-specific URL and a temporary .part
filename. Verified size and publisher MD5 before renaming to the final filename:

- Actual size: **83,957,398 bytes**, matching the source metadata.
- Published and local MD5: `5ea300952642e0fc54245144499db9bb`.
- Local SHA-256: `4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac`.
- Acquisition date: **2026-10-09 UTC**.

Created `data/manifest.json` with record/version, DOI, stable URLs, relative
filename/path, byte size, source/local checksums, license, citation, and date.
Created [dataset.md](dataset.md) with experiment, source expectations, acquisition
and verification commands, and limitations. Updated README and project-brief
status to reflect acquisition while leaving internal inspection pending.

### Commands and outcomes

- Read-only Git status, history, tracking comparison, remotes, and private-file
  exclusion/tracking checks passed.
- Initial standard-library API request was blocked by runtime socket permissions
  (WinError 10013). Approved `Invoke-RestMethod` calls to the exact official
  record API succeeded. The browser's API fetch failed, but the official record
  page and the approved local API request provided the required evidence.
- `curl.exe --fail --location --retry 2 --connect-timeout 30 --max-time 300`
  fetched only the selected file URL to a .part file. PowerShell
  `Get-Item` and `Get-FileHash -Algorithm MD5`/`SHA256` verified it;
  `Move-Item` renamed only after size and source MD5 matched.
- Independently recomputed byte size, MD5, and SHA-256 on the final filename
  using project Python and its standard library; all values agreed.
- Reused the verified file when checking the dataset card's PowerShell
  reproduction commands; no second download was performed.
- Manifest JSON parsing, documentation consistency/whitespace review, and Git
  raw/processed-data exclusion checks passed. Manifest and dataset card remain
  trackable; recording and private guidance remain untracked.

No scientific dependency, loader, processing, model, test, CI, or demo was added.
The source file was not parsed internally. Hash agreement establishes byte
identity, not scientific validity. Storage format/shapes, channel count/regions,
unit count, duration, missingness, timestamp origins, coverage, alignment, and
actual suitability remain pending Phase 1B. The source-documented fields/units
are recorded explicitly as expectations, not local observations.

### Concepts, learning status, and next increment

Explained recording sessions, provenance, dataset licensing, checksums, and the
paired experiment: implanted electrodes recorded spikes concurrently with hand
movement; the tracker determined cursor position; decoding is learned afterward.
The selected source is public nonhuman-primate data, with no human/live-control
performance claim. Dataset licensing remains separate from project code licensing.

Understanding check update — October 10, 2026: identified neural recordings as
the neural activity measurements and tracker recordings as the paired movement
reference. Clarified that the recorded inputs are spike-event timestamps and
that cursor positions will later supply derived velocity labels. Correctly
explained checksum matching as verification of the expected file bytes. After
clarification, correctly chose the manifest-recorded dataset version for
reproduction rather than automatically using a newer release. Exact version
identity and byte verification serve distinct purposes; neither proves scientific
suitability. No broader schema or neuroscience mastery is claimed.

Acquisition is locally verified; commit/push status has not been rechecked during
this understanding update.
The last verified commit/recorded push remains `aac6d3a` on main/origin/main.
The assistant performed no staging, commit, push, merge, or branch switch.

After the user understanding/Git checkpoint, the next increment is **Phase 1B:
inspect the actual file structure and build a minimal validated loader**.

## Phase 1B: inspect and load the recording — October 10, 2026

### Verified state and Stage 1 inspection

Actual branch: `main`. Phase 1A is committed as `3797161` (progress) and
`6656b14` (provenance/acquisition documents). Main and local origin/main both
point to `6656b14`, recording the prior push; no new fetch/remote query was
performed. Tracked working files and the index were clean. Preserved the existing
untracked Windows cache directory. Read the complete local guidance, progress,
dataset card, manifest, and package configuration; no applicable AGENTS.md was
found. The private guidance and raw recordings remain excluded and untracked.

Used the manifest's selected `indy_20161005_06.mat`, Zenodo record 583331 v1.
Rechecked 83,957,398 bytes and SHA-256 against the manifest: matched. The header
identifies MATLAB 7.3; h5py confirms HDF5. Inspected top-level metadata and
references without reading waveform payloads. Added only h5py and NumPy using
`uv add "h5py>=3.11,<4" "numpy>=1.26,<3" --cache-dir .uv-cache` through approved
network-enabled execution. uv generated the lockfile changes and installed
h5py 3.16.0 and NumPy 2.4.6 in the project environment.

Observed behavior: 93,501 x/y samples, timestamps 1288–1662 s, span 374 s;
mean interval 0.004 s (250 Hz), with floating-point variation only. No nonfinite
behavioral values, duplicate/decreasing behavior times, or intervals above
1.5 × the median were observed. This threshold is a reporting heuristic, not a
scientific exclusion policy. Stored time shape is (1, 93501); stored positions
are (2, 93501). Source schema and matched sample counts establish the returned
sample-major position orientation (93501, 2).

Channels are M1 001–096 in source order. Stored spike references have shape
(5, 96), establishing five unit slots per channel. There are 323,237 spike events
and 231 encoded empty trains; eleven channels have all five slots empty. Slot
totals are 128,756 / 84,879 / 81,701 / 23,654 / 4,247. No nonfinite or decreasing
spike times or duplicate within-train timestamps were observed. Spike coverage
is approximately 1283.002344–1666.995917 s, including 5,454 events before behavior
and 5,364 afterward. These events and empty channels are preserved.

Units (seconds, cursor millimeters) come from source documentation, not numeric
magnitudes; relevant datasets lack explicit unit attributes. The observed
sampling interval agrees with the documented rate. An optional finger/cursor
consistency comparison found a maximum 3.038031 mm discrepancy from the source's
simple mapping, with varying per-axis residuals. Its cause is unresolved; it is
not repaired or explained as filtering/lag without evidence. The stored cursor
positions remain authoritative inputs to this loader, with caveats in dataset.md.

### Stage 2 implementation

- `src/neurodecode/data/__init__.py`: needed data subpackage identity.
- `src/neurodecode/data/loaders.py`: source-specific HDF5 loader, named raw
  `Recording`, schema/value/order validation, and timing/coverage diagnostics.
  Returned fields are behavior_times_s [T], cursor_position_mm [T, 2], ordered
  channel_ids [C], and spike_times_s [C][U] with variable-length event arrays.
- `scripts/inspect_recording.py`: load one path and print a concise summary;
  no general CLI framework.
- `tests/test_loaders.py`: tiny generated synthetic reference fixtures and
  nineteen meaningful extraction/validation cases. No real data in tests.
- `pyproject.toml` and generated `uv.lock`: only the required reader/array
  dependencies. Existing development tools retained.
- `docs/dataset.md`: observed schema/orientation, timing, units provenance,
  occupancy, coverage, caveats, actual returned-value example, and run commands.
- `data/manifest.json`: inspection status/date updated; acquisition identity,
  source bytes and checksums unchanged.
- README/project brief: accurate loader status and reproduction instructions.

MATLAB cells are dereferenced explicitly. The empty [0, 0] dimension encoding is
recognized as zero events, not timestamps. Null references fail distinctly.
Nonfinite values, mismatched dimensions, duplicate/decreasing behavior times,
decreasing spike times, and unsupported structures fail clearly. Legitimate
empty trains, duplicate spike events, gaps, and coverage differences are preserved
and reported. The loader never sorts, truncates, repairs, bins, aggregates units
into model features, normalizes, derives velocity, or creates evaluation splits.

### Verification and limits

Commands actually run (repository root):

```text
uv --cache-dir .uv-cache run --locked --no-sync python scripts/inspect_recording.py data/raw/indy_20161005_06.mat
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider
uv --cache-dir .uv-cache run --locked --no-sync ruff check --no-cache .
uv --cache-dir .uv-cache run --locked --no-sync ruff format --check --no-cache .
uv --cache-dir .uv-cache lock --check --offline
```

The real-file summary agreed with the independent inspection. All **20 tests
passed**: nineteen synthetic loader cases plus the existing installation check.
Tests establish mapping/orientation, explicit empty handling, gap/coverage and
spike-duplicate preservation, and rejection of malformed timestamps, positions,
references, channels, empty encodings, and unsupported formats. A deliberately
unreadable waveform link confirms the loader does not access waveform data.

Lint passed. Initial format checks requested formatting of the two new Python
files; Ruff formatted only those files and the final format check passed.
Lockfile consistency and Git whitespace/exclusion review passed. Synthetic
checks do not establish biological alignment or decoding performance. Real-file
validation establishes this supported schema and the reported observations,
not suitability of every session/format. Earlier MAT files and alternative
layouts fail rather than being guessed. No tests were blocked; local runtime
cache writes were disabled to avoid previously observed permission restrictions.
CI, models, data preparation, and demo remain deferred.

### Concepts and next increment

Explained loader responsibilities, array shapes, sampling intervals, variable
spike-train lengths, cell arrays/references, empty trains versus missing data,
source-based units, and channel/unit identity. For example, time 1288.0 s maps to
cursor [-7.566217, 30.195105] mm, while channel M1 001's first listed unit events
precede that sample; event rows are not movement sample rows. After clarification,
correctly interpreted shape (100, 2) as 100 x/y movement samples and an empty unit
slot as zero recorded spike events rather than a missing position sample.
Explained that regularly sampled movement and irregular spike events are not
paired row-by-row, and that five unit slots do not imply five neurons: the first
slot contains residual unsorted events and other slots may be empty. Independent
explanations of event timing and channel/unit identity remain pending; no broader
neuroscience or loader mastery is claimed.

Implementation/checks are complete; changes remain uncommitted. The last verified
commit/recorded push remains `6656b14` on main/origin/main. The assistant
performed no staging, commit, push, merge, or branch switch.

After the user understanding/Git checkpoint, the next increment is **Phase 1C:
visualize a short segment of neural activity alongside recorded movement**.

## Phase 1C: recorded spikes and cursor plots — October 10, 2026

### Verified starting state

Actual branch: `main`, at `f44b792` (`load and val neural recording data`).
Local origin/main matches that commit, recording the previous push; no new
fetch/remote query was performed. The tracked worktree and index were clean.
Preserved the unrelated untracked Windows cache directory. Read the complete
local guidance, progress, dataset card, manifest, loader, and existing tests.
No applicable AGENTS.md was found. Private guidance and recordings remain
excluded and untracked.

Used the manifest's Indy recording from Zenodo 583331 v1. The plot script checks
its SHA-256 before calling the existing loader. The loader interface and source
file remain unchanged; there is no second parser. Added only Matplotlib via
`uv add "matplotlib>=3.8,<4" --cache-dir .uv-cache` with approved network access.
uv updated dependencies and the lockfile, selecting Matplotlib 3.11.2 and its
required dependencies. No modeling stack was added.

### Implementation and selection decisions

- `src/neurodecode/plotting.py`: fixed early interval selection, shared time
  conversion, labeled train selection, and three-panel recorded-data rendering.
- `scripts/plot_recording.py`: manifest-driven generation command; verifies
  bytes, writes PNG/JSON under ignored runs/, and refuses to overwrite artifacts.
  Optional single folder name permits reruns without an elaborate CLI.
- `tests/test_plotting.py`: nine synthetic selection/time-conversion cases.
- `pyproject.toml` and generated `uv.lock`: plotting dependency.
- Dataset card and README: command, rule, observations, interpretation, limits.
- This progress entry: evidence and the next increment.

The rule selects the earliest ten seconds in global neural/behavior overlap,
starting on a behavior sample, and requires its end inside the provisional first
70% of behavioral elapsed time. This is an exploratory guard, not generated or
finalized evaluation splits. Selected interval: **[1288.0, 1298.0) s**, before
provisional training end 1549.8 s. Only this early interval was presented; no
future held-out portion was visualized for presentation selection.

First four lexical channel identifiers and their first two unit slots yield
M1 001–004 / U1–U2, independent of movement quality or apparent correlation.
All eight rows remain, including two with zero events. U1 is residual unsorted
activity; U2 is a sorted slot, not proof of a distinct stable neuron. Interval
counts in row order: 59, 0, 43, 55, 48, 47, 18, 0 (270 total events).

Both temporal panels subtract exactly the same 1288.0 s offset. The trajectory
uses the identical 2,500 movement samples, with equal millimeter axis scaling.
Start/end markers indicate first and last included samples; the last is
1297.996 s, not the excluded endpoint. No spike binning, derived velocity,
normalization, feature engineering, model fitting/prediction, signal shift,
smoothing, or extra interpolated samples were introduced.

Gap policy: reject any loader-defined behavioral gap intersecting the fixed
interval; do not search for a replacement or bridge it. No gap is present here.
Absence of spike events cannot distinguish silence from missing packets. The
Phase 1B finger/cursor discrepancy remains unresolved; plots do not repair it.

### Checks and actual outcomes

```text
uv --cache-dir .uv-cache run --locked --no-sync python scripts/plot_recording.py
uv --cache-dir .uv-cache run --locked --no-sync python scripts/plot_recording.py phase1c-verified
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider
uv --cache-dir .uv-cache run --locked --no-sync ruff check --no-cache .
uv --cache-dir .uv-cache run --locked --no-sync ruff format --check --no-cache .
uv --cache-dir .uv-cache lock --check --offline
```

Initial rendering succeeded but reported a font-cache permission warning.
The script now defaults that cache to ignored runs/.matplotlib/; rendering into
a new folder succeeded without warnings, preserving the initial output. The
final reviewed artifact is **runs/phase1c-verified/indy_20161005_06_recorded.png**
with same-stem JSON. Both renders contain the same plotted pixels. Visual
inspection confirmed readable train labels/empty-row annotations, matching time
axes, physical units, equal trajectory scaling, and visible first/last markers.
No pixel-perfect plot tests were added.

All **29 tests passed**: 20 existing checks plus nine new synthetic cases for
half-open bounds, common offsets, channel/unit mapping and empties, movement-
independent selection, overlap-start snapping, gap refusal, provisional training
limits, absent coverage, and invalid duration. Lint, format, lock consistency,
Git whitespace/privacy/exclusion checks passed. Formatted only the three new
Python files. Loader bytes were verified unchanged. PNG/JSON/cache outputs stay
ignored; nothing is staged. No checks remain blocked or visually pending.

Recorded x/y positions vary across the clip and events appear in multiple
selected trains. This demonstrates a readable paired-data view, not causation,
precise physiological alignment, or decoding performance. Underlying units remain
source-documented as in Phase 1B. No data download or CI work was performed.

### Learning checkpoint and next increment

Explained spike rasters (ticks at event times, identified train rows), position
versus time, and equal-scale x-y trajectories. Explained identity-based selection,
a common display time origin, explicit empty rows, and the distinction between
co-occurrence and model evidence. Correctly identified raster ticks as spike
events and the figure as recorded activity/movement rather than decoder evidence.
After clarification, correctly explained that a shared time-origin subtraction
changes displayed timestamps while preserving relative timing. Independent
explanations of identity-based selection and equal spatial scaling remain pending;
no broader plotting or neuroscience mastery is claimed.

Implementation and verification are complete; changes remain uncommitted. The
last verified commit/recorded push remains f44b792 on main/origin/main. The
assistant performed no staging, commit, push, merge, or branch switch.

The next increment will **establish chronological split boundaries and the causal
timing contract before generating spike-count features or velocity targets**.

## Phase 2A: chronological splits and causal timing — October 10, 2026

### Verified starting state

Branch main at ead8c2c (phase 1c), following 2fa20b6 (visualization implementation).
Local origin/main matches ead8c2c, recording the prior push; no new remote
query/fetch was performed. Phase 1C is present in main's history, not assumed
merged from another branch. Tracked files and the index were clean on entry.
The unrelated untracked Windows cache directory was preserved. Read complete
local guidance, progress, dataset card, manifest, loader, plotting code, scripts,
and existing tests. No applicable AGENTS.md was found. Private guidance remains
locally excluded and untracked; raw recordings remain ignored and untracked.

Reused manifest-selected indy_20161005_06.mat, Zenodo 583331 v1. SHA-256 still
matches 4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac;
actual size remains 83,957,398 bytes. Loader timing audit confirms 93,501 behavior
samples spanning 1288–1662 s, approximately 4 ms intervals, and no detected gaps.
Global neural coverage extends beyond behavior on both sides. No waveform
payloads, held-out movement patterns, or correlations were examined to choose
boundaries. Existing loader/visualization source and interfaces were retained.

### Implemented and decisions

- configs/base.toml: 20 ms bins, ten-bin history, no additional lag/horizon,
  fixed 6 ms behavioral sample-age/gap limits, elapsed-time fractions, and the
  previously explored interval ledger.
- src/neurodecode/splits.py: validated timing configuration, half-open interval
  ownership, conservative support calculations, past-only behavioral timestamp
  selection, chronological planning, gap segmentation, purge and warm-up metadata.
  It does not read position values, bin events, or calculate velocity.
- scripts/plan_splits.py: verifies acquired SHA-256, reuses the loader, checks
  exploration/training ownership, saves source-bound specification, prints summary,
  and refuses to replace an existing different specification.
- data/splits/indy_20161005_06.json: generated small trackable split specification,
  recording source/version/checksum and config fingerprint, grid, bounds,
  endpoint conventions, purge/gap/reset rules, and null eligible feature indices.
- tests/test_splits.py: 26 synthetic timestamp/config/support cases.
- docs/methods.md: timing, label definition, exact sample-time example, rationale,
  endpoint/gap/state rules, reproducible commands, and limitations.
- docs/decisions.md: substantive timing/evaluation policy choices.
- README.md and docs/dataset.md: accurate status and links to the split contract.
- This entry: verified results, pending learning checks, and next increment.

Nominal 70/15/15 applies to elapsed overlap time, including any gaps:
[1288, 1549.8), [1549.8, 1605.9), [1605.9, 1662).
The conservative lookback is max(200 ms neural history, 26 ms target-source
lookback plus 1 ns age tolerance) = 200 ms. Purge 200 ms on each side of each
boundary. Retained intervals, rounded for display:
training [1288, 1549.6), validation [1550, 1605.7), test [1606.1, 1662).
Retained durations: 261.6 / 55.7 / 55.9 s; purge discard 0.8 s total; additional
detected-gap discard 0 s. Machine endpoints in JSON are authoritative.

Full warm-up is separately required inside each retained continuous segment:
200 ms here. Purge time belongs to no partition; warm-up time supplies local
history. All source support and actual selected samples must belong to the same
partition and segment. Future model state resets at session/partition/gap starts.
No held-out movement may initialize a decoder. Fixed >6 ms behavioral intervals
(+1 ns comparison tolerance) split continuity; never bridge or repair gaps.
The tolerance does not permit future sample selection or expanded ownership.
Short fragments cannot form full windows; insufficient partitions fail.
Global spike silence does not identify neural acquisition outages.

Phase 1C artifact metadata confirms explored interval [1288, 1298) and the same
source SHA-256. It is entirely inside retained training: no conflict.
For a possible emission at 1288.200 s, features cover (1288.000, 1288.200];
actual selected position-source timestamps are 1288.180 and 1288.200 s
(indices 45 and 50). The future completed-bin label uses their displacement
divided by 0.020 s; no target value was calculated. This is not forecasting.

### Commands and outcomes

Commands actually run from the repository root:

~~~text
uv --cache-dir .uv-cache run --locked --no-sync python scripts/inspect_recording.py data/raw/indy_20161005_06.mat
uv --cache-dir .uv-cache run --locked --no-sync python scripts/plan_splits.py
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider
uv --cache-dir .uv-cache run --locked --no-sync ruff check --no-cache .
uv --cache-dir .uv-cache run --locked --no-sync ruff format --check --no-cache .
uv --cache-dir .uv-cache lock --check --offline
git diff --check
~~~

The summary and generated specification agree. An identical generation rerun
succeeded. An additional one-off project-Python check used prediction_support
and select_behavior_samples at 1288.200 s; it printed the support bounds and
actual indices/timestamps reported above without reading position values.

All 55 tests passed: 29 retained checks and 26 new synthetic cases. Coverage
includes ordering/disjointness, time proportions, half-open endpoints, support
bounds and age tolerance, past-only selection, stale/cross-boundary refusal,
gap segmentation/reset warm-up, absent/disjoint neural coverage, impossible
settings, and insufficient continuous recording length. These are synthetic
test results, distinct from the real recording's timing observations.

Initial lint found two import-order issues in new files; fixed those only.
Ruff formatted only the three new Python files; final lint and format passed.
Lock consistency and whitespace review passed. Git emitted its existing
LF-to-CRLF normalization notice, not a whitespace failure. Config hashing
normalizes text to LF for portability; raw recording hashing uses exact bytes.
Privacy/raw/processed/runs exclusion checks passed; config, split JSON and public
docs remain trackable. No checks remain blocked. The local environment was not
reinstalled; no dependency or lockfile changes were needed.

These checks establish planning/support behavior for tested synthetic timelines,
and source identity/recording metadata for this file. They do not establish
decoding accuracy, physiological lag, perfect acquisition, future bin/target
correctness, or final eligible example counts. Exact grid eligibility remains
pending construction. No scientific transforms, models, data download, CI,
or demo were added.

### Concepts, pending understanding, and next increment

Discussed chronological splitting, leakage from shared support or learned
preprocessing, neural history windows, past-or-present behavioral sample selection,
completed-interval estimation versus forecasting, purge versus local warm-up,
and state resets. Understanding update: correctly distinguished permanently
excluded purge time from within-partition warm-up history. After clarification,
correctly selected a 4 ms-old behavioral sample over a future sample; the same
partition/segment requirement was also explained. Identified the need to keep
training and validation separate. Clarified that validation selects model
settings and the final test evaluates frozen choices. An independent explanation
of that distinction and why borrowing training history violates validation
support remains pending. No broader timing or evaluation mastery is claimed.
Earlier schema and plotting learning status remains as recorded above; no
new mastery is inferred from passing software checks.

Phase 2A implementation/verification is complete and committed as 25ebf63
(define chronological splits, timing rules). Main and local origin/main match
that commit, recording the prior push; no new remote query/fetch was performed.
Tracked files and the index were clean before this learning-status update;
the unrelated untracked Windows cache directory was preserved. This progress
update is uncommitted. The assistant performed no staging, commit, push, merge,
or branch switch.
After the understanding and user Git checkpoint, the next increment is
**Phase 2B: turn spike timestamps into causal spike-count bins using this contract**.

## Phase 2B: causal spike-count bins — October 10, 2026

### Verified starting state and contract

Actual branch main at 96e40d0 (phase 2a), following 25ebf63 (split/timing
implementation). Main and local origin/main match, recording the prior push;
no new remote query/fetch was performed. Tracked files and the index were clean.
The prior learning-status update is now committed. Preserved the unrelated
untracked Windows cache directory. Read complete local guidance, progress,
dataset card, methods, decisions, manifest, actual configuration/specification,
loader and planner. No applicable AGENTS.md was found. Private guidance remains
locally excluded and untracked; recordings and processed data remain ignored.

Reused verified Indy session indy_20161005_06, Zenodo 583331 v1. The command
checks source SHA-256, exact source identity, config UTF-8/LF fingerprint, and
recomputed timestamp-only split metadata before counting. Source SHA-256 remains
4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac.
Config/split specification, raw data, loader/planner, dependencies and lockfile
were retained without changes. No boundary proportions or timing settings changed.

The official record's source description confirms that U1 contains residual
unsorted threshold crossings left after sorting other units, not all events
duplicated from those units. Direct browser retrieval timed out; the search-
indexed official record provided the same documented role. This is source
documentation, not independent proof of biological unit identity.

### Implementation and policies

- src/neurodecode/data/binning.py: BinnedSpikes and importable segment-grid/
  partition counting logic with timestamp, coverage, identity, and gap guards.
- scripts/bin_recording.py: verified training-only reproduction command,
  independent per-channel event conservation, and a fixed early raw/count example.
- tests/test_binning.py: 22 visibly synthetic cases; no real recording fixtures.
- docs/methods.md: exact grid/edge, aggregation, gap, provenance, and cache policy.
- README.md and docs/dataset.md: current status, output shape and reproduction.
- This progress entry: actual evidence, concepts, pending checks, next increment.

Grid origin is the saved 1288 s; width is 20 ms. Every edge is computed from
origin plus an integer index times width, with shared adjacent edges and no
cumulative addition. Count events in (start, end] using right-sided searches.
There is no spike-time rounding/epsilon or boundary snapping.

Construct complete bins separately inside retained partitions and valid segments,
requiring start >= segment start and end < segment end. Drop partial edge bins
and full bins ending at excluded endpoints. Machine endpoints remain authoritative,
including their floating-point tails; marginal bins can be conservatively dropped.
Return per-row partition/segment identity and segment edge-exclusion metadata.
Known gaps/purges have no rows; valid zero-spike bins remain zero-valued rows.
Refuse stale segments crossing >6 ms (+1 ns) behavioral gaps. Do not infer neural
packet loss from silence. Warm-up count bins remain for later history formation;
no model-ready history or scoring eligibility is claimed.

Select all M1-prefixed channel identifiers in lexical order; if no M1 identifiers
exist, retain all identifiers lexically with an explicit reported fallback.
Aggregate every available unit slot once per channel, including residual
unsorted U1 and empty slots, with zero-based slot mapping aligned to channel IDs.
Keep quiet channels; no activity-based filtering or single-neuron claim.

Counts stay in memory: no processed-array cache is saved or reused. Future cache
persistence/identifier validation is deferred. No bulk arrays or recordings enter
Git. The script prints source/config/split fingerprints and effective policy.

### Actual training observations and checks

Training output is int64 [13079, 96]: 13,079 complete 20 ms bins and 96 channel
columns, M1 001–096. All five source slots are included per channel.
There is one valid segment, head exclusion 0 s and tail exclusion approximately
20 ms. First bin (1288.000, 1288.020]; last (1549.560, 1549.580].
The bin ending at 1549.600 is excluded by the saved half-open ownership rule.

First three training counts for M1 001 are 0, 2, 1. The second bin contains
U1 event 1288.033454282609 s and U3 event 1288.0246070068824 s: count 2.
The third contains U3 event 1288.0577023718893 s: count 1. Presentation uses the
first ten bins/first three lexical channels, never a search for favorable activity.
Independent masks over original event lists match the count sums for every
selected channel, totaling 219,913 included training events. This is not a
whole-recording count. Real validation/test counts/patterns were not inspected;
partition behavior was verified with synthetic data.

Commands actually run from the repository root:

~~~text
uv --version
uv --cache-dir .uv-cache run --locked --no-sync python -c "import sys, numpy; print(sys.version); print(numpy.__version__)"
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider tests/test_binning.py
uv --cache-dir .uv-cache run --locked --no-sync python scripts/bin_recording.py
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider
uv --cache-dir .uv-cache run --locked --no-sync ruff check --no-cache .
uv --cache-dir .uv-cache run --locked --no-sync ruff format --check --no-cache .
uv --cache-dir .uv-cache lock --check --offline
git diff --check
~~~

Observed uv 0.12.23, Python 3.11.17, NumPy 2.4.6. No environment reinstall or
dependency addition was needed. Formatted only the three new Python files.
Focused initial tests passed; final full suite passed: 77 tests, including
55 retained checks and 22 new binning cases. Lint, format, lock consistency,
whitespace and exclusion reviews passed. The final training command passed
again after the ownership guard/summary refinement. No computational check
remains blocked. Git's existing LF-to-CRLF notices are normalization notices,
not whitespace failures.

Synthetic evidence covers hand counts/shared-edge ownership, duplicate list
entries, empty trains and genuine zero rows, multi-unit aggregation, identity
order/fallback, conservation over contiguous/disjoint intervals, future-event
mutation, one-float-step future exclusion, drift-free grid construction, partial/
excluded edges, gap omission/refusal, all partition boundaries, binless fragments,
overlap rejection, and concrete invalid inputs. Actual recording evidence is
separate: verified source/config/plan identity, training shape/intervals, original
event example and per-channel conservation. Neither establishes physiological
alignment, uninterrupted acquisition, decoding accuracy, or future target correctness.

### Concepts, learning status, and next increment

Explained irregular event timestamps versus counts, 20 ms width, right-closed
event intervals, raw units versus channel aggregates, quiet versus invalid bins,
edge omission, and retaining warm-up source counts. Hand example events
[1.000, 1.010, 1.020, 1.025] yield 2 then 1 in adjacent 20 ms bins; the event
on the shared edge belongs once to the earlier bin. Understanding update:
correctly assigned a shared-edge event to the earlier bin and identified rows as
20 ms intervals and columns as ordered recording channels. Correctly rejected
zero counts as proof of missing data and recognized that movement may still occur.
Clarified that zero means no recorded events in the included channel/unit slots
for that bin, not absence of all motor-cortex activity or proof of uninterrupted
acquisition. Independent explanation of that measurement limitation remains
pending. Prior demonstrated timing/purge understanding and outstanding
validation/test distinction remain as recorded in Phase 2A.

No targets, normalization, smoothing, activity-based channel selection, history
windows, models, data download, demo, or CI were added. Phase 2B implementation
and verification are complete locally; changes remain uncommitted. Last verified
commit/recorded push is 96e40d0 on main/local origin/main. No staging, commit,
push, merge, or branch switch was performed by the assistant.

After the understanding and user Git checkpoint, the next increment is
**Phase 2C: derive cursor-velocity targets and align them with these bins**.

## Phase 2C: aligned cursor-velocity targets — October 11, 2026

### Verified starting state and prior output

Branch main at 4f62fca (bin spike events into causal channel counts). Main and
local origin/main match, recording the prior push; no new remote query/fetch
was performed. Phase 2B code, docs and learning update are committed. Tracked
files/index were clean on entry. Preserved the unrelated untracked Windows cache
directory. Read the complete local guidance, progress, methods, dataset card,
decisions, actual split/configuration, manifest, loader, binning and support code.
No applicable AGENTS.md was found. Private guidance remains locally excluded
and untracked; raw recordings and bulk processed arrays remain ignored/untracked.

Reproduced Phase 2B's training-only command before implementation: [13079, 96]
int64 counts, M1 001–096, all five slots, first/last bin bounds and independent
219,913-event conservation matched its recorded results. Source remains Indy
indy_20161005_06, Zenodo 583331 v1, SHA-256
4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac.
No saved Phase 2B array/cache existed to reuse; computation remains in memory.

### Timing amendment and implemented files

Phase 2A's methods proposed fixed nominal 20 ms division. Phase 2C explicitly
replaces that denominator with actual selected behavioral elapsed seconds,
as recorded in [decisions.md](decisions.md). Neural bins remain 20 ms.
Source selection/support, fixed 6 ms age limit, partitions, purge gaps and local
warm-up are unchanged. The actual split specification/configuration are unchanged;
they do not prescribe a target denominator. No silent scientific-policy change
or held-out-dependent tuning was made.

- src/neurodecode/data/targets.py: AlignedTargets and actual-elapsed backward
  differences, original-bin pairing, validity mask/reasons and source-time metadata.
- scripts/bin_recording.py: --targets audit, retaining existing no-flag behavior
  and source/config/split verification. Reports quality counts for all partitions
  and only fixed first-valid training position/velocity examples.
- tests/test_targets.py: 24 synthetic target/cause/identity/invalid-input cases.
- docs/methods.md: effective formula, sampling, denominator bounds, structural/
  row rejection, alignment, noise caveats, real example and audit results.
- docs/decisions.md: explicit denominator amendment and reason.
- README.md and docs/dataset.md: accurate implemented status and run command.
- This entry: actual outcomes, learning state and next increment.

Select latest samples <= the stored shared grid start/end, never future samples,
with no interpolation. Divide x/y displacement by b1−b0 in seconds.
Record both source indices/times/ages, actual elapsed time and actual-minus-nominal
difference. Require positive elapsed time and absolute deviation <= maximum
sample age + 2 ns (6.000002 ms here), derived from bounded ages and floating-grid
representation. Never substitute nominal time or clamp a rejected denominator.

Reuse existing binning to revalidate source counts, complete geometry, order,
grid IDs, partition/segment identity and channel/unit mapping before pairing.
Fail on structural mismatches rather than trimming/reordering. Duplicate,
descending or nonfinite source timestamps and bad position dimensions fail.
Stale segment metadata crossing a known gap fails for replanning/rebinning.
Properly split gaps omit rows and reset support. Invalid row labels are NaN with
explicit reasons, while counts and original identities remain intact.

Reasons distinguish missing past samples, cross-segment selections, staleness,
nonpositive elapsed time, excessive elapsed deviation, warm-up, nonfinite selected
positions and nonfinite arithmetic. Full local support/cardinality is enforced
without constructing history arrays. Selected nonfinite positions on a directly
supplied Recording invalidate affected rows; existing raw-loader strict finite
validation is retained. Counts and velocities must use the same mask.

### Commands and verification

Executed from the repository root:

~~~text
uv --version
uv --cache-dir .uv-cache run --locked --no-sync python -c "import sys, numpy; print(sys.version); print(numpy.__version__)"
uv --cache-dir .uv-cache run --locked --no-sync python scripts/bin_recording.py
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider tests/test_targets.py
uv --cache-dir .uv-cache run --locked --no-sync python scripts/bin_recording.py --targets
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider
uv --cache-dir .uv-cache run --locked --no-sync ruff check --no-cache .
uv --cache-dir .uv-cache run --locked --no-sync ruff format --check --no-cache .
uv --cache-dir .uv-cache lock --check --offline
git diff --check
~~~

Observed uv 0.12.23, CPython 3.11.17 and NumPy 2.4.6. No dependency was added
or environment reinstalled. Formatted only target module, target tests and the
modified script. Initial 22 target cases passed; after adding empty-fragment and
finite-input overflow cases, all 101 tests passed (77 retained plus 24 new).
The final no-flag reproduction also preserved Phase 2B shape and per-channel
conservation after the command extension. Lint, formatting, lock consistency
and whitespace reviews passed. Git's existing
LF-to-CRLF notices are normalization notices. No checks remain blocked.

Synthetic evidence includes constant velocity sign/units, an irregular 16 ms
selected interval using the actual denominator, exact and one-float-step future
boundaries, future-position mutation, stale samples, valid gap resets/stale-gap
refusal, purge ownership, nonpositive separation, selected nonfinite positions,
future nonfinite values not changing past labels, finite-input arithmetic
overflow, malformed clocks/shapes, mismatched row lengths/order/timestamps/IDs/
counts/channel metadata, and empty aligned shapes. This evidence is separate
from real recording observations and does not establish physiological alignment,
noise-free labels, model performance or inference latency.

### Actual recording alignment

| Partition | Count rows retained | Valid aligned targets | Invalid reason |
| --- | --- | --- | --- |
| Training | 13079 | 13070 | 9 warmup |
| Validation | 2784 | 2775 | 9 warmup |
| Test | 2793 | 2784 | 9 warmup |

There are 22 / 222 / 222 valid selected-time separations differing from nominal
by >1 ns in training/validation/test respectively. These are timestamp-quality
counts, not movement-pattern statistics. Actual deviations are retained and
all pairs obey the predeclared bounds. Held-out labels were deterministically
prepared for validity auditing only; no held-out movement patterns, velocity
statistics, predictions or scores were displayed or used to tune rules.

First valid training example: original row 9, global end ID 10, segment 0,
neural bin (1288.180, 1288.200], first three channel counts [1, 1, 0].
Selected positions are:
at 1288.180 s, [-7.882167580583823, 30.560668915346305] mm;
at 1288.200 s, [-7.945765364028281, 30.651556978673803] mm.
Displacement [-0.06359778344445832, 0.09088806332749755] mm divided by actual
0.01999999999998181 s gives [-3.179889172225808, 4.54440316637901] mm/s.
Both source ages are zero; nominal-time deviation is -1.81903103690928e-14 s.
The example is fixed by chronological valid identity, not selected for quality.

### Concepts, pending understanding and next increment

Explained position (mm) versus velocity (mm/s), signed x/y backward differences,
past-only sample selection and explicit bin identity, actual versus nominal
elapsed time, shared-grid floating-point boundaries, and noise amplification.
Worked example: displacement [2, -1] mm over 16 ms gives [125, -62.5] mm/s;
a 0.1 mm displacement error over 20 ms contributes 5 mm/s error. No noise
filtering is implied. Understanding update: correctly selected the actual
16 ms sample separation as denominator and rejected using a future sample for
the current bin. Clarified conversion to 0.016 seconds for mm/s. Corrected the
position/velocity distinction: negative x velocity means decreasing x, even when
both positions are positive. Clarified that availability in a later bin does not
guarantee selection; each boundary uses its latest eligible past-or-present sample.
After a numerical recheck, correctly explained that increasing x from -5 mm
to -3 mm gives positive x velocity despite negative positions. Signed velocity
understanding was demonstrated in that example. An independent explanation of
later-boundary sample selection remains pending. Earlier demonstrated bin-edge/shape understanding and remaining
measurement/evaluation clarifications are retained as recorded in Phase 2B.

No normalization, history arrays, models, replay, data download, CI, scientific
dependencies or bulk-array persistence were added. Loader, binning implementation,
split/support code, manifest, configuration and lockfile remain unchanged.
Phase 2C is locally implemented and verified; changes remain uncommitted.
Last verified commit/recorded push is 4f62fca on main/local origin/main.
The assistant performed no staging, commit, push, merge or branch switch.

After the understanding and user Git checkpoint, the next increment is
**Phase 2D: fit count normalization on training data only**.


## Phase 2D — causal histories and training-only transforms (2026-10-11)

### Verified starting checkpoint and repository state

Read the private project guide, applicable public progress/methods/dataset notes,
saved split/configuration and current binning/target implementations. No applicable
AGENTS.md was found in checked ancestors or affected directories.
Starting branch main, HEAD 56d16d9 (align cursor velo with neural bins).
The local origin/main reference also points to 56d16d9; ahead/behind is 0/0.
This establishes a committed Phase 2C and the locally recorded remote state;
no network refresh/push/merge was performed or inferred from earlier notes.
The historical Phase 2C uncommitted status above is superseded by this inspection.

Tracked working tree and index were clean before this increment. The unrelated
untracked %SystemDrive%/ cache directory was preserved.
CODEX.md remains untracked via .git/info/exclude; data/raw/, data/processed/ and
runs/ remain ignored and have no tracked files. The acquired recording and
bulk outputs were not added to Git. No branch switch or Git mutation was performed.

Reused the existing Python 3.11.17 project environment and locked dependencies.
No dependency, loader, binning, target-alignment, timing configuration, source
manifest or split-specification change was needed. Reproduced the Phase 2C
target audit before implementation, including the existing 13070/2775/2784 valid
target counts and nine warm-up exclusions in each partition.

### Implemented scope and design

Added src/neurodecode/data/features.py: validated oldest-to-newest ten-bin
raw histories, explicit source/target IDs and timestamps, [N,10,96] features
paired with [N,2] mm/s targets, reversible [N,960] C-order flattening, and a frozen
training-only per-channel scaler. It consumes existing processed structures;
it does not read or independently align recordings.

All source rows must be consecutive on the exact saved grid and inside one
retained partition/continuous segment. Warm-up target rows can supply valid
counts; other quality failures and explicit invalid-bin masks break histories.
Missing grid bins and segment changes cannot be hidden by adjacent array rows.
Zero-count bins remain valid. Drop incomplete histories without stitching,
sorting, interpolation, repair or future lookahead.

Fit population mean/std over unique valid original training count rows once,
including warm-up source bins, rather than repeated overlapping window entries.
Replace exact zero training standard deviation with scale 1 while retaining
the channel. Apply read-only training statistics identically at every lag and
to every partition; never fit/update from held-out values. Raw counts and
velocity targets are preserved separately. Intentional offline refitting after
a change to later training data may change statistics; applying a frozen scaler
does not update them.

Extended scripts/bin_recording.py with mutually exclusive --features/--targets.
The feature command reuses the existing source/config/split checks and
bin/target implementations, prints per-partition eligibility/tensor summaries
and only the first three chronological training examples.
Small ignored runs/phase2d/count_scaler.json contains full means/scales, constant
flags, ordered channels/unit slots, history/flattening conventions, units and
source/split/configuration/target-policy/implementation identities, plus fit-bin
and raw-fit-count hashes. Identical reruns are allowed; conflicting metadata
raise an error without overwriting it. No bulk array persistence/cache was added.

Updated README setup/status and commands, docs/dataset.md status,
docs/methods.md detailed eligibility/formulas/provenance and the real example,
and docs/decisions.md the unique-bin fit and offline/frozen causality distinction.
Added tests/test_features.py; existing tests remain intact.

### Commands actually executed and evidence

From the repository root, using the installed environment and an ignored local
uv cache:

~~~powershell
uv --cache-dir .uv-cache run --locked --no-sync python scripts/bin_recording.py --targets
uv --cache-dir .uv-cache run --locked --no-sync ruff format src/neurodecode/data/features.py tests/test_features.py scripts/bin_recording.py
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider tests/test_features.py
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider
uv --cache-dir .uv-cache run --locked --no-sync ruff check --no-cache .
uv --cache-dir .uv-cache run --locked --no-sync ruff format --check --no-cache .
uv --cache-dir .uv-cache lock --check --offline
uv --cache-dir .uv-cache run --locked --no-sync python scripts/bin_recording.py --features
~~~

Initial focused run passed 21 tests; three further cases brought the final new
test count to 24. The complete suite passed 125 tests, including 101 retained
checks and the existing outside-checkout installation/import smoke test.
Lint passed; all 25 Python files passed formatting; lock consistency passed.
The feature command ran twice with identical stdout and identical metadata.
Whitespace/index/ignore checks also passed. No required check remained blocked.
A fresh environment install was not repeated; --no-sync uses the previously
verified installed environment.

Synthetic checks cover oldest/newest order and final-target pairing, shapes,
flatten/inverse order, zero/empty windows, explicit invalid rows, target-quality
failures, missing grid IDs, segment resets/crossing refusal, retained partition
ownership, future-count mutation, frozen transformations, held-out changes,
unique-bin fit versus repeated-window weighting, constant/silent channels,
source/channel/history identity mismatch, target/timestamp/mapping mismatch,
explicit invalid-bin fit exclusion and raw-count preservation.

### Actual recording preparation

| Partition | Valid count bins | Valid targets / retained windows | Warm-up excluded | Other invalid windows | Feature shape | Target shape |
| --- | --- | --- | --- | --- | --- | --- |
| Training | 13079 | 13070 | 9 | 0 | [13070,10,96] | [13070,2] |
| Validation | 2784 | 2775 | 9 | 0 | [2775,10,96] | [2775,2] |
| Test | 2793 | 2784 | 9 | 0 | [2784,10,96] | [2784,2] |

Raw/normalized feature shapes match; flattened shapes are [13070,960],
[2775,960], [2784,960]. Held-out reporting used aggregate counts/shapes only;
rules were not tuned from held-out activity or movement patterns.

Scaler fit used all 13079 unique valid training bins once. Eleven constant
training channels retain scale 1: M1 009, 047, 055, 058, 061, 068, 071, 078,
079, 082, 096. M1 001 mean=0.12890893799220124 events/bin and
scale=0.36460280279149737 events/bin.

First real training example: source rows 0–9, end IDs 1–10, segment 0,
target row 9/end ID 10. Bin ends:
[1288.020,1288.040,1288.060,1288.080,1288.100,1288.120,1288.140,1288.160,
1288.180,1288.200] s, with shared starts beginning at 1288.000 s.
One feature array is [10,96]; its final-bin target is
[-3.179889172225808,4.54440316637901] mm/s, retaining Phase 2C alignment.
M1 001 raw history [0,2,1,2,1,1,2,2,1,1] has oldest normalized feature
(0−0.12890893799220124)/0.36460280279149737 = −0.35355992056352736.
Negative normalized values mean below training mean, not negative spike counts.
The first three training histories were inspected by their recorded source
bin identities/timestamps, with no favorable-example selection.

### Concepts demonstrated, pending understanding and next increment

Explained neural history as recent activity context, [N,L,C] with actual L=10
and C=96, final-bin label pairing, training-only preprocessing leakage,
zero observations versus absent/invalid observations, oldest-first flattening,
unique-bin fitting versus overlap weighting, safe constant-channel handling,
and offline training fit versus causal application of frozen statistics.
Synthetic tests demonstrate those implementation properties. Understanding
check: correctly excluded a bin ending at 1288.220 s from a history for a target
ending at 1288.200 s because it is future activity. Correctly identified that a
valid zero-count bin does not break a history. Final-bin target pairing and
training-only normalization leakage were explained after the initial checks;
independent explanations of those two concepts remain pending. Oldest-to-newest
feature order also remains pending independent explanation. Prior demonstrated/
pending concepts remain as recorded in the Phase 2C learning update.

Checks establish tested preparation logic and recording eligibility, not neural
predictiveness, physiological alignment, independent overlapping examples,
noise-free velocity, decoding quality or latency. No target normalization,
model/baseline training, metrics, replay, dependency installation, data download,
CI or bulk processed arrays were added.

Phase 2D is locally implemented and verified; its edits remain uncommitted.
The assistant performed no staging, commit, push, merge or branch switch.
After review, understanding checks and the user's Git checkpoint, the next
increment is **Phase 3A: establish zero-velocity and training-mean baselines,
then define evaluation metrics**.


## Phase 3A — trivial velocity predictors and validation metrics (2026-10-11)

### Actual starting checkpoint

Read unchanged private guidance in full, applicable public progress/methods/
decisions, the saved split/configuration, prepared-window implementation and
existing tests. No applicable AGENTS.md was found in ancestors/affected paths
or the repository scan. Starting branch main, commit 249ce68 (build causal
decoding windows + the train-only transforms). Local origin/main also points to
that commit, ahead/behind 0/0. This supersedes Phase 2D's historical uncommitted
status above. No remote refresh was performed, so this is locally recorded push
state, not a new server check.

Tracked files and index were clean; the unrelated untracked %SystemDrive%/
cache directory remains preserved. CODEX.md remains excluded via the local
exclude file and untracked. Raw/processed data and runs/ remain ignored and
untracked. Recording identity, splits, timing, eligibility and Phase 2D output
were preserved. No staging, commit, push, merge or branch switch was performed.

Existing environment: Python 3.11.17, NumPy 2.4.6, h5py 3.16.0, Matplotlib 3.11.2,
with existing pytest/Ruff. No dependency/lockfile, data acquisition, scientific
preparation or scaler change was needed.

### Implemented and documented

Added models/baselines.py with zero velocity and a frozen training-target mean.
The minimal models/__init__.py makes that implemented package explicit.
TrainingMeanVelocity.fit accepts training only; predict accepts label-free
example metadata. It cannot receive validation labels or update its mean.

Added evaluation.py with explicit indexed [N,2] velocity series and strict shared
scoring. Exact timestamps, global bin IDs, segment IDs, partition, source/split/
configuration/policy identity, channel/unit order and masks must agree.
Malformed lengths/order, nonfinite values or arithmetic overflow fail; no
sorting, trimming, implicit masking or flattering undefined-score replacements.

Metrics: per-axis RMSE in mm/s; overall RMSE averages squared errors across
both axes/examples before sqrt; per-axis R²=1−SSE/SST; unweighted mean R².
Constant axes or fewer than two scored rows give explicit null R²/status.
Mean R² is null if either axis is undefined. No scored rows is an error.
Scoring-set means are reference quantities inside the metric, not predictor inputs.

Added scripts/evaluate_baselines.py, reusing the source/config/split verification
extracted unchanged from scripts/bin_recording.py and existing bin/target/window
functions. Prepare only train and validation. Fit mean on 13070 eligible training
targets once, generate both predictions from label-free metadata, then access
validation targets for scoring. All 2775 eligible validation rows share one
all-true mask; no additional exclusion or parameter tuning was introduced.

Extended plotting.py for the fixed first eligible ten validation seconds,
with x/y velocity units, shared time offset, unsmoothed actual derived labels
and both constant predictions. Refuse gaps, segment changes, masked rows or
insufficient first-interval coverage; do not search for better-looking movement.

Ignored runs/phase3a-baselines/ contains run.json, metrics.json, predictions.csv
and validation_velocity.png. Metadata save effective settings, source/full split/
configuration/policy/code hashes, channel/unit order, frozen mean, eligibility,
environment, Git commit/dirty state, plotting rule and whole-run duration.
The base commit was 249ce68 but the new implementation was dirty/uncommitted;
results must not be attributed to that commit's code alone.
CPU identifier was unavailable; duration is not an inference-latency measurement.

Updated README, methods and decisions; added the small provenance-verified
reports/phase3a_baselines.md. Added tests/test_evaluation.py.
Generated predictions/plots/run metadata remain outside Git.

### Checks actually run

From the repository root:

~~~powershell
uv --cache-dir .uv-cache run --locked --no-sync python -c "import sys, numpy, matplotlib; print(sys.version); print(numpy.__version__, matplotlib.__version__)"
uv --cache-dir .uv-cache run --locked --no-sync ruff format src/neurodecode/evaluation.py src/neurodecode/models src/neurodecode/plotting.py scripts/bin_recording.py scripts/evaluate_baselines.py tests/test_evaluation.py
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider tests/test_evaluation.py
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider
uv --cache-dir .uv-cache run --locked --no-sync ruff check --no-cache .
uv --cache-dir .uv-cache run --locked --no-sync ruff format --check --no-cache .
uv --cache-dir .uv-cache lock --check --offline
uv --cache-dir .uv-cache run --locked --no-sync python scripts/bin_recording.py
uv --cache-dir .uv-cache run --locked --no-sync python scripts/evaluate_baselines.py phase3a-baselines
uv --cache-dir .uv-cache run --locked --no-sync python scripts/evaluate_baselines.py phase3a-repeat
~~~

All 28 new synthetic tests passed; all 153 tests passed including 125 retained
checks and the existing outside-checkout import smoke test. Ruff lint passed;
all 30 Python files passed formatting; lock consistency passed.
Initial string-escaping parse errors and an import-order issue were corrected
before final checks and real measurement. The existing training-count command
still reports [13079,96] and per-channel conservation of 219913 included events.

Synthetic evidence includes hand-computed two-axis/overall metrics, perfect and
negative R², constant-axis/single-example undefined values, training-only frozen
means unchanged under validation-label mutation, exact shape/timestamp/order/ID/
segment/context/mask mismatch failures, nonfinite/overflow rejection, explicit
shared masks, final-source-bin identity and value-independent fixed plot rules.

Audited exported CSV separately: 2775 finite ordered validation rows, all scored,
correct target/final source identities and both fixed prediction pairs.
Independent score recalculation from CSV matched saved metrics. The second run
produced byte-identical prediction CSV and metrics JSON. An intentional rerun
using the existing phase3a-baselines directory exited with the expected
preservation error; original run metadata were unchanged. This is a tested
refusal, not a blocked acquisition or failed scientific run.

The actual figure was opened and visually reviewed: first interval
[1550.200,1560.200) s, 500 rows, correctly shared timestamps and mm/s labels.
Both constant predictions form flat nearly overlapping lines near zero.
No required verification remains blocked. Fresh installation was not repeated;
--no-sync uses the previously verified environment.

### Measured validation results and one example

The [validation report](../reports/phase3a_baselines.md) holds source/checksum/
configuration provenance and full protocol. All scored bin ends run from
1550.200 to 1605.680 s inclusive (55.480 s timestamp span).
Training mean from 13070 eligible targets is
[-0.1479870893879215,0.1386043376635596] mm/s.

| Predictor | RMSE x (mm/s) | RMSE y (mm/s) | Overall RMSE (mm/s) | R² x | R² y | Mean R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero velocity | 78.418547 | 66.748063 | 72.817486 | −2.097390e−7 | −2.416840e−8 | −1.169537e−7 |
| Training mean | 78.418755 | 66.748186 | 72.817654 | −5.499573e−6 | −3.690502e−6 | −4.595038e−6 |

At bin end 1550.200 s, ID 13110, actual derived velocity is
[14.162167438059216,3.7882881059650577] mm/s.
Zero prediction [0,0] gives signed prediction−target error
[-14.162167438059216,-3.7882881059650577] mm/s.
The frozen training-mean pair gives errors
[-14.310154527447137,-3.649683768301498] mm/s.
These one-row signed errors are not aggregate RMSE.

These are baseline validation references only. They do not establish neural
decoding ability, final test performance, human/live control, significance or
latency. Test target/window construction, predictions, plots and scoring were
not run in this increment. The common loader reads the source recording as
before; prior deterministic test eligibility audits remain documented.
No rule/model/plot interval was selected using test outcomes.

### Learning status and next increment

Explained why complex decoders need simple references; zero versus training
mean; signed error versus RMSE; per-axis/overall RMSE; R² as squared-error
comparison with the scoring-set mean, including legitimate negative scores
and undefined constant-axis cases; validation development versus closed final
test; label-free prediction versus later scoring-label access.
Hand example: targets [[0,0],[2,4]], zero predictions give axis RMSE
[sqrt(2),sqrt(8)], overall sqrt(5), both R²=−1.
Initial understanding check left training-mean target provenance, negative
R² and validation/test roles uncertain. Clarified that the mean uses only the
13070 eligible training targets, separately per axis, then freezes. Negative
R² describes squared error worse than the scoring-set mean reference, not
negative correlation. One-axis example: actual [0,2], zero predictions give
SSE=4 versus mean-reference SSE=2, hence R²=−1. Validation provides development/
selection feedback; final test scoring stays closed until choices are fixed.
Understanding follow-up: correctly explained that changing validation
velocities does not change the frozen training-mean prediction. Prediction
invariance under validation-label changes is demonstrated; clarified that
validation scores can still change. Understanding follow-up: correctly explained that negative R² means
greater squared prediction error than the scoring-set mean reference. This
interpretation is demonstrated. Understanding follow-up: correctly explained that validation guides tuning
and the final test set evaluates the model after those choices are fixed.
Validation versus final test roles are demonstrated. The three checkpoint
follow-ups now demonstrate frozen training-mean prediction, negative R²
interpretation and validation/test separation. Independent explanations of
RMSE aggregation and training-target mean calculation remain pending.
Phase 2D demonstrated future-bin exclusion and valid zero-count understanding
remain recorded; its outstanding independent explanations remain pending.

No ridge/other learned model, target scaling, trajectory integration, demo,
test-score opening, dependency addition, CI or bulk data tracking was added.
Phase 3A is locally implemented and verified; its changes remain uncommitted.
The assistant performed no staging, commit, push, merge or branch switch.
Next increment after understanding/review and the user's Git checkpoint:
**Phase 3B: fit and validate a ridge-regression decoder on the same examples**.


## Phase 3B — ridge movement decoder (2026-10-18)

### Verified starting state and prior checkpoint

Read unchanged CODEX.md completely (including sections 15/16), applicable public
progress/methods/decisions, saved split/configuration and baseline/evaluation/
window/normalization code. No applicable AGENTS.md was found in checked ancestor
and affected directories. Starting branch main, HEAD cd71d47 (velo baselines and
eval metrics), local origin/main matching, ahead/behind 0/0.
Phase 3A is committed; its historical uncommitted status above is superseded.
GitHub main was also verified at cd71d47 in the preceding status check in this
session; this increment did not refresh or modify remote references.

Tracked working tree and staging were clean. Preserved the unrelated untracked
%SystemDrive%/ cache directory and all existing generated outputs.
CODEX.md stays untracked through .git/info/exclude; data/raw/, data/processed/
and runs/ remain ignored and have no tracked files. No Git stage, commit, push,
merge or branch switch was performed. No dataset or split/timing change occurred.

Inspected actual Phase 3A metrics/run metadata in runs/phase3a-baselines.
They record the previously measured 2775 validation rows, near-zero R² and
overall RMSE 72.817486/72.817654 mm/s for zero/training mean. That run's code
was dirty relative to its recorded base commit, as already documented.

### Concepts explained before implementation

One [10,96] history becomes 960 oldest-to-newest features, channels fastest.
Weights [960,2] map standardized counts to vx/vy; an unpenalized [2] intercept
provides the constant output offset. Lambda penalizes large weights: stronger
shrinkage can reduce noise fitting but may suppress useful variation.
Worked one-output example: features [2,1], weights [3,−1], intercept 0.5 give
2*3−1+0.5=5.5. Training-only normalization prevents held-out activity from
changing feature scaling. The exact Phase 2D fit basis remains unique training
bins, including source warm-up, not repeated window entries.

Declared before candidate results: lambda grid [0.1,1,10,100,1000,10000],
primary metric overall validation RMSE, tie tolerance absolute 1e-9 mm/s,
largest tied lambda wins. Explained that validation may change the selected
lambda but must not fit any candidate's weights or normalization.
Phase 3A demonstrated frozen-mean prediction, negative R² interpretation and
validation/test separation remain recorded. Ridge regularization, flattening/
matrix dimensions and model comparison understanding remain pending teach-back;
implementation/test evidence is not proof of independent understanding.

### Implemented files and scope

- configs/ridge.toml: predeclared solver/grid/selection policy, separate from the
  unchanged source-bound base timing configuration.
- src/neurodecode/models/ridge.py: label-free normalized RidgeInputs, sklearn
  SVD training-only fitting, immutable W/b, full transform fingerprint, explicit
  numeric JSON persistence, strict loading/context checks and deterministic
  selection/tie rule. Intercept is unpenalized; no explicit matrix inversion.
- src/neurodecode/data/features.py: restore/validate frozen fit metadata and a
  thin shared prepare_partition composition of existing bin/target/window code.
  Core eligibility, raw counts, timing/targets and fit_count_scaler are unchanged.
- scripts/train_ridge.py: training-only six-candidate fit, validation selection,
  common scoring of selected ridge and both trivial predictors, ignored run
  artifacts and exact model-load prediction check. No test partition preparation.
- scripts/evaluate_baselines.py: use shared preparation and a shared timestamped
  exporter; retain original baseline CSV columns/values. Its git_state helper is
  reused for run provenance. Existing run directories remain preserved.
- src/neurodecode/plotting.py: allow a third predictor on the same baseline
  validation interval; show the ridge trace with the original units/time rule.
- tests/test_ridge.py: 30 focused synthetic checks.
- pyproject.toml and uv.lock: add scikit-learn>=1.7,<1.8 through uv. Actual installed
  scikit-learn 1.7.2 brings SciPy 1.17.1, joblib 1.6.0, threadpoolctl 3.7.0 and
  cloudpickle 3.1.2. Existing NumPy 2.4.6 remains unchanged.
- README, dataset/methods/decisions/progress and reports/phase3b_ridge.md record
  commands, objective, protocol, verified validation results and limitations.

Consulted official versioned sklearn Ridge documentation and installation
requirements. Inspected the installed library's intercept reconstruction;
large-lambda constant-target tests independently verify an unpenalized intercept.
Initial dependency access was blocked by restricted networking; the approved
network-enabled uv add completed successfully and generated the lockfile.
No future ML stack or dataset download was added.

### Actual verification and reproduction

Commands executed from the repository root:

~~~powershell
uv --cache-dir .uv-cache add "scikit-learn>=1.7,<1.8"
uv --cache-dir .uv-cache run --locked --no-sync ruff format src/neurodecode/models/ridge.py src/neurodecode/data/features.py scripts/evaluate_baselines.py src/neurodecode/plotting.py scripts/train_ridge.py tests/test_ridge.py
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider tests/test_ridge.py --tb=short -x
uv --cache-dir .uv-cache run --locked --no-sync pytest -p no:cacheprovider --tb=short
uv --cache-dir .uv-cache run --locked --no-sync ruff check --no-cache .
uv --cache-dir .uv-cache run --locked --no-sync ruff format --check --no-cache .
uv --cache-dir .uv-cache lock --check --offline
uv --cache-dir .uv-cache run --locked --no-sync python scripts/train_ridge.py phase3b-ridge
uv --cache-dir .uv-cache run --locked --no-sync python scripts/evaluate_baselines.py phase3b-baseline-check
uv --cache-dir .uv-cache run --locked --no-sync python scripts/train_ridge.py phase3b-ridge-repeat
~~~

Focused tests passed 30/30; all 183 tests passed including 153 retained checks.
Lint passed; all 34 Python files passed formatting; lock consistency passed.
An initial incomplete synthetic split fixture lacked overlap metadata and was
corrected before final tests/real measurement; a source-string escaping error
was also corrected during development. No dataset/eligibility relaxation was
used to fix a test. No required check remains blocked. Tests include the existing
outside-checkout installation/import smoke test; no fresh temporary environment
installation was repeated.

New synthetic evidence covers recovery of a known two-output linear mapping,
comparison with sklearn predict, actual oldest/channel feature order,
unpenalized intercept under large lambda, stronger weight shrinkage,
save/load equality, incompatible channel/transform/history/context refusal,
nonfinite/malformed input/artifact refusal, validation-label independence of
candidate fitting/normalization, selection/ties and preserved fit policies.

The real command restored Phase 2D normalization from the existing ignored
metadata and verified exact agreement with a fresh training-only fit.
Normalizer fingerprint:
603aa299f0a1d75625acea41a18b9ecf56c58e4e0515ec8322ea1aa34abe68dd.
Its 13079 fit source bins and all 96 channel identities remain unchanged.

The selected model's save/load gave exact validation prediction equality.
A repeated real run produced byte-identical ridge.json, predictions.csv and
metrics.json. Independent recalculation from the exported CSV reproduced ridge
axis/overall RMSE and R². Every original baseline CSV column/row matches the
Phase 3A export exactly; rerunning its entry point also reproduced its metrics.
The real figure was opened and visually inspected: correct x/y velocity units,
shared offset, fixed [1550.200,1560.200) s interval and 500 samples, no shift/
smoothing/integration. Both constant baseline traces remain visible near zero.

Generated artifacts stay under ignored runs/phase3b-ridge and the repeat/check
directories: full model/normalization JSON, effective configuration/provenance,
all candidate scores, package versions/native thread settings, Git commit/dirty
state, code/artifact hashes, metrics, timestamped predictions and PNG.
New code was uncommitted relative to base cd71d47 when measured; do not attribute
the result to that base commit alone. Native thread pools were limited to one.
Whole-run duration includes preparation/grid fitting/export and is not latency.

### Actual selection, comparison and example

All 13070 eligible training and 2775 validation examples retain [10,96] histories,
flattened to [960], with unchanged [2] mm/s labels and exact identities/masks.
Scored validation ends span 1550.200–1605.680 s inclusive (55.480 s span).

| Lambda | Overall validation RMSE (mm/s) |
| ---: | ---: |
| 0.1 | 52.886882 |
| 1 | 52.886503 |
| 10 | 52.882726 |
| 100 | 52.846123 |
| 1000 | 52.571380 |
| 10000 | 52.807856 |

Selected lambda=1000, an interior value. No tie applied and no grid extension
or test-based selection occurred.

| Predictor | RMSE x (mm/s) | RMSE y (mm/s) | Overall RMSE (mm/s) | R² x | R² y | Mean R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero velocity | 78.418547 | 66.748063 | 72.817486 | −2.097390e−7 | −2.416840e−8 | −1.169537e−7 |
| Training mean | 78.418755 | 66.748186 | 72.817654 | −5.499573e−6 | −3.690502e−6 | −4.595038e−6 |
| Ridge, lambda=1000 | 56.844212 | 47.919052 | 52.571380 | 0.474546 | 0.484606 | 0.479576 |

Selected training overall RMSE is 50.301849 mm/s; training mean R² 0.578469.
These are training diagnostics, separate from validation. The selected
validation score reflects six-way model selection and remains development
evidence, not final test performance or a significance claim.

First validation prediction at 1550.200 s (ID 13110):
normalized [10,96] → flattened [960] → W [960,2] plus intercept
[-0.10485421436834126,0.12171317024650069] gives
[31.39011385437117,12.002288083723482] mm/s.
Actual [14.162167438059216,3.7882881059650577] yields signed errors
[17.22794641631195,8.213999977758423] mm/s. Squared errors aggregate through
the unchanged Phase 3A metric formulas. This individual example is worse than
zero velocity; aggregate ridge performance is better. Retained the first
chronological example without favorable-example selection.

### Limits, learning and next checkpoint

No test targets/windows, predictions, plots or scores were generated. The
existing loader still reads the acquired recording/global coverage metadata;
prior deterministic test preparation audits remain documented. Final test
performance stays closed. No target definition, split, channel filter, neural
lag, forecast horizon, smoothing or normalization policy was changed.
No other model, Kalman fitting, trajectory replay/demo, CI or latency benchmark
was implemented.

Ridge beats both trivial baselines, and current checks reveal no defect requiring
a focused diagnostic/fix increment. Explained that an individual error can be
larger while aggregate RMSE improves. Understanding follow-up: correctly
identified that increasing lambda increases the penalty on large weights.
Recognized identical scoring timestamps as necessary for fair comparison and
training-only normalization as protection against leakage. Clarified that all
predictors must score the same movement examples, and held-out activity must
not influence feature means/scales. These high-level answers are demonstrated;
independent explanations of feature dimensions, weight shrinkage and the
normalization leakage mechanism remain pending. Previously demonstrated
baseline and causal-window concepts are retained without claiming all model
mathematics are mastered.

Phase 3B is locally implemented and verified; its edits remain uncommitted.
The assistant performed no staging, commit, push, merge or branch switch.
After review/teach-back and the user's Git checkpoint, next: **the Kalman
decoder**, retaining these same evaluation timestamps and closed final test.
