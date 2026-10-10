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
