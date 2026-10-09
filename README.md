# NeuroDecode

A project to estimate two-dimensional cursor velocity from recorded
motor-cortex spike activity. The planned comparison is ridge regression, a linear
Gaussian Kalman decoder, and a small causal GRU, followed by recorded-data replay
and controlled robustness experiments.

**Status:** Phase 0A project definition is committed, and the Phase 0B minimal
package/environment has been verified on Windows. No dataset has been acquired;
no decoders, scientific results, or latency measurements exist yet. This is an
independent project that makes no clinical claims.

## Experiment and evaluation plan

In the original reaching experiment, a monkey moves its hand while a position
tracker controls a cursor and implanted electrodes record motor-cortex spikes.
These are intracortical recordings, not scalp EEG. NeuroDecode will learn from
these paired recordings afterward: causal spike-count histories are decoder
inputs, tracker-derived velocity supplies training and scoring labels, and the
output is predicted x/y cursor velocity. Held-out movement labels will not be
fed into the decoder.

The planned evaluation uses chronological train/validation/test partitions,
boundary gaps, and training-only preprocessing. Zero and training-mean baselines,
ridge regression, a linear Gaussian Kalman decoder, and a small unidirectional
GRU will be compared on the same eligible timestamps using velocity R² and RMSE.
All models and scientific evaluation remain planned.

A later visual replay will compare actual and decoded cursor trajectories,
with decoded position obtained by integrating predicted velocity from a labeled
known initial position. It will be labeled as offline replay of held-out neural
recordings. Velocity scores will remain primary; playback speed will not be
presented as inference latency. Recorded-data results will not establish live
closed-loop control, human performance, or clinical usability.

See the [project brief](docs/project_brief.md) for timing, evaluation, replay,
and limitations, and [progress](docs/progress.md) for verified repository state.

## Setup

Python 3.11 is the supported compatibility target for this checkpoint;
compatibility with other Python versions has not been tested. Use uv 0.12.23.
There are no runtime dependencies yet. Existing pytest and Ruff development
configuration is retained; scientific dependencies will be added when needed.

### What the setup files do

- `pyproject.toml` describes the package, Python compatibility, dependencies,
  and Hatchling build configuration. It explicitly locates the package under
  `src/neurodecode/`.
- `.python-version` selects Python 3.11 for uv. The launcher/system Python used
  to install uv may be a different version.
- `uv.lock` records resolved dependency versions and hashes. Keep it in Git
  and let uv generate changes; do not edit it by hand. Build-backend requirements
  are declared separately in `pyproject.toml` and may require a download.
- `.venv/` is the local Python environment. Keep it out of Git and recreate it
  from the project configuration and lockfile on another machine.
- An editable install connects the environment to `src/neurodecode/`, so
  Python source edits take effect without reinstalling for each edit.

Use `uv sync --locked` to install the locked dependencies and the editable
package. `--locked` fails if dependency metadata needs a lockfile update instead
of silently changing it. `uv run --locked` selects the project environment, so
manual environment activation and `sys.path` edits are unnecessary. See
[uv's locking and syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

Run the following from the repository root after cloning.

### Windows PowerShell

Check for the uv executable first:

```powershell
uv --version
```

If uv is missing, install the pinned version using the Python launcher, then
restart PowerShell if needed to make `uv` available on PATH:

```powershell
py -m pip install --user uv==0.12.23
```

Use the executable directly. Availability of `uv` does not guarantee that
`py -m uv` works under every account or launcher-selected Python.

```powershell
uv python install 3.11
uv sync --locked
uv run --locked python --version
uv run --locked python -I -c "import sys, neurodecode; print(sys.executable); print(neurodecode.__file__)"
```

The interpreter path should be under `.venv\Scripts\`, and the package path
should point to `src\neurodecode\__init__.py` in this checkout. Configure your
editor to use `.venv\Scripts\python.exe` as well.

### macOS / Linux

Install uv 0.12.23 using the [official installer](https://docs.astral.sh/uv/getting-started/installation/).
These platform instructions have not been executed in this checkout.

```bash
curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh
uv python install 3.11
uv sync --locked
uv run --locked python --version
uv run --locked python -I -c "import sys, neurodecode; print(sys.executable); print(neurodecode.__file__)"
```

Use `.venv/bin/python` as the editor interpreter on these platforms.

### Installation check and existing development tools

```text
uv lock --check
uv run --locked pytest tests/test_package.py
uv run --locked ruff check .
uv run --locked ruff format --check .
```

The existing smoke test imports the installed package from a temporary directory
using an isolated Python process. This verifies discovery outside the checkout,
not neural decoding. Phase 0B reran the installation check; lint/format checks
remain existing tooling for the next checkpoint. See [progress](docs/progress.md)
for actual checks and outcomes.

For an optional repository-local uv cache, use
`uv --cache-dir .uv-cache sync --locked` (or the corresponding uv subcommand).
That directory is ignored. Network access is needed for missing Python
versions, dependencies, or build requirements. `--offline` works only when the
required artifacts are already available locally.

Raw/processed recordings, environments, caches, and generated runs remain
untracked. Small provenance files and curated results can be tracked. Dataset
selection, attribution, licensing, and verification of schema/units are pending
the data audit. Timing and evaluation choices are documented plans, not validated
implementations. A license for this project's code has not been chosen.
