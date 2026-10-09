# NeuroDecode

A project to estimate two-dimensional cursor velocity from recorded
motor-cortex spike activity. The planned comparison is ridge regression, a linear
Gaussian Kalman decoder, and a small causal GRU, followed by recorded-data replay
and controlled robustness experiments.

**Status:** Project definition documentation and an existing Phase 0 package/tooling
scaffold are present. No dataset has been acquired;
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

Python 3.11 is the current compatibility target. Use uv 0.12.23 to manage the
project environment. `pyproject.toml` declares dependencies, `uv.lock` records
their resolved versions, and `.python-version` selects Python 3.11. `sync --locked`
creates `.venv` and installs the package in editable mode without changing the
lockfile. No virtual-environment activation is needed with the commands below.

Run from the repository root after cloning.

### Windows PowerShell

The Python launcher (`py`) must be available. Its existing Python installs uv;
uv then installs the project's Python 3.11 separately.

```powershell
py -m pip install --user uv==0.12.23
py -m uv python install 3.11
py -m uv sync --locked
py -m uv run --locked python --version
py -m uv run --locked python -c "import neurodecode; print(neurodecode.__name__)"
py -m uv run --locked ruff check .
py -m uv run --locked ruff format --check .
py -m uv run --locked pytest
```

### macOS / Linux

Install the same uv version with the [official installer](https://docs.astral.sh/uv/getting-started/installation/),
then restart the shell if needed to find `uv`.

```bash
curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh
uv python install 3.11
uv sync --locked
uv run --locked python --version
uv run --locked python -c "import neurodecode; print(neurodecode.__name__)"
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

The smoke test imports the installed package from a temporary directory using an
isolated Python process. This checks package discovery outside the checkout; it
does not test neural decoding. Scientific dependencies will be added when their
implementation phase needs them. See [progress](docs/progress.md) for checks
actually executed and pending work.

Raw/processed recordings, environments, caches, and generated runs remain
untracked. Small provenance files and curated results can be tracked. Dataset
selection, attribution, licensing, and verification of schema/units are pending
the data audit. Timing and evaluation choices are documented plans, not validated
implementations. A license for this project's code has not been chosen.
