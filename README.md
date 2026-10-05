# NeuroDecode

An educational project to estimate two-dimensional cursor velocity from recorded
motor-cortex spike activity. The planned comparison is ridge regression, a linear
Gaussian Kalman decoder, and a small causal GRU, followed by recorded-data replay
and controlled robustness experiments.

**Status:** Phase 0 package and tooling setup only. No dataset has been acquired;
no decoders, scientific results, or latency measurements exist yet. This is an
independent project with no Neuralink affiliation or clinical claims.

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
selection, attribution, licensing, timing, and evaluation protocols are pending
the data audit. A license for this project's code has not been chosen.
