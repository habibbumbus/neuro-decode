<a id="top"></a>

<img src="docs/assets/neurodecode-banner.svg" alt="NeuroDecode" width="100%">

**Predicting movement from recorded brain activity.**

NeuroDecode estimates x/y cursor velocity from motor-cortex spikes and compares ridge regression, a Kalman filter and a small GRU against simple baselines.

In the original monkey reaching experiment, implanted **intracortical electrodes** recorded spikes while tracked hand movement controlled the cursor. That movement supplies our training and scoring labels; the decoders receive neural activity. This project is **offline analysis and replay**, not EEG or live brain control.

[How it works](#how-it-works) · [Results](#held-out-results) · [Get started](#setup-and-checks) · [Demo](#fixed-interval-demo)

## How it works

```text
Spike timestamps → 20 ms channel counts → training-fitted normalization
                 → decoder → x/y velocity → scoring or cursor replay
```

Ridge learns a regularized linear mapping; Kalman estimates velocity through a state-space filter; the GRU learns a nonlinear mapping from recent activity. The recording provides 96 M1 channels. Ridge and GRU receive the same ten-bin history, while Kalman combines the current bin with its previous estimated state.

### Design choices

| Choice | Reason |
| --- | --- |
| Completed-bin predictions | Use only activity available at the prediction timestamp. |
| Training-only normalization | Keep held-out data out of fitted preprocessing. |
| Chronological train/validation/test split | Select settings on validation before freezing the later test evaluation. |
| Common scoring rows | Compare every model on identical targets after nine history warm-up bins. |
| Explicit state resets | Clear history at gaps/boundaries; start each GRU window with zero hidden state. |

[Timing, equations and model details →](docs/methods.md)

## Held-out results

**Ridge (λ = 1000) had the lowest overall velocity RMSE.** All models scored the same **2,784 test rows** from one recording. Lower RMSE is better; higher R² is better.

| Model | Overall RMSE [mm/s] | Mean R² |
| --- | ---: | ---: |
| **Ridge, λ = 1000** | **60.422** | **0.464** |
| GRU, seed 2026 | 67.361 | 0.338 |
| Zero velocity | 83.144 | -0.000027 |
| Training mean | 83.144 | -0.000029 |
| Kalman | 134.060 | -1.538 |

Values are rounded. GRU improved on the trivial baselines but trailed ridge; Kalman performed worse than both baselines. Positive velocity R² does not guarantee accurate position tracking—small velocity errors can accumulate into drift.

[Full per-axis results, validation comparison and audit →](reports/phase7b_final.md)

<a id="setup-and-checks"></a>

## Get started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```text
git clone https://github.com/habibbumbus/neuro-decode.git
cd neuro-decode
uv python install 3.11.17
uv sync --locked --python 3.11.17
uv run --locked python -m pytest -p no:cacheprovider
```

The synthetic tests need no recording or saved model. CPU is sufficient. For lint checks, Windows temp-folder troubleshooting and the verified setup walkthrough, see the [reproduction guide](docs/reproduction.md).

### Work with the real recording

1. **Acquire the data:** download `indy_20161005_06.mat` from the documented Zenodo v1 source into `data/raw/` and verify its checksum. Follow the [dataset guide](docs/dataset.md).
2. **Rebuild validation experiments:** follow the ordered [artifact-generation recipe](docs/reproduction.md#rebuild-validation-artifacts). New runs have their own provenance and need not match historical file hashes.
3. **Review the frozen result:** restore the [original trusted artifact bundle](docs/reproduction.md#restore-the-exact-frozen-bundle) and use the documented read-only audit. The exact bundle is not currently published as a download.

Raw recordings, checkpoints, generated predictions and full videos are outside Git. The reproduction guide records the original final-evaluation command and its prerequisites; do not overwrite preserved runs or retune against the already opened test block.

<a id="fixed-interval-demo"></a>

## Cursor replay

![Actual middle frame of the fixed held-out replay, showing reference integration, decoded paths and visible drift](docs/assets/phase7b-replay-still.png)

The verified clip uses **[1606.320, 1616.320) s**, selected by timestamp before test access. Its reference integrates tracker-derived velocity; it is not the original sampled tracker-position path. All paths share one initial anchor, with no later position corrections.

The full local video is `runs/phase7b-final-test/cursor_replay.mp4` after restoring the evidence bundle. This is offline playback, not a live-control demonstration or latency benchmark.

[Demo provenance and short presentation script →](docs/demo.md)

## Find your way around

| Location | Contents |
| --- | --- |
| [src/neurodecode/](src/neurodecode/) | Data contracts, models, training, metrics and replay |
| [scripts/](scripts/) · [configs/](configs/) | Experiment commands, audits and frozen settings |
| [tests/](tests/) | Synthetic timing, causality, leakage and numerical checks |
| [reports/](reports/) · [docs/](docs/) | Measured results, methods, reproduction and [interview notes](docs/interview_notes.md) |

## Scope and limitations

One session and one held-out time block do not establish cross-session transfer, human/clinical usefulness or closed-loop control. Correlated time bins do not support an independent-trial significance claim. Playback speed is separate from measured processing latency, and there is no hard real-time guarantee. Exact historical reproduction also requires the frozen runtime and original ignored artifacts.

**Data:** O'Doherty et al. (2017), [Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology, v1](https://doi.org/10.5281/zenodo.583331), CC BY 4.0. A code license has not yet been chosen.

[Final audit and open checks](reports/phase7d_audit.md) · [Project progress](docs/progress.md) · [Design decisions](docs/decisions.md)
