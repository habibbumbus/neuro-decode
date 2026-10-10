# NeuroDecode project brief

## Goal and current scope

Estimate two-dimensional cursor velocity from recorded motor-cortex spike
activity, and compare accuracy, temporal modeling, CPU processing latency, and
robustness. The repository currently has a minimal Python package/tooling scaffold
and project documentation. One recording has been acquired and checksum-verified;
internal inspection, decoders, evaluation, and replay are pending. There are no
scientific results or measured latency claims.

The first scientific milestone will use one public nonhuman-primate reaching
session. The selected source/version, license, attribution, and checksums are
recorded in the [dataset card](dataset.md). Channel metadata, timestamps, storage
layout, and physical units remain pending inspection before modeling assumptions
are implemented.

## Original experiment and labels

A monkey physically moves its hand or fingertip during a reaching task. A
position tracker maps that movement to a cursor, while implanted electrodes
record neural activity concurrently. The original cursor follows the tracker.
NeuroDecode will learn the relationship between neural activity and movement
afterward from the paired recordings.

Intracortical spikes are brief electrical events recorded by electrodes inside
cortex. They differ from scalp EEG, which measures electrical activity through
electrodes on the head. The proposed pipeline will use provided spike-event
timestamps rather than implement raw-voltage spike detection or sorting.

Tracker-derived cursor positions provide reference labels, often called ground
truth. Velocity labels will be derived from position changes over time. They
are measurements with sampling and differentiation error, not perfect
instantaneous velocity. For example, a 2 mm x-position change over 0.020 s gives
an average x velocity of 100 mm/s.

## Decoder inputs and outputs

Inputs will be spike counts per electrode in just-completed time bins and,
where required, a bounded history of those bins. Channel identity and ordering
will be explicit; a recording channel is not necessarily one stable neuron.
Initial settings are provisional: 20 ms bins and ten bins of history (200 ms).
Future spike bins must not influence an earlier prediction.

The output will be two values: x and y cursor velocity, intended in mm/s once
source units are verified. The initial task estimates velocity over the
just-completed bin and emits a prediction after the bin closes. It does not
forecast future movement.

The initial label policy will use a backward position difference divided by bin
duration, using available position samples at or before each boundary, with
sample-age and gap checks. Position and velocity labels may be used for model
fitting on training data and for scoring predictions. Held-out decoder inputs
and state initialization must not include reference movement labels.

## Chronological evaluation

Partition recording time before fitting transforms or creating history windows.
The provisional split is 70% training, 15% validation, and 15% test, in that order.
Randomly shuffled time bins would mix nearby, correlated movement into training
and evaluation and risk an overly optimistic result.

Windows and target differences must stay within their own partition. Purge gaps
at boundaries covering history and target support, plus any later smoothing,
lag, or forecast horizon. Reset histories and model state at partition boundaries
and discontinuities; declare warm-up exclusions. Fit normalization and any
data-dependent channel selection only on training data.

Use validation to choose model settings and freeze the recipe before opening
test results. Compare every model on the same eligible scoring timestamps.
Planned primary metrics are per-axis R² and RMSE, with explicit mean/overall
aggregation and physical units. Negative R² is a valid outcome. Save source,
configuration, split, channel order, transforms, seed, and code identity so
results can be reproduced. These policies still require implementation and
meaningful verification with later components.

## Planned model comparison

| Model | Purpose |
| --- | --- |
| Zero velocity and training-mean velocity | Establish simple reference predictions without held-out-label access. |
| Ridge regression with causal history | Test a regularized linear mapping from spike features to velocity. |
| Linear Gaussian Kalman decoder | Combine a velocity dynamics model with current neural observations through a causal filter. |
| Small unidirectional GRU | Test a nonlinear recurrent mapping over bounded causal histories. |

The initial GRU plan resets hidden state for each bounded window and uses the
same policy during replay. Kalman state initialization will use training
statistics. A more complex model is not assumed to outperform ridge.

## Visual replay, runtime, and robustness

The planned exportable 60–90 second demonstration will show synchronized actual
and decoded cursor positions, trails, matching axes, physical units, and elapsed
time. Actual position comes from the tracker. Decoded position integrates
predicted velocity using actual bin durations, starting from the same known
initial position with that initialization labeled. Subsequent reference
positions must not correct the decoded trajectory; any reset must be disclosed.

Develop rendering on synthetic or validation predictions. Preselect one
continuous held-out segment before viewing its prediction quality and retain
its source, bounds, configuration, seed, and checkpoint identifiers. Label the
clip **Offline replay of held-out neural recordings**. Use saved predictions
from the shared inference path rather than a separate demo decoder. The story
will explain the experiment, compare implemented models on the same segment,
show a predefined channel-loss example, and finish with full-test metrics and
limitations. Unavailable comparisons will remain labeled planned.

Velocity metrics are primary because integration accumulates position error.
Playback speed and animation rendering are separate from CPU inference latency.
Later measurements will report model-only and feature-plus-model processing,
with hardware, warm-up, and latency percentiles; acquisition time is additional.
No latency target has been achieved or measured yet.

Planned controlled robustness experiments include channel loss, count-valid
noise/rate drift, and missing packets. Freeze perturbation settings using
validation, preserve clean comparisons, and distinguish missing observations
from genuine zero-spike bins. Simulated corruption is not evidence of biological
drift or cross-session generalization.

## Limitations and success criteria

One recorded monkey session cannot establish performance across animals,
sessions, humans, or implanted clinical systems. Nearby time bins remain
correlated even with chronological splits and boundary gaps. Tracker noise,
time alignment, channel selection, and differentiation can affect labels and
scores. Spike counts also only approximate the Kalman model's Gaussian
observation assumption.

Offline replay does not demonstrate live closed-loop control or hard real-time
guarantees. A useful completed milestone will be a reproducible baseline with
honest held-out velocity scores, verified causal timing, and stated failure
modes. Broader completion will add the planned comparisons, replay agreement,
measured runtime, robustness experiments, and an exportable demonstration.
None of those scientific outcomes is claimed by this documentation checkpoint.
