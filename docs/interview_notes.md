# Interview notes

Practice notes tied to the [frozen final report](../reports/phase7b_final.md),
not a record of demonstrated mastery or a claim of live BCI control.

## A 60-second explanation

NeuroDecode estimates x/y hand-linked cursor velocity from recorded intracortical
spikes during a monkey reaching task. The monkey's tracked hand controlled the
original cursor; that tracker supplies our movement labels. We count events in
20 ms bins, apply a training-only scaler, and compare trivial baselines, ridge,
Kalman and a small GRU. Ridge and GRU use ten-bin histories; replay preserves the
same causal inputs and reset rules as offline evaluation. Settings were selected
on chronological validation before freezing test artifacts and reporting rules.
On 2,784 common held-out rows, ridge lambda 1000 achieved 60.421673 mm/s overall
RMSE, versus GRU 67.360708 and Kalman 134.059526. Independent metric and identity
audits passed. A timestamp-selected cursor replay integrates velocity without
position corrections, so drift remains visible. This is one-session offline
evidence, not cross-session transfer, human performance or closed-loop control.

## Five likely questions

### 1. What information does one prediction use, and why is it causal?

At a completed neural-bin end, ridge/GRU have the last ten consecutive raw count
vectors in fixed channel order, normalized by frozen training statistics.
Ridge uses oldest-to-newest flattened features. GRU starts that independent
window from zero hidden state; it does not carry a hidden state between windows.
Kalman uses the current standardized count vector and its own previous estimated
velocity/covariance, resetting to a training prior at boundaries/gaps. The shared
warm-up makes the scored rows identical. Movement values are absent from decoder
inputs. Counts/targets use no samples after their stated endpoints; the task
estimates just-completed-bin velocity, not a future movement forecast.

### 2. Why did ridge beat GRU here?

**Measured:** ridge has lower frozen overall test RMSE and higher mean R² on the
same rows. Both models receive the same ten-bin history, so ridge also uses
temporal information. The selected GRU's extra nonlinear capacity did not yield
better held-out performance in this declared experiment.

**Hypotheses:** ridge's regularization may suit this dataset and its correlated
examples; the small GRU's bounded training/search budget or generalization gap
may limit it. More capacity is not automatically useful. No controlled ablation
separates these explanations, and the result does not establish that GRUs are
generally inferior. Do not add epochs/settings after test to make GRU win.

### 3. Why did Kalman fail?

**Measured:** Kalman is worse than zero/training mean on this block, with negative
R² on both axes. Earlier validation diagnostics also showed bias/excess spread;
the numerical implementation and offline/replay agreement passed checks.

**Hypotheses:** stationary linear velocity dynamics, a current-bin linear neural
observation model, and diagonal/no-cross-time observation-noise assumptions may
be inadequate. Large covariance-system conditioning is a disclosed diagnostic,
not proof of the cause. Correct filtering arithmetic does not validate those
modeling assumptions. This comparison does not identify a biological mechanism
or establish that Kalman decoding generally fails. Keep the poor result visible.

### 4. What leakage did you prevent, and what do the scores mean?

Chronological partitions precede windows/fits; purges and within-segment support
checks prevent histories/targets crossing boundaries. Only training bins fit the
scaler; only training labels fit models, the mean baseline and Kalman prior.
Validation chooses settings/checkpoints; frozen test labels score predictions.
Prediction functions never receive movement values, and model/scaler digests are
unchanged by evaluation. The inherited preparation supplies quality flags; this
is not a deployed label-free acquisition quality detector.

RMSE has mm/s units and measures velocity error; R² compares squared error with
the scored axis's own mean reference. That mean is a metric denominator, not a
held-out predictor input. Positive R² does not prove position tracking: integrating
velocity errors can drift. Overlapping time windows are correlated, so these
scores are not independent-trial significance evidence.

### 5. What experiment would test cross-session generalization next?

Predeclare a **separate recording session**, e.g. another day, and verify source,
units/timing and channel correspondence without selecting channels by favorable
held-out scores. Same channel names do not prove the same neural population.
First test the frozen session-A model/scaler with a fixed common-row protocol on
untouched session-B data, reporting all declared models and missing-channel rules.

If supervised recalibration is a separate candidate, reserve a labeled prefix
and separate development data for it; predeclare its budget and score only the
remaining untouched session-B segment. Do not normalize on the whole new session
or choose a favorable block after inspecting errors. Another block from the
original recording only strengthens within-session evidence. This next experiment
is proposed, not implemented or authorized in Phase 7C.

## Defensible wording

“Compared frozen decoders on one chronological held-out block, audited replay
equivalence and exported an offline cursor visualization” is supported.
“Built a live brain-controlled cursor,” “proved human/clinical utility,” and
“ridge tracks position accurately because R² is positive” are not supported.

The [reproduction guide](reproduction.md) states what a fresh clone can run and
which original ignored artifacts remain unavailable as a public download. Be
equally explicit about that limitation in a technical walkthrough.
