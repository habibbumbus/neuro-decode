# Phase 6C: validation-only visual cursor replay

Implemented a reproducible ten-second video and predetermined middle-frame
still from the verified clean sequential Phase 6A predictions. Reference,
ridge, representative GRU seed 2026 and Kalman are all shown with synchronized
timing, one fixed common physical scale, full trails and uncorrected drift.
The title says “Offline replay of recorded intracortical activity”; validation,
not-live-control, shared anchor and reference limitations stay visible.

## Alignment and integration audit

The interval remains [1550.200,1560.200) s, selected before viewing trajectories.
It contains 500 exact original prediction IDs 13110–13609, ends 1550.200–1560.180 s.
All saved targets, timestamps, source window IDs, segments, masks and source/
split/transform identities match established preparation. Models were not
rerun or tuned. Original target velocities reconstruct exactly from tracker
position differences over their selected behavioral intervals.

Exact original-position alignment is not established:40/500 behavioral
intervals last 16 or 24 ms, versus neural durations
0.01999999999998181–0.020000000000209184 s. Maximum source age/duration mismatch
is 4 ms; source endpoint chain breaks are zero. Integrating targets over neural
durations differs from sampled tracker endpoints by up to 0.822694453 mm per
coordinate. This is why the displayed reference is labeled **tracker-derived
reference velocity integration** rather than original tracker positions.

Every path starts at relative (0,0) mm at 1550.180 s, before the first selected
bin end. For every row use its actual bin_end − bin_start duration and add
velocity × duration to the previous integrated position. No later reference
position is an input or correction. There are 501 stored positions including
that anchor and 500 displayed endpoints, not an off-by-one shifted trace.
The endpoint frames span elapsed 0–9.98 s; at 50 fps the encoded duration is 10 s.
The 500 durations sum to 10 s from the anchor to the last endpoint.

Missing, nonfinite, unordered, cross-segment or discontinuous support is rejected
explicitly; no interpolation, smoothing, shifting, snapping, resets or alternate
clip selection. The actual selected clean interval has complete support.

## Export and inspection

Final outputs are under ignored runs/phase6c-validation-cursor-final:

| Artifact | Verified result |
| --- | --- |
| cursor_replay.mp4 | 488,786 bytes; H.264/yuv420p; 1600×900; 50 fps; 500 frames; 10.000000 s |
| still.png | 143,176 bytes; predetermined frame 250, elapsed 5.000 s |
| encoded_frame_01/02/03.png | Decoded MP4 frames 0/250/499, inspected visually |
| positions.csv / alignment.csv | All integrated positions and original timing audit |
| inputs.json / run.json / verification.json | Full provenance, export settings, hashes and checks |

Common axes remain x[-297.620224,304.783605] and y[-547.221219,55.182610] mm
throughout, with equal aspect. Kalman's poor drift is retained rather than
clipped or hidden. First/middle/last frames retain readable reference/model
labels, elapsed/bin-end time, physical units and integration explanation.
The initial runs/phase6c-validation-cursor export is preserved. A final export
only shortened sidebar text and capitalized GRU; position CSV bytes are identical.
There was no numerical, interval or model adjustment for presentation quality.

At the final endpoint, distance from the integrated reference is ridge 43.839497,
GRU 42.419501 and Kalman 441.346509 mm. These are illustrative integrated endpoint
distances, not a new model-ranking criterion. The screen preserves full-validation
velocity RMSE ridge 52.571380 < GRU 54.791919 < Kalman 102.838507 mm/s on 2775 rows.
A slightly closer GRU endpoint does not establish superior velocity accuracy.

## Verification and provenance

All 13 new focused tests passed: actual variable durations, completed-bin
timestamps, shared anchoring/input ownership, future-value independence,
missing/nonfinite bins, gaps/segments/order, tracker timing mismatch and invalid
future tracker samples. Full suite 332 passed (319 retained). Ruff lint/format,
offline lock consistency and locked environment checks passed. No dependency
was installed. Independent export audit checks all recorded hashes and confirms
every displacement against saved velocity × actual duration, maximum residual
2.842171e-14 mm. Common anchors are exact; all inspection axes match; still.png
has identical bytes to the selected middle render. FFprobe confirms encoded
frame count, dimensions, frame rate and duration.

Actual branch main, base 1bb05680123b9c95abb0772d3ddb3f9bb1737ed1, confirmed
on GitHub before work. Tracked tree/index were clean; unrelated %SystemDrive%/
and README work were preserved. New code/config were dirty/uncommitted during
rendering. Actual LF-normalized implementation hashes, Git status, source SHA,
complete split, target/scaler/channel identities, Phase 6A CSV/run and frozen
model hashes, selected IDs, durations, anchor, integration and encoder settings
are saved. Source remains indy_20161005_06, Zenodo 583331 v1. Python 3.11.17,
NumPy 2.4.6, Matplotlib 3.11.2; Pillow and installed FFmpeg/ffprobe versions are
recorded. FFmpeg build N-122609-g364d5dda91-20260131; libx264 medium, CRF 20.
Generated media and logs remain under ignored runs/; no media is intended for Git.

Reproduction and interpretation are documented in [demo instructions](../docs/demo.md).
Actual final commands used the existing environment:

~~~powershell
.venv/Scripts/python.exe -u scripts/render_cursor_replay.py phase6c-validation-cursor-final
.venv/Scripts/python.exe scripts/audit_cursor_replay.py phase6c-validation-cursor-final
~~~

Use a new run ID to reproduce. No fresh-clone/cross-platform export was tested.
This is precomputed validation playback, not real-time inference, closed-loop
control, original-position ground truth, clinical evidence or a latency benchmark.
Acquisition/behavioral timing limitations remain; numerical integration accumulates
velocity errors. Common global metadata loading still reads the acquired source
recording, but no test windows/targets/predictions/scores were constructed. No
model tuning, new disruption round or final-test opening. Stop after Phase 6C.
