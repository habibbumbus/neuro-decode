# Validation cursor replay — Phase 6C

This is a ten-second **offline replay of recorded intracortical activity**, not
live brain control. It uses the fixed first validation interval
[1550.200,1560.200) s from earlier figures, without selecting a favorable clip.
The test partition remains closed. No models are fitted, tuned or run by the
renderer: it consumes verified clean sequential Phase 6A predictions.

## Recreate the video and still

From the repository root with the locked Python environment, acquired recording,
saved Phase 6A inputs/checkpoints and existing FFmpeg/ffprobe on PATH:

~~~powershell
uv run --locked python scripts/render_cursor_replay.py phase6c-review
uv run --locked python scripts/audit_cursor_replay.py phase6c-review
~~~

Use a new run ID; existing media folders/verification files are refused.
The actual final run is runs/phase6c-validation-cursor-final. Its outputs are:

- cursor_replay.mp4: H.264, 1600×900, yuv420p, 50 fps, 500 frames, exactly 10 s.
- still.png: the predetermined middle frame, index 250 at 1555.200 s, elapsed 5 s.
- frame_first/middle/last.png and encoded_frame_01/02/03.png: source renders and
  actual decoded MP4 frames for visual inspection.
- positions.csv: common initial anchor plus 500 integrated endpoints, all paths.
- alignment.csv: original tracker source times/positions and actual bin durations.
- inputs.json: source/split/target/channel/prediction/model identities, hashes,
  alignment audit, settings, library/export-tool versions and actual Git/code state.
- run.json and verification.json: fixed axes, video probe, artifact hashes,
  trajectory audit and terminal distances.

Every generated file stays under ignored runs/. Do not stage the video, PNGs,
raw recordings, checkpoints or CODEX.md. The renderer uses existing matplotlib,
Pillow and FFmpeg; no Python dependency was added. Existing FFmpeg build is
N-122609-g364d5dda91-20260131. Fresh machines need their own compatible FFmpeg
installation; that setup was not tested here.

## What the reference and timing mean

The on-screen reference is explicitly **tracker-derived reference velocity
integration**, not the original tracker-position trace. All saved target
velocities match the original tracker differences divided by their actual
selected behavioral intervals exactly. But 40 of 500 behavioral intervals are
16/24 ms instead of the neural bins' approximately 20 ms, with source age up to
4 ms. Source endpoints chain correctly, yet integration over neural-bin durations
differs from original sampled tracker positions by up to 0.822694 mm per coordinate.
Therefore exact original-position alignment is not asserted.

For every path, displacement for bin i is velocity_i × (bin_end_i − bin_start_i).
Start from one relative origin (0,0) mm at 1550.180 s, the first selected bin's
start. The 500 selected prediction timestamps are 1550.200 through 1560.180 s,
all inside the fixed half-open interval. The first displayed point has already
integrated the first completed bin. The initial anchor is context, not an extra
frame outside the requested interval. The last frame's elapsed time is 9.98 s;
holding each of 500 frames for 1/50 s makes a 10-second video. Actual integrated
bin durations sum to 10 s from the pre-clip anchor to the last endpoint.

No smoothing, shift, snap, repeated anchoring, reference-position correction
or interpolation is applied. Missing bins, nonfinite values, unordered IDs,
nonshared edges or segment changes cause explicit failure, not an alternative
clip or a bridged path. The actual fixed clean clip has no such discontinuity.

## Reading the display honestly

Reference, ridge, seed-2026 GRU and Kalman have the same fixed physical-unit
axes and equal x/y scale, covering every complete path including Kalman drift.
Bounds are chosen once from these fixed validation paths and never change with
the frame; they do not enter decoder computation. Current distance from the
integrated reference is illustrative. Full-validation velocity RMSE remains
ridge 52.571, GRU 54.792, Kalman 102.839 mm/s on the original 2775 rows.

The shorter clip's final distances (43.8395/42.4195/441.3465 mm) do not establish
that GRU beats ridge: integrated errors can cancel, and one endpoint is not the
declared validation metric. An attractive trajectory does not prove decoding
accuracy. At inference time the models have completed neural bins and their
declared state/history, never movement labels or later reference positions.
Playback rate is unrelated to measured processing latency; see the Phase 6A
report. This video is not a new latency measurement or a live BCI demonstration.

The first, middle and last decoded video frames were visually checked for
labels, timing, common axes and visible drift. There were no cropped paths or
automatic recentering. The initial export was preserved; the final export only
tightened sidebar text/capitalization and has identical positions.csv bytes.
All 332 tests passed, including 13 new integration/alignment/gap checks.
Independent artifact audit verifies hashes, anchor equality and every exported
displacement against saved velocities (maximum residual 2.842171e-14 mm).
FFprobe verifies frame count, dimensions, rate and duration. No final-test score.
