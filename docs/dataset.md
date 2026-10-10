# Dataset provenance and acquisition

## Experiment and selected recording

Implanted electrodes recorded neural spikes while a monkey physically moved its
hand in a reaching task. Tracked movement determined cursor position. These are
intracortical recordings, not EEG. NeuroDecode will estimate cursor velocity
from neural recordings afterward; tracker-derived behavior supplies labels.

Selected **indy_20161005_06.mat** from the explicitly requested
[Zenodo record 583331](https://zenodo.org/records/583331), **v1**, published
May 26, 2017. The record page offers newer versions; this acquisition retained v1.
The API has no explicit version string, so the v1 label comes from the record page.
The file has been acquired and its raw schema validated in Phase 1B.
Downstream modeling suitability remains unverified.

## Provenance, license, and citation

[Official metadata API](https://zenodo.org/api/records/583331) provided the
filename, exact byte size, published MD5, creators, publication date, and license.
The small [manifest](../data/manifest.json) records these details and local hashes;
the original recording remains outside Git.

| Item | Verified value |
| --- | --- |
| Record/version | 583331 / v1 |
| DOI | 10.5281/zenodo.583331 |
| File | indy_20161005_06.mat |
| Published and local size | 83,957,398 bytes (about 84 MB) |
| Published MD5, matched locally | 5ea300952642e0fc54245144499db9bb |
| Local SHA-256 | 4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac |
| Acquisition date (UTC) | 2026-10-09 |
| Relative destination | data/raw/indy_20161005_06.mat |

Version-specific [download link](https://zenodo.org/records/583331/files/indy_20161005_06.mat?download=1)
and [API content link](https://zenodo.org/api/records/583331/files/indy_20161005_06.mat/content)
identify the same selected file. No complete archive or broadband supplements
were downloaded.

The source API specifies **CC BY 4.0**. Preserve author/source attribution, link
[the license](https://creativecommons.org/licenses/by/4.0/), and identify changes
when distributing adaptations. The downloaded source bytes were not modified.
The dataset license does not choose a license for NeuroDecode's code; that choice
remains pending.

Citation:

O'Doherty, Joseph E., Cardoso, Mariana M. B., Makin, Joseph G., & Sabes, Philip N. (2017). Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology (v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.583331

## Source-documented field expectations

These are expectations from the [source description](https://zenodo.org/records/583331),
**not facts verified inside this file**. Here k is behavioral samples, n is
recording channels, and u is unit entries.

| Field | Documented role and logical shape |
| --- | --- |
| t | k × 1 timestamps for behavior samples, in seconds. |
| cursor_pos | k × 2 x/y cursor positions, in millimeters. |
| spikes | n × u cell array containing spike-event time vectors, in seconds. |
| chan_names | n × 1 cell array of channel identifier strings. |

The first spike-unit entry contains residual unsorted threshold crossings, not
a duplicate of all sorted units. The source also lists waveform snippets; they
are not needed for this acquisition or the planned timestamp-based decoder.
The verified storage layout, counts, timing, and remaining alignment uncertainties
are documented below. Do not infer a particular session's channel count from the
source's general description or silently transpose arrays.

## Verified file structure and loader — Phase 1B, October 10, 2026

The original bytes still match the manifest's SHA-256. The header identifies
MATLAB 7.3, and h5py confirms HDF5. Top-level keys are `#refs#`, `chan_names`,
`cursor_pos`, `finger_pos`, `spikes`, `t`, `target_pos`, and `wf`.
Only metadata was inspected for `wf`; waveform payloads are not loaded.

| Field | Stored HDF5 shape/type | Returned representation |
| --- | --- | --- |
| t | (1, 93501), float64, MATLAB double | behavior_times_s: (93501,), seconds |
| cursor_pos | (2, 93501), float64, MATLAB double | cursor_position_mm: (93501, 2), x/y millimeters |
| chan_names | (1, 96), MATLAB cell/object references | channel_ids: tuple of 96 strings in source order |
| spikes | (5, 96), MATLAB cell/object references | spike_times_s[channel_index][unit_index], variable-length event arrays in seconds |
| finger_pos | (3, 93501), float64 | inspected for source-mapping consistency only; not returned |
| target_pos | (2, 93501), float64 | metadata only; not returned |
| wf | (5, 96), references | metadata only; not returned |

Each channel-name reference points to a (6, 1) uint16 MATLAB char dataset. The
loader decodes UTF-16 code units. Each spike-cell reference points either to a
(1, event_count) double vector or to a MATLAB-encoded empty array. The 231 empty
trains use a (2,) uint64 dimension vector [0, 0] with `MATLAB_empty=1`; these
zeros are not spike timestamps. A null reference is missing/unsupported data,
not interchangeable with this explicit empty encoding.

MATLAB cells can hold different-length vectors. HDF5 references point to these
individual objects, so reading the cell table alone does not read spike values.
See [h5py reference documentation](https://docs.h5py.org/en/stable/refs.html).

Dimension reordering is deliberate: the source defines positions as samples × 2,
and the stored array is 2 × samples with the same sample count as `t`. Thus only
`cursor_pos` is transposed to return [T, 2]. The 96 channel-name entries establish
that the second stored spike axis is channels; cell [unit, channel] maps to
returned [channel][unit]. The loader accepts this observed layout only rather
than guessing orientations for arbitrary MAT files.

### Timing, identities, and event observations

- Behavioral sample count: **93,501**; range **1288.0–1662.0 seconds**;
  elapsed span **374.0 seconds**. The original nonzero time origin is retained.
- Sampling intervals: min 0.0039999999999054126 s, median
  0.0039999999999054126 s, mean 0.004 s, max 0.004000000000132786 s,
  standard deviation 1.1207100615686515e-13 s. This agrees with the source's
  documented 250 Hz sampling to floating-point precision.
- No nonfinite behavioral times/positions, duplicate behavior timestamps, or
  decreasing behavior times. No intervals exceed 1.5 × the observed median
  (approximately 6 ms). This is a gap-reporting heuristic, not proof of perfect
  acquisition or a future data-quality exclusion rule.
- Channels: **M1 001 through M1 096**, exactly in that order, unique. Names
  identify recording channels; they do not establish stable individual neurons.
- Five unit slots per channel: Python indices 0–4 correspond to MATLAB units 1–5.
  Index 0 is source-documented residual unsorted activity. Sorted slots are not
  aggregated and are not asserted to persist across sessions.
- Total events **323,237**. Slot totals, in source order:
  **128,756; 84,879; 81,701; 23,654; 4,247**. These are audit totals, not binned
  model features. Empty trains per slot: **11, 26, 36, 71, 87** (231 total).
- Eleven all-empty channels are preserved: M1 009, 047, 055, 058, 061, 068, 071,
  078, 079, 082, 096. Channel totals range from 0 to 12,622 events.
- No nonfinite spike values, descending spike times, or within-train duplicate
  timestamps were observed. The loader permits and explicitly reports duplicate
  spike events in other inputs without dropping them.
- Spike coverage: **1283.0023444054618–1666.9959174171197 seconds**, a span of
  about 383.994 s. **5,454** events precede the first behavior sample and **5,364**
  follow the last. All are retained. Future alignment must choose shared coverage
  explicitly; the loader does not truncate or zero timestamps.

### Units and unresolved issues

Seconds for timestamps/events and millimeters for cursor coordinates come from
source documentation. Relevant HDF5 datasets carry MATLAB class attributes but
no explicit unit attributes. Numerical magnitude alone cannot establish units.
The observed behavioral intervals agree with documented sampling, but do not
independently calibrate physical positions or prove neural/behavior synchronization.

As an optional consistency check, comparing cursor positions with the source's
simple `-10 * finger_pos[1:3]` transform gave a maximum absolute difference of
**3.038030960232419 mm**. Per-axis difference standard deviations were about
0.361502 and 0.299662 mm. The difference is not a fixed translation. Its cause
(e.g. processing or timing differences) is unresolved, not inferred. The loader
returns stored `cursor_pos` unchanged rather than reconstructing it from fingers.
The relationship warrants later inspection if used; no coordinate repair was made.

Empty cells encode absence of events, but the file alone does not explain why a
channel/slot is empty. A unit slot is not proof of a detected neuron. Exact
hardware synchronization, biological lag, calibration, and downstream usefulness
remain unverified even though this raw schema validates.

### Returned structure and reproduction

`load_recording(path)` returns a `Recording` with behavior timestamps,
positions, channel identifiers, nested spike vectors, and timing/coverage
`diagnostics`. For example, actual loaded values are:

```text
behavior_times_s[:3] = [1288.0, 1288.004, 1288.008]
cursor_position_mm[0] ≈ [-7.566217, 30.195105]
channel_ids[0] = "M1 001"
spike_times_s[0][0][:3] ≈ [1285.477042, 1285.519067, 1285.548762]
```

The first listed spikes precede behavior coverage. They are event times, not
three measurements paired row-by-row with the three position samples.
Full channel/unit counts are available as
`recording.diagnostics["spike_counts_by_channel_unit"]`.

From the repository root, after normal uv setup:

```powershell
uv sync --locked
uv run --locked python scripts/inspect_recording.py data/raw/indy_20161005_06.mat
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
```

The summary script loads one file and prints observations; it is not a general
CLI or preprocessing pipeline. NumPy and h5py are the only new dependencies.
The loader fails on unsupported formats/layouts, missing/broken references,
nonfinite values, mismatched dimensions/channel counts, duplicate/decreasing
behavior timestamps, and decreasing spike timestamps. It preserves legitimate
empty spike trains, differing coverage, and behavioral gaps without sorting,
truncating, dropping, normalizing, or repairing measurements.

Nineteen loader tests use tiny synthetic HDF5 reference fixtures, including
MATLAB empty cells and a deliberately unreadable waveform link. They check
mapping/orientation, gaps/coverage preservation, and meaningful invalid inputs.
Together with the existing installation check, **20 tests passed**. The real
recording passed loader validation separately; it is not a test fixture. Lint,
formatting, and lockfile consistency checks passed.

## Recorded-data exploration — Phase 1C, October 10, 2026

From the repository root, generate the fixed exploratory figure with:

```powershell
uv run --locked python scripts/plot_recording.py
```

The script reuses `load_recording`, verifies the manifest SHA-256, and writes
`runs/phase1c/indy_20161005_06_recorded.png` plus a JSON record of source identity,
interval bounds, selection/gap rules, and train identities/event totals. These
outputs and Matplotlib's cache under `runs/.matplotlib/` are ignored. Existing
PNG/JSON files are not overwritten. To reproduce into a new named folder:

```powershell
uv run --locked python scripts/plot_recording.py phase1c-review
```

The already generated, visually reviewed final artifact is
`runs/phase1c-verified/indy_20161005_06_recorded.png`. Its interval is fixed
independently of movement quality:

1. Find overlap between behavioral coverage and the global spike-event span.
   Start at the first behavior timestamp at/after its start.
2. Use the half-open ten-second interval **[1288.0, 1298.0) s** here. All panels
   use exactly the same sample/event masks. Subtract 1288.0 s from both spike
   and behavior times for display; no neural/behavior offset is introduced.
3. Require the interval to end within the provisional first 70% of behavioral
   elapsed time (end 1549.8 s here). This guard keeps presentation exploration
   early; it does not create or finalize evaluation partitions. No later held-out
   portion was visualized or selected for presentation.
4. Choose the first four channel identifiers in lexical order and their first
   two unit slots, regardless of activity or apparent movement correlation.
   Here these are M1 001–004, U1 and U2. U1 is residual unsorted activity; U2 is
   a sorted slot that may be empty. Rows are trains, not necessarily neurons.
5. Refuse a behavioral gap intersecting the fixed interval, using the loader's
   1.5 × median interval heuristic. Do not search for a replacement clip or
   interpolate across gaps. No such gap occurs in this interval. Spike silence
   does not identify missing packets; global event coverage alone cannot prove
   uninterrupted neural acquisition.

Read the panels as follows:

- **Raster:** each tick marks an event in an identified train. Eight rows show
  270 events in total; interval counts are 59, 0, 43, 55, 48, 47, 18, 0 in
  channel/unit order. Both empty rows are retained and labeled "No events in
  interval," which does not mean missing behavior samples.
- **Position versus time:** blue x and orange y show 2,500 recorded position
  samples in millimeters. Time is shared with the raster. Connected adjacent
  samples are displayed without smoothing or added samples.
- **x-y trajectory:** the same positions form a spatial path with equal axis
  scaling. Green circle marks the first sample; red X marks the last included
  sample at 1297.996 s, not the excluded endpoint 1298.0 s.

Selected x positions range from about -50.364 to 52.199 mm, and y positions from
29.046 to 104.817 mm. The cursor changes position across several portions of the
clip, and events occur in multiple selected trains. These are descriptions of
recorded data only. No signals were shifted until they looked aligned. The
unresolved finger/cursor mapping discrepancy from Phase 1B remains unresolved.
Temporal co-occurrence is not proof of causation, neural/behavior synchronization,
or decoder accuracy. There are no model predictions, spike bins, derived velocity,
normalization, or engineered features in this figure.

Matplotlib 3.11.2 was added through uv with a generated lockfile update. Nine
synthetic tests cover common half-open bounds, shared time offsets, identity
mapping, empty trains, movement-independent selection, overlap start, gap refusal,
and the provisional training guard. All 29 tests (new and existing) passed, with
lint, format, and lockfile checks. Visual inspection confirmed legible labels,
shared times, empty rows, equal trajectory scaling, and start/end markers.

## Reproduce acquisition and verification (Windows PowerShell)

Run from the repository root. This downloads only the selected file when absent,
verifies existing bytes before reuse, and refuses to overwrite an existing partial
file. curl is the Windows executable; hashing uses built-in PowerShell commands.
No scientific dependencies are required.

```powershell
$ErrorActionPreference = 'Stop'
$recording = 'data/raw/indy_20161005_06.mat'
$partial = "$recording.part"
$url = 'https://zenodo.org/records/583331/files/indy_20161005_06.mat?download=1'
$expectedSize = 83957398
$expectedMd5 = '5ea300952642e0fc54245144499db9bb'
$expectedSha256 = '4b023e668d641960c7e3f4d277c781e606013125013a4cead1f4731951ff8eac'
New-Item -ItemType Directory -Force -Path data/raw | Out-Null

if (Test-Path -LiteralPath $recording) {
    $candidate = $recording
} else {
    if (Test-Path -LiteralPath $partial) { throw 'Inspect existing partial file before retrying' }
    curl.exe --fail --location --retry 2 --connect-timeout 30 --max-time 300 --output $partial $url
    if ($LASTEXITCODE -ne 0) { throw 'Download failed; partial file retained' }
    $candidate = $partial
}

$size = (Get-Item -LiteralPath $candidate).Length
$md5 = (Get-FileHash -LiteralPath $candidate -Algorithm MD5).Hash.ToLowerInvariant()
$sha256 = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
if ($size -ne $expectedSize -or $md5 -ne $expectedMd5 -or $sha256 -ne $expectedSha256) {
    throw 'Verification failed; do not use or overwrite these bytes'
}
if ($candidate -eq $partial) { Move-Item -LiteralPath $partial -Destination $recording }
[pscustomobject]@{size_bytes=$size;md5=$md5;sha256=$sha256}
```

If automated access fails, open the version-specific record page, select only
indy_20161005_06.mat, save it at the relative destination above, and rerun the
verification block. Do not select the newer version, download-all option, or
broadband supplement. A partial/error response is not a verified acquisition.

The initial download used curl to a .part file, checked size/MD5, computed SHA-256,
and renamed only after successful verification. Both hashes were independently
recomputed on the final filename with Python's standard library. Matching MD5
checks agreement with the publisher's bytes; local SHA-256 supplies a stronger
fingerprint for later reproduction. Neither proves timestamp alignment, correct
units, scientific suitability, or trustworthiness of the source itself.

## Git and scope limits

```powershell
git check-ignore -v data/raw/indy_20161005_06.mat data/processed/probe.npy
git check-ignore -v data/manifest.json docs/dataset.md
git ls-files -- data/raw/indy_20161005_06.mat
```

The first command should show shared ignore rules; the second should print
nothing (exit code 1 means neither small file is ignored). The last must print
nothing. Commit only the manifest/documentation, never recording bytes.

Phase 1C adds recorded-data plots using the validated loader. Feature bins,
velocity targets, evaluation splits, and training remain unimplemented.
The data come from recorded nonhuman-primate behavior; no human or live
closed-loop performance claim follows. The next increment will establish chronological split boundaries and the causal
timing contract before spike-count features or velocity targets are generated.
