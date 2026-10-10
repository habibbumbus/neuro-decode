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
The file is available and manageable in size. Actual modeling suitability is
pending internal inspection.

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
The actual storage representation, dimensions, channel count/regions, unit count,
duration, timestamp origins, sampling coverage, missingness, gaps, and alignment
are all **pending Phase 1B inspection**. Do not infer a particular session's
channel count from the source's general description or silently transpose arrays.

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

Phase 1A verifies identity and acquisition only. No MATLAB loader, feature bins,
velocity targets, evaluation splits, training, or visualization exists yet.
The data come from recorded nonhuman-primate behavior; no human or live
closed-loop performance claim follows. The next increment is **Phase 1B: inspect
the actual file structure and build a minimal validated loader**.
