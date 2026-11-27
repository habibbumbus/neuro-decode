# Phase 7D: final repository and presentation audit

**Local repository/results checks pass with a documented Windows fixture-storage
workaround. Final sign-off remains open:** rendered-README visual review is still
unverified and no hosted CI workflow/run exists. The original scope is therefore
**not marked fully complete**. No scientific setting, frozen identity, model,
metric or selected interval changed; no real-data training or test experiment.

## Verified checkpoint and repository contents

Phase 7C HEAD and fresh GitHub main lookup both returned
`3343eebd3788abad4ec35362a98b1f5f6cba66b8`. Tracked tree/index were clean on entry;
unrelated untracked `%SystemDrive%/` remains preserved. CODEX.md and applicable
ancestor guidance were checked; no AGENTS.md found. History was inspected without
rewriting it. No agent stage/commit/push.

Public ref inventory contains main and docs/project-foundation, both represented
in the inspected local history; no public tags were listed. No extra public
branch was silently omitted from the inventory.

Read-only inventory: 103 tracked files, 433 reachable Git objects, 228 unique
reachable blobs. No forbidden raw/processed/runs/private-brief/model/video/key
paths in current tracked files or reachable history; no blob exceeds 5 MB.
The only tracked data files are data/manifest.json and the chronological split
JSON, both provenance/partition metadata, not processed movement examples.
No high-signal credential matches across 213 UTF-8 text blobs. This is a pattern
scan plus file inventory, not proof against every possible secret representation
or unreachable/unfetched objects. No credential values are reproduced here.

All nine tracked PNGs match original ignored run PNGs byte-for-byte. They were
visually inspected for labels, units, selected intervals and captions: four
declared GRU learning curves, the five-model validation trace, three validation
disruption figures and the actual predetermined test still. Learning-rate/seed/
selected-epoch labels match their reports. Validation disruption plots are not
presented as test robustness. The banner's source explicitly labels its graphic
conceptual rather than measured data. No new figures or replacement clips.

Preview SHA-256 remains
`4c017df3e63b1ae99b398f87174e1f51ec65a84926d3c404d1c5ab0907a9a1bb`.
All 29 bound implementation hashes and the original test run JSON remain intact;
the original test run SHA-256 is
`87c10c06a2635a3b506b693cf2d024c3cd39ecadf971a521ecea32f803c94bc2`.
Evidence is under ignored runs/phase7d-final-audit/.

## Reader-facing scientific claims

README correctly describes invasive intracortical spike recordings during a
monkey reaching task, hand-linked tracker-derived movement labels, completed-bin
velocity estimation and offline replay. It explicitly distinguishes EEG and
live cursor control, keeps the chronological split and training-only transforms,
and explains nine-bin warm-up/common scoring rows. The final table was checked
against every axis/overall RMSE and axis/mean R² row of the Phase 7B report.

| Model | Frozen held-out overall RMSE [mm/s] | Mean R² |
| --- | ---: | ---: |
| Zero | 83.143527 | -0.000027 |
| Training mean | 83.143644 | -0.000029 |
| Ridge lambda 1000 | 60.421673 | 0.464330 |
| Kalman | 134.059526 | -1.538225 |
| GRU seed 2026 | 67.360708 | 0.338009 |

All five models use the same 2,784 eligible test rows, target ends
1606.320–1661.980 s, 100% eligible-row coverage. Ridge is best by the frozen
overall velocity RMSE on this block; the result does not establish accurate
integrated position tracking or transfer to another session. Poor Kalman results
and GRU underperformance remain visible. No significance, live-control,
human/clinical or hard real-time guarantee is claimed. The proposed separate-
session experiment and explanations of failure are clearly hypotheses/plans.

## Fresh-clone walkthrough and failures retained

Created a fresh local clone of Phase 7C under
runs/phase7d-final-audit/clone/, with no raw recording, original model/evidence
runs or private local brief. New locked Python 3.11.17 environment installed 34
packages from the populated cache, including clone-local editable NeuroDecode.
Isolated import confirms both interpreter and package resolve to the clone.
This is a new environment on the same Windows machine, not a fresh-host or
non-Windows claim.

| Step | Fresh clone outcome |
| --- | --- |
| Locked cached setup and isolated package import | Passed with a distinct clone-local environment |
| Synthetic tests | Passed after the fixture-storage correction described below |
| Ruff lint/format and lock consistency | Passed |
| Help/entry-point/config paths | Consistent with actual scripts; no fitting commands executed |
| Frozen final preflight without ignored artifacts | Expected stop at missing runs/phase3b-ridge/ridge.json, before any recording loader |
| Recording inspection or real-data inference | Not attempted; requires separately downloaded verified raw input |
| Exact historical validation/test inference | Not attempted; requires original ignored model/reference/evidence bundle and frozen export tools |

In the original checkout with restored inputs, local frozen preflight passed
without source loading. The documented read-only final artifact audit also
passed on all 2,784 saved rows and confirmed run SHA-256 87c10c06…94bc2, unchanged
from Phase 7B. It recomputes existing saved metrics and identities; it does not
construct recording examples or perform another model inference experiment.

The exact default test command exposed an environment issue: **300 passed,
53 fixture setup errors in 97.69 s**, all from access denied to the shared
Windows pytest temp directory. No model/test assertion failed in those setup
errors. The first workspace-temp retry retained the same 53 setup errors because
the fresh clone had no parent runs/ directory (300 passed in 29.62 s). That
workaround omission was corrected in the documentation; no permissions, source
or pytest configuration were changed.

With the parent created and a new absent workspace-local base directory, the
same complete synthetic suite passed: **353 passed in 34.04 s**. This rerun is
unit/fixture verification, not another held-out experiment. Lint/format and lock
consistency passed. The original errors and retry are disclosed rather than
silently replaced by the final pass. Exact successful retry from the clone:

```powershell
if (Test-Path -LiteralPath runs/phase7d-pytest) { throw 'Preserve existing fixture directory' }
New-Item -ItemType Directory -Force runs | Out-Null
.venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp=runs/phase7d-pytest --tb=short
```

The [portable setup/reproduction guide](../docs/reproduction.md) supplies uv-based
commands and the temp-permission troubleshooting case. Preserve existing fixture
directories; pytest can clear its explicitly selected base directory. No raw
dataset was downloaded, models refitted or test predictions generated here.

## Concrete fixes and remaining limits

- README file map now uses complete src/neurodecode/ paths for training,
  streaming and integration modules.
- Demo documentation uses the existing held-out video and clearly separates
  validation disruption figures from test results; no new export is implied.
- Reproduction guide documents the Windows temp-directory failure and creates
  the parent before using a new ignored pytest base directory.

Public setup and synthetic checks are reproducible as above. Real-data
inspection/inference requires the separately downloaded public Zenodo v1 file.
Exact historical preflight/evaluation needs the original trusted ignored model/
reference/validation-evidence bundle and frozen Python/package/export tool
identities; that bundle and the MP4 have no published download release. Fresh
fits are not exact frozen hashes. Missing artifacts must not be replaced by
changing the freeze. Saved-artifact-only auditing is separately documented.

Manual rendered-README visual review remains open from Phase 7C's Computer Use
URL-confidence block. A new read-only Phase 7D inventory confirmed that the browser
access host remains unavailable. No UI bypass or claim of a passed screenshot
review was made. External link verification is also partial: Git transport
confirms the repository, the official uv installation page was retrieved, while
the web fetcher could not retrieve the GitHub page or follow the DOI to Zenodo.
Provider/cache restrictions are not evidence that those URLs are broken; no
valid source URL was rewritten based on such an error. No hosted CI run is
claimed; no workflow currently exists. Code-license choice remains pending.

## Project summary

The system estimates x/y hand-linked cursor velocity from recorded implanted
spike events, with causal count/history contracts, frozen decoders, matched
scoring, verified sequential replay, validation timing/disruption studies and
an uncorrected integrated-path visualization. It does not acquire live neural
data, close a feedback loop or establish human/clinical performance.

Strongest supported conclusion: ridge lambda 1000 has the lowest declared
velocity error among these frozen candidates on this one session/time block.
R² has no units; its positive value does not remove accumulated position drift
or prove generalization across recording sessions.

Start with [README](../README.md), [final report](phase7b_final.md),
[reproduction](../docs/reproduction.md) and [demo documentation](../docs/demo.md).
The public still is in
docs/assets/phase7b-replay-still.png; the full local MP4 is
runs/phase7b-final-test/cursor_replay.mp4 when the original evidence is restored.

Final sign-off remains open until the outstanding presentation/CI checks are
resolved. No further scientific experiment is implied by this audit.
