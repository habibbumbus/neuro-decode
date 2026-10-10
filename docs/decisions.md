# Design decisions

## 2026-10-10 — Phase 2A timing and chronological splits

Use 20 ms bins, ten bins of history, zero additional lag, and zero forecast
horizon. Estimate completed-bin displacement rate using stored cursor positions
selected at or before each boundary. Fix maximum sample age and behavioral-gap
threshold at 6 ms, based on source-documented 4 ms sampling. Preserve source time
origin and channel/unit identities; hardware alignment remains unresolved.

Use nominal 70/15/15 fractions of elapsed shared coverage. Do not choose
boundaries from movement quality or future performance. Purge the maximum of
neural history and target-source lookback on each side of both boundaries,
currently 200 ms per side. Require independent full warm-up inside each retained
partition/continuous segment. Purging is a conservative support guard, not proof
of temporal independence.

Use half-open partition/segment ownership, including exclusion of the terminal
recording endpoint as an emission. Behavioral intervals >6 ms (+1 ns round-off
tolerance) split continuity. Never bridge gaps. The tolerance permits neither
future behavioral samples nor expanded partition ownership. Reset future model
state/history at session, partition, and segment starts.

Keep the reviewed Phase 1C interval [1288, 1298) s inside training. Save a small
source-bound specification now; exact eligible indices remain pending. No count
bins, velocity arrays, learned preprocessing, or model-state implementation are
included. See [methods.md](methods.md) for support formulas, endpoint conventions,
reproduction, rationale, and limitations.
