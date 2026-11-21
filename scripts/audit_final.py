"""Independent exported-metric/row/provenance audit; no recording loader or inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_gru import read_rows
from evaluate_final import canonical_sha
from replay_validation import sha, write_json

from neurodecode.final_evaluation import MODEL_ORDER, check_metrics


def audit(output):
    output = output.resolve()
    if not output.is_relative_to(Path("runs").resolve()):
        raise ValueError("Only trusted project evidence is audited")
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    freeze = protocol["freeze"]
    if run["status"] != "passed" or run["freeze_sha256"] != canonical_sha(freeze):
        raise ValueError("Missing successful status/exact freeze identity")
    for relative, expected in run["artifact_sha256"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or sha(path) != expected:
            raise ValueError(f"Changed evaluation evidence: {relative}")
    rows = read_rows(output / "predictions.csv")
    ids = np.array([int(r["bin_id"]) for r in rows])
    if len(set(ids)) != len(ids) or np.any(np.diff(ids) <= 0):
        raise ValueError("Duplicate/unordered row identity")
    split = json.loads(
        Path("data/splits/indy_20161005_06.json").read_text(encoding="utf-8")
    )
    if (
        canonical_sha(split)
        != freeze["json_inputs_canonical_sha256"]["data/splits/indy_20161005_06.json"]
    ):
        raise ValueError("Frozen split provenance mismatch")
    part = next(p for p in split["partitions"] if p["name"] == run["partition"])
    origin, width = split["bin_grid"]["origin_s"], split["bin_grid"]["width_s"]
    for row, bin_id in zip(rows, ids, strict=True):
        end, start = float(row["time_s"]), float(row["bin_start_s"])
        segment = part["continuous_segments"][int(row["segment_id"])]
        if (
            row["partition"] != run["partition"]
            or end != origin + bin_id * width
            or start != origin + (bin_id - 1) * width
            or int(row["first_source_bin_id"]) != bin_id - 9
            or float(row["first_source_start_s"]) != origin + (bin_id - 10) * width
            or not segment["start_s"]
            <= float(row["first_source_start_s"])
            < end
            < segment["end_s"]
            or not segment["start_s"] <= float(row["behavior_start_s"]) <= start
            or not float(row["behavior_start_s"]) < float(row["behavior_end_s"]) <= end
        ):
            raise ValueError("Row identity, timing or no-leakage support audit failed")
    actual = np.array(
        [[float(r[f"actual_v{a}_mm_s"]) for a in ("x", "y")] for r in rows]
    )
    for name in MODEL_ORDER:
        sequential = np.array(
            [[float(r[f"{name}_v{a}_mm_s"]) for a in ("x", "y")] for r in rows]
        )
        offline = np.array(
            [[float(r[f"offline_{name}_v{a}_mm_s"]) for a in ("x", "y")] for r in rows]
        )
        t = freeze["tolerances"][name]
        np.testing.assert_allclose(
            sequential, offline, atol=t["atol_mm_s"], rtol=t["rtol"]
        )
        check_metrics(actual, sequential, run["metrics"][name])
    integrity = json.loads((output / "integrity.json").read_text(encoding="utf-8"))
    if integrity["parameter_digests_before"] != integrity["parameter_digests_after"]:
        raise ValueError("Frozen parameters changed")
    return {
        "status": "passed",
        "run_json_sha256": sha(output / "run.json"),
        "freeze_sha256": run["freeze_sha256"],
        "partition": run["partition"],
        "rows": len(rows),
        "metric_check": "Python fsum from exported CSV, all five models, tolerance 1e-10",
        "identities_provenance_support": "verified; source-window/behavior times within frozen partition/segments; no future source sample",
        "scope": "artifact-only audit; no recording/test examples/inference loaded",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    output = Path("runs") / args.run_id
    verification = audit(output)
    write_json(output / "verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
