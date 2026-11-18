"""Read-only Phase 5B artifact audit; no recording load, training or test access."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from neurodecode.data.features import FrozenCountScaler
from neurodecode.models.gru import GRUDecoder


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def audit(directory):
    """Check saved hashes, independently recompute CSV metrics and bound head output."""
    root = Path("runs").resolve()
    directory = directory.resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Audit only trusted project runs")
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    for relative, expected in metadata["artifact_sha256"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Artifact must stay within run directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed saved artifact: {relative}")
    scaler = FrozenCountScaler.from_metadata(metadata["normalization"])
    reference = read_rows(directory / "predictions.csv")
    identity_fields = (
        "partition",
        "segment_id",
        "bin_id",
        "time_s",
        "score_mask",
        "first_source_bin_id",
        "last_source_bin_id",
        "actual_vx_mm_s",
        "actual_vy_mm_s",
    )
    results = []
    for record in metadata["records"]:
        if record["status"] != "complete":
            continue
        path = (directory / record["run"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Checkpoint must stay within trusted experiment")
        rows = read_rows(path / "predictions.csv")
        if len(rows) != len(reference):
            raise ValueError("Seed prediction count changed")
        for row, previous in zip(rows, reference, strict=True):
            if any(row[key] != previous[key] for key in identity_fields):
                raise ValueError("Seed scored identity/target changed")
        actual = np.array(
            [[float(row[f"actual_v{a}_mm_s"]) for a in ("x", "y")] for row in rows]
        )
        prediction = np.array(
            [[float(row[f"gru_v{a}_mm_s"]) for a in ("x", "y")] for row in rows]
        )
        mse = np.square(prediction - actual).mean(axis=0)
        r2 = 1 - np.square(prediction - actual).sum(axis=0) / np.square(
            actual - actual.mean(axis=0)
        ).sum(axis=0)
        metrics = record["validation_metrics"]
        np.testing.assert_allclose(
            np.sqrt(mse), metrics["rmse_mm_s"], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            np.sqrt(mse.mean()), metrics["overall_rmse_mm_s"], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(r2, metrics["r2"], rtol=0, atol=1e-12)
        history = json.loads((path / "history.json").read_text(encoding="utf-8"))
        earliest = min(
            history, key=lambda h: h["validation_metrics"]["overall_rmse_mm_s"]
        )
        if earliest["epoch"] != record["best_epoch"]:
            raise ValueError("Checkpoint does not follow earliest minimum rule")
        model = GRUDecoder.load(path / "gru.pt", expected_scaler=scaler)
        # h0=0 and GRU's convex blends of tanh candidates keep each h in [-1,1].
        # Therefore the fixed linear head cannot exceed bias +/- sum(abs(weights)).
        width = model.head.weight.detach().abs().sum(dim=1).numpy().astype(float)
        bias = model.head.bias.detach().numpy().astype(float)
        lower, upper = bias - width, bias + width
        outside = (actual < lower) | (actual > upper)
        results.append(
            {
                "run": record["run"],
                "rows": len(rows),
                "independent_csv_metrics": "agree within 1e-12",
                "head_output_lower_bound_mm_s": lower.tolist(),
                "head_output_upper_bound_mm_s": upper.tolist(),
                "target_outside_head_range_count_xy": outside.sum(axis=0).tolist(),
                "target_outside_head_range_fraction_xy": outside.mean(axis=0).tolist(),
                "actual_min_mm_s": actual.min(axis=0).tolist(),
                "actual_max_mm_s": actual.max(axis=0).tolist(),
                "validation_epoch1_rmse_mm_s": history[0]["validation_metrics"][
                    "overall_rmse_mm_s"
                ],
                "validation_last_rmse_mm_s": history[-1]["validation_metrics"][
                    "overall_rmse_mm_s"
                ],
            }
        )
    return {
        "scope": "artifact-only audit; no refitting, recording load or test access",
        "audit_implementation_sha256_utf8_lf": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").encode()
        ).hexdigest(),
        "run_json_sha256": hashlib.sha256(
            (directory / "run.json").read_bytes()
        ).hexdigest(),
        "all_recorded_artifact_hashes": "verified",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    directory = Path("runs") / args.run_id
    result = audit(directory)
    with (directory / "verification.json").open("x", encoding="utf-8") as file:
        file.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
