"""Verify exported cursor positions and video metadata without model inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from replay_validation import sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    output = (Path("runs") / args.run_id).resolve()
    if not output.is_relative_to(Path("runs").resolve()):
        raise ValueError("Only trusted local replay outputs may be audited")
    metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    inputs = json.loads((output / "inputs.json").read_text(encoding="utf-8"))
    for filename, expected in metadata["artifact_sha256"].items():
        path = (output / filename).resolve()
        if not path.is_relative_to(output) or sha(path) != expected:
            raise ValueError(f"Changed export artifact: {filename}")
    source_csv = Path(inputs["settings"]["prediction_run"]) / "predictions.csv"
    if sha(source_csv) != inputs["predictions_csv_sha256"]:
        raise ValueError("Saved source predictions changed")
    with source_csv.open(newline="", encoding="utf-8") as file:
        saved = {int(r["bin_id"]): r for r in csv.DictReader(file)}
    with (output / "positions.csv").open(newline="", encoding="utf-8") as file:
        positions = list(csv.DictReader(file))
    ids = inputs["selected_bin_ids"]
    assert len(positions) == len(ids) + 1 == 501
    assert positions[0]["is_initial_anchor"] == "True" and positions[0]["bin_id"] == ""
    assert [int(r["bin_id"]) for r in positions[1:]] == ids
    ends = np.array(inputs["selected_bin_end_s"])
    starts = np.array(inputs["selected_bin_start_s"])
    np.testing.assert_array_equal([float(r["time_s"]) for r in positions[1:]], ends)
    assert np.array_equal(starts[1:], ends[:-1]) and np.all(np.diff(ids) == 1)
    residuals = {}
    for name in ("reference", "ridge", "gru", "kalman"):
        values = np.array(
            [[float(r[f"{name}_{a}_mm"]) for a in ("x", "y")] for r in positions]
        )
        np.testing.assert_array_equal(
            values[0], inputs["initial_anchor"]["position_mm"]
        )
        prefix = "actual" if name == "reference" else name
        velocity = np.array(
            [[float(saved[i][f"{prefix}_v{a}_mm_s"]) for a in ("x", "y")] for i in ids]
        )
        # Difference exported positions instead of repeating cumulative integration.
        residual = np.diff(values, axis=0) - velocity * (ends - starts)[:, None]
        assert float(np.abs(residual).max()) < 1e-10
        residuals[name] = float(np.abs(residual).max())
    limits = metadata["visual"]["common_axis_limits_mm"]
    assert all(
        frame["axis_limits_mm"] == limits
        for frame in metadata["visual"]["inspection_frames"].values()
    )
    stream = metadata["encoded_video_probe"]["streams"][0]
    assert (
        stream["nb_frames"] == "500"
        and stream["avg_frame_rate"] == "50/1"
        and float(stream["duration"]) == 10.0
    )
    assert sha(output / "still.png") == sha(output / "frame_middle.png")
    result = {
        "scope": "saved-artifact verification only; no new inference, tracker processing or test data",
        "recorded_hashes": "all verified",
        "common_anchor": "exact equality for all four paths",
        "maximum_per_step_displacement_residual_mm": residuals,
        "frame_timing": "500 exact selected endpoints; 501 positions including the pre-clip anchor; 50 fps / 10 seconds",
        "axis_limits": "identical at first, middle and last inspection frames",
        "representative_still": "exact middle-frame PNG bytes",
        "run_json_sha256": sha(output / "run.json"),
        "audit_sha256_utf8_lf": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").encode()
        ).hexdigest(),
    }
    write_json(output / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
