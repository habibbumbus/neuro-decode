"""Audit saved replay/timing evidence and isolate GRU batch arithmetic differences."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_gru import read_rows
from bin_recording import load_verified_inputs
from replay_validation import timing_summary
from threadpoolctl import threadpool_limits

from neurodecode.data.features import FrozenCountScaler, prepare_partition
from neurodecode.models.gru import GRUDecoder, gru_inputs
from neurodecode.training import cpu_training_runtime


def audit(directory: Path) -> dict:
    """Check artifact hashes/timing arithmetic; no model selection or benchmark."""
    directory = directory.resolve()
    if not directory.is_relative_to(Path("runs").resolve()):
        raise ValueError("Only trusted project runs may be audited")
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    for filename, expected in metadata["artifact_sha256"].items():
        path = (directory / filename).resolve()
        if (
            not path.is_relative_to(directory)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Changed artifact: {filename}")
    samples = read_rows(directory / "timing_samples.csv")
    for name in ("ridge", "gru", "kalman"):
        rows = [row for row in samples if row["model"] == name]
        if len(rows) != metadata["emitted_bins"]:
            raise ValueError("Timing population changed")
        for field, title in (
            ("model_ns", "model_api"),
            ("preparation_model_ns", "preparation_plus_model"),
        ):
            if (
                timing_summary([int(row[field]) for row in rows])
                != metadata["latency"][name][title]
            ):
                raise ValueError("Saved timing summary disagrees with raw samples")
        if any(int(row["model_ns"]) > int(row["preparation_model_ns"]) for row in rows):
            raise ValueError("Timing boundaries must be nested")
    recording, specification, timing = load_verified_inputs()
    if specification != metadata["split_specification"]:
        raise ValueError("Split changed")
    _, windows = prepare_partition(recording, specification, timing, "validation")
    scaler = FrozenCountScaler.from_metadata(metadata["normalization"])
    checkpoint = Path(metadata["model_artifacts"]["gru"]["path"]).resolve()
    if (
        not checkpoint.is_relative_to(Path("runs").resolve())
        or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        != metadata["model_artifacts"]["gru"]["sha256"]
    ):
        raise ValueError("Trusted representative checkpoint changed")
    model = GRUDecoder.load(checkpoint, expected_scaler=scaler)
    with threadpool_limits(limits=1), cpu_training_runtime(2026):
        direct = model.predict(gru_inputs(windows, scaler), batch_size=1).values_mm_s
    rows = read_rows(directory / "predictions.csv")
    if len(rows) != len(direct):
        raise ValueError("Replay prediction population changed")
    for i, row in enumerate(rows):
        if (
            int(row["bin_id"]) != windows.target_bin_end_indices[i]
            or float(row["time_s"]) != windows.target_time_s[i]
        ):
            raise ValueError("Replay row identity changed")
    replay = np.array(
        [[float(row[f"gru_v{axis}_mm_s"]) for axis in ("x", "y")] for row in rows]
    )
    np.testing.assert_array_equal(replay, direct)
    return {
        "scope": "validation-only verification; no fitting, timing rerun or test access",
        "all_recorded_artifact_hashes": "verified",
        "raw_timing_statistics": "exactly reproduce all saved summaries",
        "gru_offline_batch_one_vs_replay_maximum_absolute_difference_mm_s": float(
            np.abs(replay - direct).max()
        ),
        "conclusion": "same batch size gives exact equality; differences from saved batch-256 predictions arise from batch arithmetic, not history/state drift",
        "run_json_sha256": hashlib.sha256(
            (directory / "run.json").read_bytes()
        ).hexdigest(),
        "audit_implementation_sha256_utf8_lf": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").encode()
        ).hexdigest(),
    }


def main() -> None:
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
