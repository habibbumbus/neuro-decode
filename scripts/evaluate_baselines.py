"""Phase 3A: validation-only constant velocity baselines on Phase 2D examples."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from bin_recording import load_verified_inputs

from neurodecode.data.binning import bin_partition
from neurodecode.data.features import build_history_windows
from neurodecode.data.targets import align_velocity_targets
from neurodecode.evaluation import (
    evaluate_velocity,
    index_from_windows,
    targets_from_windows,
)
from neurodecode.models.baselines import TrainingMeanVelocity, predict_zero
from neurodecode.plotting import save_baseline_velocity_figure


def git_state() -> dict:
    """Read-only commit/dirty evidence; no persistent Git configuration change."""
    prefix = ["git", "-c", f"safe.directory={Path.cwd().as_posix()}"]
    commit = subprocess.run(
        prefix + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        prefix + ["status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    return {"commit": commit, "dirty": bool(status), "status": status.splitlines()}


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="New directory name under ignored runs/")
    args = parser.parse_args()
    if (
        not args.run_id
        or args.run_id in (".", "..")
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in args.run_id
        )
    ):
        raise ValueError("Run ID must use letters, digits, hyphens or underscores")
    output = Path("runs") / args.run_id
    if output.exists():
        raise ValueError(f"Existing run preserved: {output}; choose a new run ID")
    recording, specification, config = load_verified_inputs()
    prepared = {}
    # Intentionally never construct, predict, plot or score the test partition.
    for partition in ("train", "validation"):
        bins = bin_partition(recording, specification, config, partition)
        targets = align_velocity_targets(recording, bins, specification, config)
        prepared[partition] = build_history_windows(targets, specification, config)
    training, validation = prepared["train"], prepared["validation"]
    mean_model = TrainingMeanVelocity.fit(training)
    index = index_from_windows(validation)  # Label-free predictor inputs.
    predictions = {
        "Zero velocity": predict_zero(index),
        "Training mean": mean_model.predict(index),
    }
    # Predictions already exist and are frozen before validation labels are accessed for scoring.
    labels = targets_from_windows(validation)
    metrics = {
        name: evaluate_velocity(labels, prediction)
        for name, prediction in predictions.items()
    }
    output.mkdir(parents=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "partition",
                "segment_id",
                "bin_id",
                "time_s",
                "score_mask",
                "first_source_bin_id",
                "last_source_bin_id",
                "actual_vx_mm_s",
                "actual_vy_mm_s",
                "zero_vx_mm_s",
                "zero_vy_mm_s",
                "mean_vx_mm_s",
                "mean_vy_mm_s",
            ]
        )
        for row in range(index.time_s.size):
            writer.writerow(
                [
                    index.partition,
                    int(index.segment_ids[row]),
                    int(index.bin_ids[row]),
                    float(index.time_s[row]),
                    bool(index.score_mask[row]),
                    int(validation.source_bin_end_indices[row, 0]),
                    int(validation.source_bin_end_indices[row, -1]),
                    *labels.values_mm_s[row].tolist(),
                    *predictions["Zero velocity"].values_mm_s[row].tolist(),
                    *predictions["Training mean"].values_mm_s[row].tolist(),
                ]
            )
    os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))
    plot = save_baseline_velocity_figure(
        labels,
        predictions,
        specification["source"]["session"],
        output / "validation_velocity.png",
    )
    implementation_files = [
        "scripts/evaluate_baselines.py",
        "scripts/bin_recording.py",
        "src/neurodecode/models/baselines.py",
        "src/neurodecode/evaluation.py",
        "src/neurodecode/data/features.py",
        "src/neurodecode/data/targets.py",
        "src/neurodecode/data/binning.py",
        "src/neurodecode/data/loaders.py",
        "src/neurodecode/splits.py",
        "src/neurodecode/plotting.py",
    ]
    metadata = {
        "schema_version": 1,
        "checkpoint": "Phase 3A validation baselines",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": specification["source"],
        "identity": dict(validation.identity),
        "split_specification": specification,
        "timing": asdict(config),
        "channel_ids": list(validation.channel_ids),
        "unit_indices": [list(u) for u in validation.unit_indices],
        "train_summary": training.summary,
        "validation_summary": validation.summary,
        "training_mean_mm_s": mean_model.mean_mm_s.tolist(),
        "mean_fit_examples": mean_model.fitted_examples,
        "mean_fit_rule": "unweighted mean of eligible Phase 2D training targets; one target per window",
        "feature_transform": "unused: constant baselines ignore neural features",
        "score_mask_rule": "all eligible Phase 2D validation windows; identical mask for both baselines",
        "metric_policy": {
            "overall_rmse": "sqrt(mean squared error across all scored examples and both axes)",
            "r2": "1-SSE/SST per axis; constant axes or fewer than 2 scored examples -> null",
            "mean_r2": "unweighted axis mean; null if either axis undefined",
            "nonfinite": "fail, including unscored rows; no implicit filtering",
            "errors": "prediction minus target, mm/s",
        },
        "plot": plot,
        "test_evaluation": "not opened; no test windows constructed",
        "git": git_state(),
        "device": "CPU",
        "seed": None,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "matplotlib": version("matplotlib"),
            "h5py": version("h5py"),
            "cpu": platform.processor(),
            "threads": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "implementation_sha256_utf8_lf": {
            path: hashlib.sha256(
                Path(path).read_text(encoding="utf-8").encode()
            ).hexdigest()
            for path in implementation_files
        },
        "eligible_validation_identity_sha256": hashlib.sha256(
            np.column_stack((index.segment_ids, index.bin_ids)).astype("<i8").tobytes()
            + index.time_s.astype("<f8").tobytes()
        ).hexdigest(),
        "metrics": metrics,
        "duration_s": time.perf_counter() - started,
        "duration_scope": "whole preparation/scoring/export run; NOT inference latency",
    }
    for filename, value in (("run.json", metadata), ("metrics.json", metrics)):
        (output / filename).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(
        f"Validation comparison: {index.time_s.size} examples, {index.time_s[0]:.3f}–{index.time_s[-1]:.3f} s"
    )
    print(
        "Baseline             RMSE x     RMSE y    Overall     R² x      R² y    Mean R²"
    )
    for name, score in metrics.items():
        values = [
            *score["rmse_mm_s"],
            score["overall_rmse_mm_s"],
            *score["r2"],
            score["mean_r2"],
        ]
        print(
            f"{name:20s} "
            + " ".join(" undefined" if v is None else f"{v:10.6f}" for v in values)
        )
    print(
        json.dumps(
            {
                "first_validation_example": {
                    "bin_id": int(index.bin_ids[0]),
                    "time_s": float(index.time_s[0]),
                    "actual_mm_s": labels.values_mm_s[0].tolist(),
                    "predicted_mm_s": {
                        name: p.values_mm_s[0].tolist()
                        for name, p in predictions.items()
                    },
                    "errors_mm_s": {
                        name: (p.values_mm_s[0] - labels.values_mm_s[0]).tolist()
                        for name, p in predictions.items()
                    },
                },
                "training_mean_mm_s": mean_model.mean_mm_s.tolist(),
                "run_directory": output.as_posix(),
                "plot": plot,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
