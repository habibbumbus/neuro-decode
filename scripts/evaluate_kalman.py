"""Phase 4B: frozen Kalman filtering and matched validation comparison only."""

import argparse
import csv
import hashlib
import json
import os
import platform
import time
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from bin_recording import load_verified_inputs
from evaluate_baselines import git_state, write_predictions
from threadpoolctl import threadpool_info, threadpool_limits

from neurodecode.data.features import prepare_partition
from neurodecode.evaluation import (
    evaluate_velocity,
    index_from_windows,
    targets_from_windows,
)
from neurodecode.models.baselines import TrainingMeanVelocity, predict_zero
from neurodecode.models.kalman import (
    KalmanParameters,
    current_bin_observations,
    filter_observations,
)
from neurodecode.models.ridge import RidgeDecoder, ridge_inputs
from neurodecode.plotting import save_baseline_velocity_figure


def verified_artifact(
    directory: Path, filename: str, hash_field: str
) -> tuple[Path, dict]:
    """Check the existing run's recorded bytes before loading its frozen model."""
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    path = directory / filename
    expected = metadata[hash_field]
    if isinstance(expected, dict):
        expected = expected[filename]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Saved artifact checksum mismatch: {path}")
    return path, metadata


def audit_information_form(parameters, observations, filtered):
    """Independent two-state posterior solve, diagnostic only; never changes output."""
    differences = []
    covariance_differences = []
    p = parameters
    for row in np.flatnonzero(observations.index.score_mask):
        if filtered.reset[row]:
            mean, covariance = p.initial_mean, p.initial_covariance
        else:
            mean = p.A @ filtered.predictions.values_mm_s[row - 1] + p.a
            covariance = p.A @ filtered.covariance[row - 1] @ p.A.T + p.Q
        precision = np.linalg.solve(covariance, np.eye(2))
        weighted_H = np.linalg.solve(p.R, p.H)
        weighted_z = np.linalg.solve(p.R, observations.values[row] - p.d)
        posterior_precision = precision + p.H.T @ weighted_H
        mean_check = np.linalg.solve(
            posterior_precision, precision @ mean + p.H.T @ weighted_z
        )
        covariance_check = np.linalg.solve(posterior_precision, np.eye(2))
        differences.append(
            float(np.max(np.abs(mean_check - filtered.predictions.values_mm_s[row])))
        )
        covariance_differences.append(
            float(np.max(np.abs(covariance_check - filtered.covariance[row])))
        )
    if (
        max(differences, default=0.0) > 1e-7
        or max(covariance_differences, default=0.0) > 1e-7
    ):
        raise ValueError("Equivalent two-state update disagrees with innovation solve")
    return {
        "rows_checked": len(differences),
        "maximum_mean_difference_mm_s": max(differences, default=0.0),
        "maximum_covariance_difference": max(covariance_differences, default=0.0),
        "purpose": "independent equivalent posterior equations; no tuning or output replacement",
    }


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="New directory under ignored runs/")
    parser.add_argument(
        "--kalman-run", type=Path, default=Path("runs/phase4a-kalman-fit")
    )
    parser.add_argument("--ridge-run", type=Path, default=Path("runs/phase3b-ridge"))
    args = parser.parse_args()
    if not args.run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in args.run_id
    ):
        raise ValueError("Run ID must use letters, digits, hyphens or underscores")
    output = Path("runs") / args.run_id
    if output.exists():
        raise ValueError(f"Existing run preserved: {output}; choose a new run ID")
    model_path, fit_run = verified_artifact(
        args.kalman_run, "kalman_parameters.json", "parameter_artifact_sha256"
    )
    ridge_path, ridge_run = verified_artifact(
        args.ridge_run, "ridge.json", "artifact_sha256"
    )
    parameters = KalmanParameters.load(model_path)
    ridge = RidgeDecoder.load(ridge_path, expected_scaler=parameters.scaler)
    if ridge.alpha != ridge_run["selected_alpha"]:
        raise ValueError("Ridge differs from the saved selected candidate")
    recording, specification, timing = load_verified_inputs()
    fit_configuration = tomllib.loads(
        Path("configs/kalman.toml").read_text(encoding="utf-8")
    )["kalman_fit"]
    if (
        fit_run["source"] != specification["source"]
        or fit_run["split_specification"] != specification
        or fit_run["timing"] != asdict(timing)
        or fit_configuration != asdict(parameters.config)
        or fit_run["normalization_sha256"] != ridge_run["normalization_sha256"]
        or fit_run["summary"]["initial_mean_mm_s"] != parameters.initial_mean.tolist()
        or fit_run["summary"]["eligible_training_observations"] != ridge.fitted_examples
    ):
        raise ValueError("Saved fit/source/split/configuration/prior identity mismatch")
    # Same Phase 2D preparation; no alternative eligibility or test preparation.
    _, windows = prepare_partition(recording, specification, timing, "validation")
    index = index_from_windows(windows)
    observations = current_bin_observations(
        windows,
        parameters.scaler,
        specification["source"]["session"],
        specification["bin_grid"]["origin_s"],
        timing.bin_s,
    )
    mean = TrainingMeanVelocity(
        parameters.initial_mean,
        ridge.fitted_examples,
        parameters.scaler.channel_ids,
        parameters.scaler.unit_indices,
        parameters.scaler.identity,
    )
    with threadpool_limits(limits=1):
        filtered = filter_observations(parameters, observations)
        numerical_audit = audit_information_form(parameters, observations, filtered)
        predictions = {
            "Zero velocity": predict_zero(index),
            "Training mean": mean.predict(index),
            "Ridge": ridge.predict(ridge_inputs(windows, ridge.scaler)),
            "Kalman": filtered.predictions,
        }
        native = threadpool_info()
    # Access scoring labels only after all frozen-model predictions exist.
    labels = targets_from_windows(windows)
    metrics = {
        name: evaluate_velocity(labels, prediction)
        for name, prediction in predictions.items()
    }
    # Require the comparison to reproduce all original Phase 3B rows/values.
    previous_csv = args.ridge_run / "predictions.csv"
    expected_csv_hash = ridge_run["artifact_sha256"]["predictions.csv"]
    if hashlib.sha256(previous_csv.read_bytes()).hexdigest() != expected_csv_hash:
        raise ValueError("Saved ridge prediction checksum mismatch")
    with previous_csv.open(newline="", encoding="utf-8") as file:
        previous = list(csv.DictReader(file))
    if len(previous) != len(index.time_s):
        raise ValueError("Saved validation row count changed")
    for row, saved in enumerate(previous):
        if (
            int(saved["bin_id"]) != index.bin_ids[row]
            or float(saved["time_s"]) != index.time_s[row]
            or int(saved["segment_id"]) != index.segment_ids[row]
            or saved["partition"] != index.partition
            or saved["score_mask"] != str(bool(index.score_mask[row]))
        ):
            raise ValueError("Saved validation timestamp/mask identity changed")
        for axis, name in enumerate(("x", "y")):
            if float(saved[f"actual_v{name}_mm_s"]) != labels.values_mm_s[row, axis]:
                raise ValueError("Saved validation target changed")
            for title, prefix in (
                ("Zero velocity", "zero"),
                ("Training mean", "mean"),
                ("Ridge", "ridge"),
            ):
                if (
                    float(saved[f"{prefix}_v{name}_mm_s"])
                    != predictions[title].values_mm_s[row, axis]
                ):
                    raise ValueError("Frozen reference prediction changed")
    output.mkdir(parents=True)
    parameters.save(output / "kalman_parameters.json")
    restored = KalmanParameters.load(
        output / "kalman_parameters.json", expected_scaler=parameters.scaler
    )
    with threadpool_limits(limits=1):
        roundtrip = filter_observations(restored, observations)
    np.testing.assert_array_equal(
        filtered.predictions.values_mm_s, roundtrip.predictions.values_mm_s
    )
    np.testing.assert_array_equal(filtered.covariance, roundtrip.covariance)
    write_predictions(output / "predictions.csv", windows, labels, predictions)
    with (output / "filter_status.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "partition",
                "segment_id",
                "bin_id",
                "time_s",
                "reset",
                "status",
                "steps_since_reset",
                "shared_history_warmup_bins",
            ]
        )
        for row in range(len(index.time_s)):
            writer.writerow(
                [
                    index.partition,
                    int(index.segment_ids[row]),
                    int(index.bin_ids[row]),
                    float(index.time_s[row]),
                    bool(filtered.reset[row]),
                    filtered.status[row],
                    int(filtered.steps_since_reset[row]),
                    timing.history_bins - 1,
                ]
            )
    with (output / "filter_covariance.npz").open("xb") as file:
        np.savez_compressed(
            file,
            bin_ids=index.bin_ids,
            time_s=index.time_s,
            covariance_mm2_s2=filtered.covariance,
        )
    traces = []
    for row, step in filtered.traces:
        traces.append(
            {
                "row": row,
                "bin_id": int(index.bin_ids[row]),
                "bin_start_s": float(observations.bin_start_s[row]),
                "time_s": float(index.time_s[row]),
                "status": filtered.status[row],
                "predicted_mean_mm_s": step.predicted_mean.tolist(),
                "predicted_covariance": step.predicted_covariance.tolist(),
                "channel_ids": list(index.channel_ids),
                "raw_current_counts": windows.X_raw[row, -1].tolist(),
                "observation": observations.values[row].tolist(),
                "expected_observation": (
                    parameters.H @ step.predicted_mean + parameters.d
                ).tolist(),
                "innovation": step.innovation.tolist(),
                "gain": step.gain.tolist(),
                "correction_mm_s": (step.gain @ step.innovation).tolist(),
                "updated_mean_mm_s": step.mean.tolist(),
                "updated_covariance": step.covariance.tolist(),
                "actual_mm_s_scoring_only": labels.values_mm_s[row].tolist(),
                "error_mm_s": (step.mean - labels.values_mm_s[row]).tolist(),
                "solve_relative_residual": step.solve_relative_residual,
            }
        )
    os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))
    plot = save_baseline_velocity_figure(
        labels,
        predictions,
        specification["source"]["session"],
        output / "validation_velocity.png",
    )
    if plot != ridge_run["plot"]:
        raise ValueError("Plot interval/policy differs from Phase 3B")
    identity_hash = hashlib.sha256(
        np.column_stack((index.segment_ids, index.bin_ids)).astype("<i8").tobytes()
        + index.time_s.astype("<f8").tobytes()
    ).hexdigest()
    if identity_hash != ridge_run["eligible_validation_identity_sha256"]:
        raise ValueError("Shared scoring identity changed")
    implementation_files = [
        "scripts/evaluate_kalman.py",
        "scripts/evaluate_baselines.py",
        "scripts/bin_recording.py",
        "src/neurodecode/models/kalman.py",
        "src/neurodecode/models/ridge.py",
        "src/neurodecode/models/baselines.py",
        "src/neurodecode/evaluation.py",
        "src/neurodecode/plotting.py",
        "src/neurodecode/data/features.py",
        "src/neurodecode/data/targets.py",
        "src/neurodecode/data/binning.py",
        "src/neurodecode/data/loaders.py",
        "src/neurodecode/splits.py",
    ]
    metadata = {
        "schema_version": 1,
        "checkpoint": "Phase 4B validation Kalman filtering",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": specification["source"],
        "split_specification": specification,
        "identity": dict(windows.identity),
        "timing": asdict(timing),
        "fit_configuration": asdict(parameters.config),
        "filter_configuration": {
            "gain": "numpy.linalg.solve(S, cross.T).T",
            "covariance": "Joseph; roundoff symmetrization only",
            "warmup": "unchanged Phase 2D nine excluded prefix bins; no additional Kalman exclusions",
            "reset": "training prior before first eligible observation; no extra transition",
            "missing": filtered.diagnostics["missing_policy"],
            "held_out_label_inputs": False,
        },
        "fitted_artifacts": {
            "kalman": {
                "path": model_path.as_posix(),
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            },
            "ridge": {
                "path": ridge_path.as_posix(),
                "sha256": hashlib.sha256(ridge_path.read_bytes()).hexdigest(),
                "alpha": ridge.alpha,
            },
        },
        "normalization_sha256": fit_run["normalization_sha256"],
        "channel_ids": list(index.channel_ids),
        "unit_indices": [list(u) for u in index.unit_indices],
        "training_mean_mm_s": parameters.initial_mean.tolist(),
        "validation_summary": windows.summary,
        "eligible_validation_identity_sha256": identity_hash,
        "metrics": metrics,
        "filter_diagnostics": filtered.diagnostics,
        "independent_numerical_audit": numerical_audit,
        "validation_velocity_mean_mm_s_scoring_only": {
            "actual": labels.values_mm_s.mean(axis=0).tolist(),
            **{
                name: pred.values_mm_s.mean(axis=0).tolist()
                for name, pred in predictions.items()
            },
        },
        "first_two_steps": traces,
        "plot": plot,
        "frozen_reference_predictions": "all Phase 3B target, zero, mean and ridge values/identities match exactly",
        "model_roundtrip": "exact prediction and covariance equality",
        "test_evaluation": "no test partition targets/windows/predictions/plots/scores constructed",
        "git": git_state(),
        "device": "CPU",
        "seed": None,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "h5py",
                    "matplotlib",
                    "threadpoolctl",
                )
            },
            "native_thread_limit": 1,
            "native_libraries": [
                {
                    key: item.get(key)
                    for key in (
                        "internal_api",
                        "prefix",
                        "num_threads",
                        "version",
                        "architecture",
                    )
                }
                for item in native
            ],
        },
        "implementation_sha256_utf8_lf": {
            path: hashlib.sha256(
                Path(path).read_text(encoding="utf-8").encode()
            ).hexdigest()
            for path in implementation_files
        },
        "artifact_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output.iterdir()
            if path.is_file()
        },
        "duration_s": time.perf_counter() - started,
        "duration_scope": "whole validation preparation/filtering/export; not inference latency",
    }
    for filename, data in (("metrics.json", metrics), ("run.json", metadata)):
        with (output / filename).open("x", encoding="utf-8") as file:
            file.write(json.dumps(data, indent=2, allow_nan=False) + chr(10))
    print(
        json.dumps(
            {
                "metrics": metrics,
                "filter_diagnostics": filtered.diagnostics,
                "run_directory": output.as_posix(),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
