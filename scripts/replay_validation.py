"""Phase 6A: validation replay equivalence, then a single declared CPU timing pass."""

import argparse
import csv
import hashlib
import json
import os
import platform
import time
import tomllib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from audit_gru import audit, read_rows
from bin_recording import load_verified_inputs
from evaluate_baselines import git_state, write_predictions
from evaluate_kalman import verified_artifact
from threadpoolctl import threadpool_info, threadpool_limits

from neurodecode.data.features import FrozenCountScaler, prepare_partition
from neurodecode.evaluation import VelocitySeries, targets_from_windows
from neurodecode.models.gru import GRUDecoder, gru_inputs
from neurodecode.models.kalman import (
    KalmanParameters,
    current_bin_observations,
    filter_observations,
)
from neurodecode.models.ridge import RidgeDecoder, ridge_inputs, scaler_sha256
from neurodecode.streaming import SequentialDecoder
from neurodecode.streaming import iter_replay_bins as packet_iterator
from neurodecode.training import cpu_training_runtime


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        file.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_name() -> str:
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def timing_summary(ns: list[int]) -> dict:
    """Declared linear percentiles of measured service times, without trimming."""
    values = np.asarray(ns, dtype=float) / 1e6
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Timing samples must be finite, positive and nonempty")
    return {
        "samples": len(values),
        **{
            f"p{p}_ms": float(np.percentile(values, p, method="linear"))
            for p in (50, 95, 99)
        },
        "maximum_ms": float(values.max()),
        "mean_ms": float(values.mean()),
        "timed_service_rate_bins_s": float(1000 / values.mean()),
    }


def compare(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> dict:
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("Prediction shape/finiteness mismatch")
    differences = np.abs(actual - expected)
    return {
        "maximum_absolute_difference_mm_s": float(differences.max()),
        "maximum_absolute_difference_by_axis_mm_s": differences.max(axis=0).tolist(),
        "atol_mm_s": atol,
        "rtol": rtol,
        "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
    }


def check_identities(results, windows) -> None:
    if len(results) != len(windows.target_time_s):
        raise ValueError("Replay eligible row count differs from offline")
    for i, result in enumerate(results):
        if (
            result.bin_id != windows.target_bin_end_indices[i]
            or result.time_s != windows.target_time_s[i]
            or result.segment_id != windows.segment_ids[i]
            or result.partition != windows.partition
            or result.source_bin_ids != tuple(windows.source_bin_end_indices[i])
        ):
            raise ValueError(
                "Replay source/timestamp/segment/partition identity changed"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    if not args.run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in args.run_id
    ):
        raise ValueError("Run ID must use letters, digits, hyphens or underscores")
    output = Path("runs") / args.run_id
    if output.exists():
        raise ValueError(f"Existing run preserved: {output}")
    protocol = tomllib.loads(Path("configs/replay.toml").read_text(encoding="utf-8"))
    expected = {
        "partition": "validation",
        "warmup_predictions": 200,
        "timed_predictions": 2775,
        "passes": 1,
        "batch_size": 1,
        "threads": 1,
        "percentile_method": "linear",
        "ridge_atol_mm_s": 1e-10,
        "ridge_rtol": 1e-12,
        "kalman_atol_mm_s": 1e-10,
        "kalman_rtol": 1e-12,
        "gru_atol_mm_s": 1e-4,
        "gru_rtol": 1e-6,
        "gru_seed": 2026,
        "model_order": ["ridge", "gru", "kalman"],
    }
    if protocol != expected:
        raise ValueError("This runner implements the declared Phase 6A protocol only")
    output.mkdir(parents=True)
    created = datetime.now(timezone.utc).isoformat()
    write_json(
        output / "protocol.json",
        {"created_utc": created, "protocol": protocol, "git": git_state()},
    )
    # Only trusted local project artifacts. Audit hashes BEFORE model deserialization.
    gru_directory = Path("runs/phase5b-gru-round1")
    previous_audit = audit(gru_directory)
    previous = json.loads((gru_directory / "run.json").read_text(encoding="utf-8"))
    selected = next(
        r
        for r in previous["records"]
        if r["learning_rate"] == previous["selected_learning_rate"]
        and r["seed"] == protocol["gru_seed"]
    )
    if selected["status"] != "complete" or selected["best_epoch"] != 50:
        raise ValueError(
            "Representative GRU differs from completed Phase 5B checkpoint"
        )
    scaler = FrozenCountScaler.from_metadata(previous["normalization"])
    ridge_path, _ = verified_artifact(
        Path("runs/phase3b-ridge"), "ridge.json", "artifact_sha256"
    )
    kalman_path, kalman_fit = verified_artifact(
        Path("runs/phase4a-kalman-fit"),
        "kalman_parameters.json",
        "parameter_artifact_sha256",
    )
    gru_path = gru_directory / selected["run"] / "gru.pt"
    if sha(gru_path) != selected["checkpoint_sha256"]:
        raise ValueError("Representative checkpoint checksum changed")
    models = {
        "ridge": RidgeDecoder.load(ridge_path, expected_scaler=scaler),
        "gru": GRUDecoder.load(gru_path, expected_scaler=scaler),
        "kalman": KalmanParameters.load(kalman_path, expected_scaler=scaler),
    }
    recording, specification, timing = load_verified_inputs()
    if (
        specification != previous["split_specification"]
        or specification["source"] != kalman_fit["source"]
    ):
        raise ValueError("Source/split changed since model fitting")
    targets, windows = prepare_partition(recording, specification, timing, "validation")
    packets = list(packet_iterator(targets, specification, timing))
    # Prepared labels remain outside SequentialDecoder; only fixed quality flags enter.
    labels = targets_from_windows(windows)
    saved_rows = read_rows(gru_directory / "predictions.csv")
    if len(saved_rows) != protocol["timed_predictions"] or len(saved_rows) != len(
        windows.target_rows
    ):
        raise ValueError("Saved reference population changed")
    for i, row in enumerate(saved_rows):
        if (
            int(row["bin_id"]) != windows.target_bin_end_indices[i]
            or float(row["time_s"]) != windows.target_time_s[i]
            or int(row["segment_id"]) != windows.segment_ids[i]
            or row["partition"] != "validation"
            or row["score_mask"] != "True"
            or int(row["first_source_bin_id"]) != windows.source_bin_end_indices[i, 0]
            or int(row["last_source_bin_id"]) != windows.source_bin_end_indices[i, -1]
            or any(
                float(row[f"actual_v{axis}_mm_s"]) != labels.values_mm_s[i, a]
                for a, axis in enumerate(("x", "y"))
            )
        ):
            raise ValueError(
                "Saved reference identities/targets do not match preparation"
            )
    saved = {
        name: np.array(
            [[float(r[f"{name}_v{a}_mm_s"]) for a in ("x", "y")] for r in saved_rows]
        )
        for name in models
    }
    origin = specification["bin_grid"]["origin_s"]
    equivalence, first_calls, untimed, benchmark, measured = {}, {}, {}, {}, {}
    with threadpool_limits(limits=1), cpu_training_runtime(2026):
        # Equivalence happens before any warmed timing pass.
        for name, model in models.items():
            stream = SequentialDecoder(model, origin, timing.bin_s)
            results = []
            for packet in packets:
                result = stream.step(packet, measure=not results)
                if result is not None:
                    results.append(result)
            check_identities(results, windows)
            untimed[name] = results
            first_calls[name] = {
                "model_ms": results[0].model_ns / 1e6,
                "preparation_model_ms": results[0].preparation_model_ns / 1e6,
                "scope": "first eligible prediction after load, before this model's offline comparison; not fresh-process startup; model loading excluded",
            }
            if name == "ridge":
                offline = model.predict(ridge_inputs(windows, scaler)).values_mm_s
            elif name == "gru":
                offline = model.predict(gru_inputs(windows, scaler)).values_mm_s
            else:
                offline = filter_observations(
                    model,
                    current_bin_observations(
                        windows,
                        scaler,
                        specification["source"]["session"],
                        origin,
                        timing.bin_s,
                    ),
                ).predictions.values_mm_s
            replay = np.array([r.velocity_mm_s for r in results])
            atol, rtol = protocol[f"{name}_atol_mm_s"], protocol[f"{name}_rtol"]
            equivalence[name] = {
                "rows": len(results),
                "resets": sum(r.reset for r in results),
                "replay_vs_saved": compare(replay, saved[name], atol, rtol),
                "replay_vs_offline_api": compare(replay, offline, atol, rtol),
                "offline_api_vs_saved": compare(offline, saved[name], atol, rtol),
            }
            print(json.dumps({"equivalence": name, **equivalence[name]}), flush=True)
        write_json(output / "equivalence.json", equivalence)
        if not all(
            check["passed"]
            for result in equivalence.values()
            for key, check in result.items()
            if key.endswith(("saved", "api"))
        ):
            raise ValueError("Equivalence failed; investigate before benchmarking")
        for name, model in models.items():
            warm = SequentialDecoder(model, origin, timing.bin_s)
            warmed = 0
            for packet in packets:
                if warm.step(packet) is not None:
                    warmed += 1
                if warmed == protocol["warmup_predictions"]:
                    break
            if warmed != protocol["warmup_predictions"]:
                raise ValueError("Insufficient warm-up predictions")
            # Fresh scientific state, warmed runtime; include the reset row in timing.
            stream = SequentialDecoder(model, origin, timing.bin_s)
            results = [
                r for p in packets if (r := stream.step(p, measure=True)) is not None
            ]
            check_identities(results, windows)
            np.testing.assert_array_equal(
                [r.velocity_mm_s for r in results],
                [r.velocity_mm_s for r in untimed[name]],
            )
            measured[name] = results
            benchmark[name] = {
                "model_api": timing_summary([r.model_ns for r in results]),
                "preparation_plus_model": timing_summary(
                    [r.preparation_model_ns for r in results]
                ),
                "warmup_predictions": warmed,
            }
            print(json.dumps({"benchmark": name, **benchmark[name]}), flush=True)
        native_pools = threadpool_info()
    titles = {"ridge": "Ridge", "gru": "GRU", "kalman": "Kalman"}
    predictions = {
        titles[name]: VelocitySeries(
            np.array([r.velocity_mm_s for r in results]), labels.index
        )
        for name, results in untimed.items()
    }
    write_predictions(output / "predictions.csv", windows, labels, predictions)
    with (output / "timing_samples.csv").open(
        "x", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["model", "bin_id", "time_s", "reset", "model_ns", "preparation_model_ns"]
        )
        for name, results in measured.items():
            writer.writerows(
                (name, r.bin_id, r.time_s, r.reset, r.model_ns, r.preparation_model_ns)
                for r in results
            )
    eligible = set(int(i) for i in windows.target_bin_end_indices)
    with (output / "bin_status.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "source_row",
                "bin_id",
                "start_s",
                "end_s",
                "segment_id",
                "partition",
                "source_valid",
                "target_valid",
                "emitted_all_models",
            ]
        )
        writer.writerows(
            (
                p.source_row,
                p.bin_id,
                p.start_s,
                p.end_s,
                p.segment_id,
                p.partition,
                p.source_valid,
                p.target_valid,
                p.bin_id in eligible,
            )
            for p in packets
        )
    paths = [
        Path("src/neurodecode/streaming.py"),
        Path(__file__),
        Path("configs/replay.toml"),
    ]
    paths += list(Path("src/neurodecode").rglob("*.py"))
    paths += [
        Path("scripts") / name
        for name in (
            "bin_recording.py",
            "evaluate_baselines.py",
            "evaluate_kalman.py",
            "audit_gru.py",
        )
    ]
    metadata = {
        "created_utc": created,
        "protocol": protocol,
        "git": git_state(),
        "source": specification["source"],
        "split_specification": specification,
        "identity": dict(windows.identity),
        "normalization_sha256": scaler_sha256(scaler),
        "channel_ids": list(scaler.channel_ids),
        "unit_indices": [list(u) for u in scaler.unit_indices],
        "normalization": scaler.to_metadata(),
        "source_bins": len(packets),
        "emitted_bins": len(windows.target_rows),
        "model_artifacts": {
            name: {"path": p.as_posix(), "sha256": sha(p)}
            for name, p in (
                ("ridge", ridge_path),
                ("gru", gru_path),
                ("kalman", kalman_path),
            )
        },
        "phase5b_artifact_audit": previous_audit,
        "saved_predictions_sha256": sha(gru_directory / "predictions.csv"),
        "equivalence": equivalence,
        "latency": benchmark,
        "first_eligible_call": first_calls,
        "timing_boundaries": {
            "model_api": "already normalized indexed input -> existing checked predict API (ridge/GRU) or kalman_step with current state -> velocity/state; includes validation, GRU tensor copying/conversion, NumPy output and Kalman numerical diagnostics",
            "preparation_plus_model": "ReplayBin already in memory -> packet validation, history update, singleton HistoryWindows, existing frozen scaler/feature adapter, model API and state update; excludes packet construction and returned record packaging",
            "excluded_both": [
                "model loading",
                "imports",
                "recording reads",
                "spike-event binning",
                "target/quality preparation",
                "packet construction",
                "acquisition time",
                "plotting",
                "playback sleep",
                "CSV/JSON logging",
                "returned record packaging",
            ],
            "clock": vars(time.get_clock_info("perf_counter")),
            "note": "nested per-emission timers; no outlier removal; one pass, fixed order; background OS scheduling and timer overhead retained; total can include a suspension between inner and outer clocks",
        },
        "environment": {
            "cpu": cpu_name(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in (
                    "numpy",
                    "torch",
                    "scipy",
                    "h5py",
                    "scikit-learn",
                    "threadpoolctl",
                )
            },
            "device": "CPU",
            "threads": 1,
            "native_threadpools": native_pools,
        },
        "first_two_replayed_bins": [
            {
                "bin_id": untimed["ridge"][i].bin_id,
                "time_s": untimed["ridge"][i].time_s,
                "source_bin_ids": list(untimed["ridge"][i].source_bin_ids),
                "predictions_mm_s": {
                    name: rows[i].velocity_mm_s.tolist()
                    for name, rows in untimed.items()
                },
                "kalman_reset": untimed["kalman"][i].reset,
                "actual_mm_s_scoring_only": labels.values_mm_s[i].tolist(),
            }
            for i in range(2)
        ],
        "test": "no test windows, targets, predictions, scores or timing; common loader reads source/global metadata",
        "implementation_sha256_utf8_lf": {
            p.as_posix(): hashlib.sha256(
                p.read_text(encoding="utf-8").encode()
            ).hexdigest()
            for p in paths
        },
        "artifact_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    }
    write_json(output / "run.json", metadata)
    write_json(output / "latency.json", benchmark)
    print(
        json.dumps(
            {
                "run": output.as_posix(),
                "first_eligible_call": first_calls,
                "cpu": metadata["environment"]["cpu"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
