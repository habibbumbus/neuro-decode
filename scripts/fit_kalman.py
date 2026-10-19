"""Phase 4A: fit training Kalman parameters and diagnostics; no filtering."""

import argparse
import hashlib
import json
import platform
import time
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from bin_recording import load_verified_inputs
from evaluate_baselines import git_state
from threadpoolctl import threadpool_info, threadpool_limits

from neurodecode.data.features import (
    FrozenCountScaler,
    fit_count_scaler,
    prepare_partition,
)
from neurodecode.evaluation import targets_from_windows
from neurodecode.models.kalman import (
    KalmanFitConfig,
    current_bin_observations,
    fit_kalman_parameters,
    normalization_sha256,
)


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="New directory under ignored runs/")
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
    config_path = Path("configs/kalman.toml")
    fit_config = KalmanFitConfig(
        **tomllib.loads(config_path.read_text(encoding="utf-8"))["kalman_fit"]
    )
    fit_config.validate()
    recording, specification, timing = load_verified_inputs()
    aligned, training = prepare_partition(recording, specification, timing, "train")
    computed_scaler = fit_count_scaler(aligned, specification, timing)
    scaler = computed_scaler
    normalization_provenance = (
        "unchanged Phase 2D rule, fitted only on unique training bins"
    )
    prior_scaler_path = Path("runs/phase2d/count_scaler.json")
    if prior_scaler_path.exists():
        prior = json.loads(prior_scaler_path.read_text(encoding="utf-8"))
        scaler = FrozenCountScaler.from_metadata(
            {key: prior[key] for key in computed_scaler.to_metadata()}
        )
        if normalization_sha256(scaler) != normalization_sha256(computed_scaler):
            raise ValueError(
                "Saved Phase 2D transform disagrees with verified training-only fit"
            )
        normalization_provenance = (
            "restored Phase 2D frozen metadata; verified against unique training bins"
        )
    observations = current_bin_observations(
        training,
        scaler,
        specification["source"]["session"],
        specification["bin_grid"]["origin_s"],
        timing.bin_s,
    )
    states = targets_from_windows(training)
    # No validation/test partition preparation, inference, metrics or filtering.
    with threadpool_limits(limits=1):
        parameters = fit_kalman_parameters(
            observations, states, scaler, timing, fit_config
        )
        native_libraries = [
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
            for item in threadpool_info()
        ]
    output.mkdir(parents=True)
    parameters.save(output / "kalman_parameters.json")
    diagnostics = parameters.diagnostics
    shapes = {
        name: list(getattr(parameters, name).shape)
        for name in (
            "A",
            "a",
            "H",
            "d",
            "Q",
            "R",
            "initial_mean",
            "initial_covariance",
        )
    }
    summary = {
        "shapes": shapes,
        "finite_parameter_checks": "passed",
        "eligible_training_observations": diagnostics["valid_training_observations"],
        "eligible_training_transitions": diagnostics["eligible_training_transitions"],
        "excluded_adjacent_rows": diagnostics["excluded_adjacent_rows"],
        "A": parameters.A.tolist(),
        "a": parameters.a.tolist(),
        "A_spectral_radius": diagnostics["A_spectral_radius"],
        "Q": parameters.Q.tolist(),
        "initial_mean_mm_s": parameters.initial_mean.tolist(),
        "covariance_diagnostics": {
            key: diagnostics[key] for key in ("Q", "R", "initial_covariance")
        },
        "normalization_sha256": normalization_sha256(scaler),
        "run_directory": output.as_posix(),
        "filtering_or_decoding_evaluation": "not implemented/run",
    }
    implementation_files = [
        "scripts/fit_kalman.py",
        "src/neurodecode/models/kalman.py",
        "scripts/bin_recording.py",
        "scripts/evaluate_baselines.py",
        "src/neurodecode/data/features.py",
        "src/neurodecode/data/targets.py",
        "src/neurodecode/data/binning.py",
        "src/neurodecode/data/loaders.py",
        "src/neurodecode/evaluation.py",
        "src/neurodecode/splits.py",
    ]
    metadata = {
        "schema_version": 1,
        "checkpoint": "Phase 4A parameter estimation only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": specification["source"],
        "split_specification": specification,
        "preparation_identity": dict(training.identity),
        "timing": asdict(timing),
        "fit_configuration": asdict(fit_config),
        "kalman_configuration_sha256_utf8_lf": hashlib.sha256(
            config_path.read_text(encoding="utf-8").encode()
        ).hexdigest(),
        "channel_ids": list(training.channel_ids),
        "unit_indices": [list(u) for u in training.unit_indices],
        "normalization_provenance": normalization_provenance,
        "normalization_sha256": normalization_sha256(scaler),
        "training_preparation_summary": training.summary,
        "summary": summary,
        "git": git_state(),
        "device": "CPU",
        "seed": None,
        "versions": {name: version(name) for name in ("numpy", "scipy", "neurodecode")},
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "native_libraries": native_libraries,
        "native_thread_limit": 1,
        "implementation_sha256_utf8_lf": {
            path: hashlib.sha256(
                Path(path).read_text(encoding="utf-8").encode()
            ).hexdigest()
            for path in implementation_files
        },
        "parameter_artifact_sha256": hashlib.sha256(
            (output / "kalman_parameters.json").read_bytes()
        ).hexdigest(),
        "duration_s": time.perf_counter() - started,
        "duration_scope": "training preparation, parameter fitting and export; not latency",
        "held_out_evaluation": "neither validation nor test targets/windows/predictions/scores constructed",
    }
    for filename, value in (("run.json", metadata), ("diagnostics.json", diagnostics)):
        with (output / filename).open("x", encoding="utf-8") as file:
            file.write(json.dumps(value, indent=2, allow_nan=False) + chr(10))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
