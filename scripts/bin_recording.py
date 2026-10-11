"""Verify counts, targets or histories; show activity examples only for training."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from neurodecode.data import features as feature_module
from neurodecode.data.binning import bin_partition
from neurodecode.data.features import (
    build_history_windows,
    fit_count_scaler,
    flatten_features,
)
from neurodecode.data.loaders import Recording, load_recording
from neurodecode.data.targets import align_velocity_targets
from neurodecode.splits import TimingConfig, plan_splits, read_config


def print_target_summary(
    recording: Recording, specification: dict[str, object], config: TimingConfig
) -> None:
    """Audit all partition eligibility, showing positions/velocities only for training."""
    for partition in specification["partitions"]:
        bins = bin_partition(recording, specification, config, partition["name"])
        targets = align_velocity_targets(recording, bins, specification, config)
        rows = targets.valid_rows
        print(
            json.dumps(
                {
                    "partition": bins.partition,
                    "count_rows": int(bins.counts.shape[0]),
                    "aligned_valid_rows": int(rows.size),
                    "invalid_reasons": dict(
                        Counter(
                            reason for reason in targets.reasons if reason != "valid"
                        )
                    ),
                    "elapsed_deviations_over_1ns": int(
                        np.count_nonzero(
                            np.abs(targets.elapsed_deviation_s[targets.valid_mask])
                            > 1e-9
                        )
                    ),
                    "target_policy": targets.policy,
                },
                indent=2,
            )
        )
        if bins.partition == "train":
            print("First three valid training targets (fixed identity order):")
            for row in rows[:3]:
                selected = targets.source_indices[row]
                positions = recording.cursor_position_mm[selected]
                print(
                    json.dumps(
                        {
                            "original_count_row": int(row),
                            "grid_end_index": int(bins.grid_end_indices[row]),
                            "segment_id": int(bins.segment_ids[row]),
                            "neural_bin_s": [
                                float(bins.bin_start_s[row]),
                                float(bins.bin_end_s[row]),
                            ],
                            "first_three_channel_counts": bins.counts[row, :3].tolist(),
                            "source_position_times_s": targets.source_time_s[
                                row
                            ].tolist(),
                            "source_positions_mm": positions.tolist(),
                            "sample_age_s": targets.sample_age_s[row].tolist(),
                            "displacement_mm": (positions[1] - positions[0]).tolist(),
                            "denominator_s": float(targets.elapsed_s[row]),
                            "denominator_minus_nominal_s": float(
                                targets.elapsed_deviation_s[row]
                            ),
                            "velocity_mm_s": targets.velocity_mm_s[row].tolist(),
                        },
                        indent=2,
                    )
                )


def print_feature_summary(
    recording: Recording, specification: dict[str, object], config: TimingConfig
) -> None:
    """Reuse aligned data, fit once on training, and audit held-out shapes only."""
    train_bins = bin_partition(recording, specification, config, "train")
    train_targets = align_velocity_targets(recording, train_bins, specification, config)
    scaler = fit_count_scaler(train_targets, specification, config)
    metadata = scaler.to_metadata()
    metadata["implementation_sha256_utf8_lf"] = hashlib.sha256(
        Path(feature_module.__file__).read_text(encoding="utf-8").encode()
    ).hexdigest()
    output = Path("runs/phase2d/count_scaler.json")
    serialized = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise ValueError(
            f"Different scaler metadata exists at {output}; preserve/review it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(
        json.dumps(
            {
                "normalization_metadata": output.as_posix(),
                "fitted_unique_training_bins": scaler.fitted_bin_count,
                "identity": dict(scaler.identity),
                "channel_count": len(scaler.channel_ids),
                "constant_training_channels": [
                    name
                    for name, quiet in zip(
                        scaler.channel_ids, scaler.constant_channels, strict=True
                    )
                    if quiet
                ],
                "first_channel": scaler.channel_ids[0],
                "first_channel_mean": float(scaler.means[0]),
                "first_channel_scale": float(scaler.scales[0]),
                "fit_basis": metadata["fit_basis"],
                "bulk_array_persistence": "none",
            },
            indent=2,
        )
    )
    for partition in specification["partitions"]:
        if partition["name"] == "train":
            targets = train_targets
        else:
            bins = bin_partition(recording, specification, config, partition["name"])
            targets = align_velocity_targets(recording, bins, specification, config)
        windows = build_history_windows(targets, specification, config)
        X = scaler.transform(windows)
        flat = flatten_features(X)
        print(
            json.dumps(
                {
                    "partition": windows.partition,
                    **windows.summary,
                    "X_raw_shape": list(windows.X_raw.shape),
                    "X_normalized_shape": list(X.shape),
                    "y_shape": list(windows.y.shape),
                    "flattened_shape": list(flat.shape),
                },
                indent=2,
            )
        )
        if windows.partition == "train":
            print("First three training histories, selected by chronological identity:")
            for example in range(min(3, windows.y.shape[0])):
                print(
                    json.dumps(
                        {
                            "original_target_row": int(windows.target_rows[example]),
                            "target_bin_id": int(
                                windows.target_bin_end_indices[example]
                            ),
                            "segment_id": int(windows.segment_ids[example]),
                            "source_bin_ids": windows.source_bin_end_indices[
                                example
                            ].tolist(),
                            "source_bin_start_s": windows.source_bin_start_s[
                                example
                            ].tolist(),
                            "source_bin_end_s": windows.source_bin_end_s[
                                example
                            ].tolist(),
                            "channel_feature_shape": list(windows.X_raw[example].shape),
                            "aligned_velocity_mm_s": windows.y[example].tolist(),
                            "first_channel_raw_counts": windows.X_raw[
                                example, :, 0
                            ].tolist(),
                            "first_channel_normalized": X[example, :, 0].tolist(),
                            "oldest_feature_calculation": {
                                "raw": int(windows.X_raw[example, 0, 0]),
                                "training_mean": float(scaler.means[0]),
                                "training_scale": float(scaler.scales[0]),
                                "normalized": float(X[example, 0, 0]),
                            },
                        },
                        indent=2,
                    )
                )


def load_verified_inputs() -> tuple[Recording, dict[str, object], TimingConfig]:
    """Reuse source/config/split checks; load once without preparing test examples."""
    manifest = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    if len(manifest["recordings"]) != 1:
        raise ValueError("This checkpoint expects exactly one acquired recording")
    acquired = manifest["recordings"][0]
    path = Path(acquired["local_path"])
    with path.open("rb") as file:
        source_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    if source_sha256 != acquired["local_checksums"]["SHA256"]:
        raise ValueError("Recording differs from verified acquisition manifest")
    spec_path = Path("data/splits") / f"{acquired['session']}.json"
    specification = json.loads(spec_path.read_text(encoding="utf-8"))
    expected_source = {
        "record_id": manifest["dataset"]["record_id"],
        "version": manifest["dataset"]["version"],
        "record_url": manifest["dataset"]["record_url"],
        "session": acquired["session"],
        "filename": acquired["filename"],
        "sha256": source_sha256,
        "actual_size_bytes": path.stat().st_size,
    }
    if specification["source"] != expected_source:
        raise ValueError("Split source identity does not match acquired recording")
    config_path = Path("configs/base.toml")
    config_sha256 = hashlib.sha256(
        config_path.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    if specification["configuration"] != {
        "path": config_path.as_posix(),
        "sha256": config_sha256,
        "hash_encoding": "UTF-8 text with LF newlines",
    }:
        raise ValueError("Configuration differs from saved split specification")
    config, fractions = read_config(config_path)
    recording = load_recording(path)
    verified_plan = plan_splits(
        recording.behavior_times_s,
        recording.diagnostics["spike_range_s"],
        config,
        fractions,
    )
    for key, value in verified_plan.items():
        if specification.get(key) != value:
            raise ValueError(
                f"Saved split contract disagrees with verified metadata: {key}"
            )
    return recording, specification, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--targets",
        action="store_true",
        help="Audit aligned targets in all partitions; show training examples only",
    )
    modes.add_argument(
        "--features",
        action="store_true",
        help="Build histories and frozen training-only normalization; aggregate held-out audit",
    )
    args = parser.parse_args()
    recording, specification, config = load_verified_inputs()
    acquired = specification["source"]
    source_sha256 = acquired["sha256"]
    config_sha256 = specification["configuration"]["sha256"]
    spec_path = Path("data/splits") / f"{acquired['session']}.json"
    if args.features:
        print_feature_summary(recording, specification, config)
        return
    if args.targets:
        print("Verified source/configuration/split identity; no arrays saved or cached")
        print_target_summary(recording, specification, config)
        return
    binned = bin_partition(recording, specification, config, partition="train")
    expected_counts = np.zeros(len(binned.channel_ids), dtype=np.int64)
    for segment in binned.segment_metadata:
        first, stop = segment["row_start"], segment["row_stop"]
        if first == stop:
            continue
        start, end = binned.bin_start_s[first], binned.bin_end_s[stop - 1]
        for column, channel in enumerate(binned.source_channel_indices):
            expected_counts[column] += sum(
                int(np.count_nonzero((events > start) & (events <= end)))
                for events in recording.spike_times_s[channel]
            )
    if not np.array_equal(expected_counts, binned.counts.sum(axis=0)):
        raise ValueError(
            "Per-channel count conservation failed against original events"
        )
    print(
        json.dumps(
            {
                "session": acquired["session"],
                "partition": binned.partition,
                "source_sha256": source_sha256,
                "split_specification": spec_path.as_posix(),
                "split_sha256_canonical_json": hashlib.sha256(
                    json.dumps(
                        specification,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest(),
                "configuration_sha256_utf8_lf": config_sha256,
                "grid_origin_s": specification["bin_grid"]["origin_s"],
                "bin_width_s": config.bin_s,
                "event_interval": "(start, end]",
                "channel_policy": binned.channel_policy,
                "counts_shape": list(binned.counts.shape),
                "dtype": str(binned.counts.dtype),
                "first_channel": binned.channel_ids[0],
                "last_channel": binned.channel_ids[-1],
                "unit_slots_per_channel": sorted({len(u) for u in binned.unit_indices}),
                "first_channel_unit_indices": list(binned.unit_indices[0]),
                "unit_index_zero_role": "residual unsorted events; other indices are sorted slots",
                "first_bin_s": [
                    float(binned.bin_start_s[0]),
                    float(binned.bin_end_s[0]),
                ]
                if binned.counts.size
                else None,
                "last_bin_s": [
                    float(binned.bin_start_s[-1]),
                    float(binned.bin_end_s[-1]),
                ]
                if binned.counts.size
                else None,
                "training_events_included": int(expected_counts.sum()),
                "conservation": "passed separately for every included channel",
                "segments": binned.segment_metadata,
                "persistence": "none: arrays remain in memory; no processed cache read or written",
            },
            indent=2,
        )
    )
    print("Fixed example: first ten training bins, first three lexical channels")
    print("Columns:", binned.channel_ids[:3])
    print(binned.counts[:10, :3])
    channel = binned.source_channel_indices[0]
    print(
        f"Original events in those bins for {binned.channel_ids[0]} (U1 = residual unsorted):"
    )
    for row in range(min(10, binned.bin_end_s.size)):
        start, end = binned.bin_start_s[row], binned.bin_end_s[row]
        events_by_unit = {
            f"U{u + 1}": events[(events > start) & (events <= end)].tolist()
            for u, events in enumerate(recording.spike_times_s[channel])
        }
        print(
            f"({start:.3f}, {end:.3f}] -> {events_by_unit} -> count {binned.counts[row, 0]}"
        )


if __name__ == "__main__":
    main()
