"""Verify provenance, bin training spikes, and print a fixed early example in memory."""

import hashlib
import json
from pathlib import Path

import numpy as np

from neurodecode.data.binning import bin_partition
from neurodecode.data.loaders import load_recording
from neurodecode.splits import plan_splits, read_config


def main() -> None:
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
