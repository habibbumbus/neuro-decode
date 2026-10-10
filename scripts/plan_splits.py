"""Verify acquired bytes and save a timestamp-only Phase 2A split specification."""

import hashlib
import json
import tomllib
from pathlib import Path

from neurodecode.data.loaders import load_recording
from neurodecode.splits import plan_splits, read_config


def main() -> None:
    manifest = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    if len(manifest["recordings"]) != 1:
        raise ValueError("This checkpoint expects exactly one recording")
    source = manifest["recordings"][0]
    path = Path(source["local_path"])
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != source["local_checksums"]["SHA256"]:
        raise ValueError("Recording differs from acquisition manifest; stop")
    config_path = Path("configs/base.toml")
    config, fractions = read_config(config_path)
    recording = load_recording(path)
    plan = plan_splits(
        recording.behavior_times_s,
        recording.diagnostics["spike_range_s"],
        config,
        fractions,
    )
    # Only timestamps and loader quality metadata determine splits, never positions.
    exploration = tomllib.loads(config_path.read_text(encoding="utf-8"))["exploration"]
    if exploration["session"] != source["session"]:
        raise ValueError("Exploration ledger belongs to a different recording")
    train = plan["partitions"][0]["retained"]
    if not (
        train["start_s"]
        <= exploration["start_s"]
        < exploration["end_s"]
        <= train["end_s"]
    ):
        raise ValueError(
            "Previously explored interval conflicts with retained training"
        )
    plan["source"] = {
        "record_id": manifest["dataset"]["record_id"],
        "version": manifest["dataset"]["version"],
        "record_url": manifest["dataset"]["record_url"],
        "session": source["session"],
        "filename": source["filename"],
        "sha256": digest,
        "actual_size_bytes": path.stat().st_size,
    }
    plan["configuration"] = {
        "path": config_path.as_posix(),
        "sha256": hashlib.sha256(
            config_path.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest(),
        "hash_encoding": "UTF-8 text with LF newlines",
    }
    plan["quality_metadata"] = {
        key: recording.diagnostics[key]
        for key in (
            "behavior_range_s",
            "sample_interval_s",
            "spike_range_s",
            "gap_after_sample_indices",
        )
    }
    plan["prior_exploration"] = {
        **exploration,
        "belongs_to_training": True,
        "evidence": "docs/progress.md Phase 1C; reviewed PNG/JSON in runs/phase1c-verified",
    }
    output = Path("data/splits") / f"{source['session']}.json"
    serialized = json.dumps(plan, indent=2, allow_nan=False) + "\n"
    # A changed plan must be reviewed explicitly; identical reruns reuse it.
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise ValueError(
            f"Different specification already exists: {output}; review before replacing"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(f"Specification: {output.as_posix()}")
    print(
        "name         nominal [s]             retained [s]            purge / gap discard [s]"
    )
    for partition in plan["partitions"]:
        nominal, retained = partition["nominal"], partition["retained"]
        print(
            f"{partition['name']:12} "
            f"[{nominal['start_s']:.3f}, {nominal['end_s']:.3f})  "
            f"[{retained['start_s']:.3f}, {retained['end_s']:.3f})  "
            f"{partition['purge_discarded_s']:.3f} / "
            f"{partition['gap_discarded_inside_retained_s']:.3f}"
        )
    print(
        f"Warm-up per continuous segment: {config.lookback_s:.3f} s; exact eligible indices pending"
    )
    print(
        "Previously explored interval belongs to retained training; no movement patterns used"
    )


if __name__ == "__main__":
    main()
