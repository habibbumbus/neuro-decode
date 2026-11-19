"""Independently audit Phase 6B CSV scores, support, clean values and perturbations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_gru import read_rows
from replay_validation import sha, write_json

from neurodecode.data.features import FrozenCountScaler
from neurodecode.robustness import Disruption, perturb_packet
from neurodecode.streaming import ReplayBin


def independent_scores(actual, predicted):
    error = predicted - actual
    mse = np.sum(error * error, axis=0) / len(actual)
    sst = np.sum((actual - np.mean(actual, axis=0)) ** 2, axis=0)
    r2 = 1 - np.sum(error * error, axis=0) / sst
    return np.sqrt(mse), float(np.sqrt(np.mean(mse))), r2, float(np.mean(r2))


def check_metrics(actual, predicted, expected):
    axis_rmse, overall, r2, mean_r2 = independent_scores(actual, predicted)
    assert len(actual) == expected["scored_examples"]
    np.testing.assert_allclose(axis_rmse, expected["rmse_mm_s"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        overall, expected["overall_rmse_mm_s"], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(r2, expected["r2"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(mean_r2, expected["mean_r2"], rtol=0, atol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    output = (Path("runs") / args.run_id).resolve()
    if not output.is_relative_to(Path("runs").resolve()):
        raise ValueError("Audit trusted project outputs only")
    metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in metadata["artifact_sha256"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or sha(path) != expected:
            raise ValueError(f"Changed evidence: {relative}")
    reference_path = Path("runs/phase6a-validation-timing/predictions.csv")
    if sha(reference_path) != manifest["phase6a_predictions_sha256"]:
        raise ValueError("Clean reference changed")
    reference = read_rows(reference_path)
    actual = np.array(
        [[float(r[f"actual_v{a}_mm_s"]) for a in ("x", "y")] for r in reference]
    )
    scaler = FrozenCountScaler.from_metadata(manifest["normalization"])
    primary_ids = manifest["primary_scoring_bin_ids"]
    input_start = int(reference[0]["first_source_bin_id"])
    primary_checks, input_checks = 0, 0
    for record in metadata["records"]:
        if record["status"] != "complete":
            continue
        condition = Disruption(
            **{
                **record["condition"],
                "channel_ids": tuple(record["condition"]["channel_ids"]),
            }
        )
        rows = read_rows(output / condition.name / "predictions.csv")
        assert len(rows) == len(reference)
        primary = np.array([r["primary_score"] == "True" for r in rows])
        available = np.array([r["emitted"] == "True" for r in rows])
        assert [
            int(r["bin_id"]) for r in rows if r["primary_score"] == "True"
        ] == primary_ids
        assert [int(r["bin_id"]) for r in rows if r["emitted"] == "True"] == manifest[
            "available_scoring_bin_ids"
        ][condition.name]
        for row, saved in zip(rows, reference, strict=True):
            for field in (
                "partition",
                "segment_id",
                "bin_id",
                "time_s",
                "first_source_bin_id",
                "last_source_bin_id",
                "actual_vx_mm_s",
                "actual_vy_mm_s",
            ):
                assert row[field] == saved[field], (condition.name, field)
        for name in ("ridge", "gru", "kalman"):
            predicted = np.array(
                [
                    [
                        float(r[f"{name}_v{a}_mm_s"])
                        if r["emitted"] == "True"
                        else np.nan
                        for a in ("x", "y")
                    ]
                    for r in rows
                ]
            )
            assert np.isfinite(predicted[available]).all()
            assert all(
                r[f"{name}_v{a}_mm_s"] == ""
                for r in rows
                if r["emitted"] == "False"
                for a in ("x", "y")
            )
            check_metrics(
                actual[primary], predicted[primary], record["primary_metrics"][name]
            )
            check_metrics(
                actual[available],
                predicted[available],
                record["available_metrics"][name],
            )
            clean = np.array(
                [[float(r[f"{name}_v{a}_mm_s"]) for a in ("x", "y")] for r in reference]
            )
            check_metrics(
                actual[available],
                clean[available],
                record["paired_clean_metrics"][name],
            )
            if condition.kind == "clean":
                np.testing.assert_array_equal(predicted, clean)
            primary_checks += 1
        input_rows = read_rows(output / condition.name / "affected_inputs.csv")
        assert [int(r["bin_id"]) for r in input_rows] == manifest[
            "affected_raw_bin_ids"
        ][condition.name]
        for row in input_rows:
            bin_id = int(row["bin_id"])
            origin = manifest["split_specification"]["bin_grid"]["origin_s"]
            width = manifest["split_specification"]["bin_grid"]["width_s"]
            packet = ReplayBin(
                np.array(json.loads(row["raw_before_json"]), dtype=np.int64),
                bin_id - input_start,
                bin_id,
                origin + (bin_id - 1) * width,
                origin + bin_id * width,
                0,
                "validation",
                manifest["source"]["session"],
                scaler.channel_ids,
                scaler.unit_indices,
                scaler.identity,
                True,
                True,
            )
            perturbed = perturb_packet(packet, condition, scaler.means)
            expected = json.loads(row["raw_after_json"])
            if perturbed.missing:
                assert expected is None and row["missing"] == "True"
                payload = b"missing"
            else:
                np.testing.assert_array_equal(perturbed.counts, expected)
                assert row["missing"] == "False" and (perturbed.counts >= 0).all()
                payload = perturbed.counts.astype("<i8").tobytes()
            assert (
                hashlib.sha256(payload).hexdigest()
                == manifest["input_packet_hashes"][condition.name][bin_id - input_start]
            )
            input_checks += 1
        affected = set(manifest["affected_raw_bin_ids"][condition.name])
        for i, value in enumerate(manifest["input_packet_hashes"][condition.name]):
            if i + input_start not in affected:
                assert value == manifest["input_packet_hashes"]["clean"][i]
    result = {
        "scope": "saved-artifact audit only; no recording load, inference, fitting or test access",
        "primary_model_conditions_audited": primary_checks,
        "affected_input_rows_regenerated": input_checks,
        "metric_agreement": "independent RMSE/R2 on primary, available and paired-clean populations agree within 1e-10",
        "identities_and_missing_outputs": "exact original targets/IDs; blank unavailable predictions; identical primary mask",
        "perturbations": "regenerated exactly; outside declared bin intervals input hashes equal clean",
        "failures_recorded": metadata["failures"],
        "run_json_sha256": sha(output / "run.json"),
        "audit_sha256_utf8_lf": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").encode()
        ).hexdigest(),
    }
    write_json(output / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
