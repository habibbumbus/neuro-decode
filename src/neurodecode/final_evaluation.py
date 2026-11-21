"""Frozen evaluation mechanics; no loading, training, selection or implicit test access."""

import hashlib
import math

import numpy as np

from neurodecode.evaluation import (
    VelocitySeries,
    evaluate_velocity,
    targets_from_windows,
)
from neurodecode.models.baselines import predict_zero
from neurodecode.models.gru import gru_inputs
from neurodecode.models.kalman import current_bin_observations, filter_observations
from neurodecode.models.ridge import ridge_inputs, scaler_sha256
from neurodecode.streaming import SequentialDecoder, iter_replay_bins

MODEL_ORDER = ("zero", "mean", "ridge", "kalman", "gru")


def authorize_operation(operation: str, token: str | None, freeze_sha256: str) -> None:
    """Fail before loading any data unless test has the explicit freeze token."""
    if operation not in ("preflight", "validation", "test"):
        raise ValueError("Unknown operation")
    if operation == "test" and token != freeze_sha256:
        raise ValueError("Test access denied: exact --open-test-sha256 token required")
    if operation != "test" and token is not None:
        raise ValueError("Test-opening token is only valid for explicit test mode")


def independent_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    """Python fsum implementation independent of the vectorized shared scorer."""
    if (
        actual.shape != prediction.shape
        or actual.ndim != 2
        or actual.shape[1] != 2
        or len(actual) == 0
    ):
        raise ValueError("Need matched nonempty [N,2] arrays")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite independent metric input")
    n, mse, r2 = len(actual), [], []
    for axis in range(2):
        y = [float(v) for v in actual[:, axis]]
        p = [float(v) for v in prediction[:, axis]]
        sse = math.fsum((b - a) ** 2 for a, b in zip(y, p, strict=True))
        mean = math.fsum(y) / n
        sst = math.fsum((a - mean) ** 2 for a in y)
        mse.append(sse / n)
        r2.append(
            None if n < 2 or all(a == y[0] for a in y) or sst == 0 else 1 - sse / sst
        )
    return {
        "scored_examples": n,
        "rmse_mm_s": [math.sqrt(v) for v in mse],
        "overall_rmse_mm_s": math.sqrt(math.fsum(mse) / 2),
        "r2": r2,
        "mean_r2": None if any(v is None for v in r2) else math.fsum(r2) / 2,
    }


def check_metrics(actual, prediction, shared, tolerance=1e-10) -> dict:
    independent = independent_metrics(actual, prediction)
    if independent["scored_examples"] != shared["scored_examples"]:
        raise ValueError("Independent scorer count mismatch")
    for key in ("rmse_mm_s", "overall_rmse_mm_s", "r2", "mean_r2"):
        a, b = independent[key], shared[key]
        if isinstance(a, list):
            for left, right in zip(a, b, strict=True):
                if left is None or right is None:
                    if left is not right:
                        raise ValueError("Undefined metric status mismatch")
                elif not math.isclose(left, right, rel_tol=0, abs_tol=tolerance):
                    raise ValueError("Independent metric mismatch")
        elif a is None or b is None:
            if a is not b:
                raise ValueError("Undefined mean metric mismatch")
        elif not math.isclose(a, b, rel_tol=0, abs_tol=tolerance):
            raise ValueError("Independent aggregate metric mismatch")
    return independent


def verify_support(targets, windows, specification, timing, partition) -> dict:
    """Independent row/support checks against fixed retained partition and segments."""
    if windows.partition != partition or targets.bins.partition != partition:
        raise ValueError("Partition identity mismatch")
    part = next(p for p in specification["partitions"] if p["name"] == partition)
    lower, upper = part["retained"]["start_s"], part["retained"]["end_s"]
    rows = windows.target_rows
    if not len(rows) or not targets.valid_mask[rows].all():
        raise ValueError("Need nonempty eligible targets only")
    expected_rows = rows[:, None] - np.arange(timing.history_bins - 1, -1, -1)
    np.testing.assert_array_equal(windows.source_rows, expected_rows)
    np.testing.assert_array_equal(windows.X_raw, targets.bins.counts[expected_rows])
    np.testing.assert_array_equal(windows.y, targets.velocity_mm_s[rows])
    np.testing.assert_array_equal(windows.target_time_s, targets.bins.bin_end_s[rows])
    np.testing.assert_array_equal(
        windows.source_bin_end_indices, targets.bins.grid_end_indices[expected_rows]
    )
    if np.any(windows.source_bin_start_s < lower) or np.any(
        windows.source_bin_end_s >= upper
    ):
        raise ValueError("Neural support crosses retained partition")
    if np.any(windows.target_time_s < lower) or np.any(windows.target_time_s >= upper):
        raise ValueError("Target timestamp crosses retained partition")
    for segment_id in np.unique(windows.segment_ids):
        choose = windows.segment_ids == segment_id
        segment = part["continuous_segments"][int(segment_id)]
        if (
            np.any(windows.source_bin_start_s[choose] < segment["start_s"])
            or np.any(windows.source_bin_end_s[choose] >= segment["end_s"])
            or np.any(targets.source_time_s[rows[choose]] < segment["start_s"])
            or np.any(targets.source_time_s[rows[choose]] >= segment["end_s"])
        ):
            raise ValueError("Neural/behavioral support crosses segment or partition")
    if np.any(
        targets.source_time_s[rows, 0] > targets.bins.bin_start_s[rows]
    ) or np.any(targets.source_time_s[rows, 1] > targets.bins.bin_end_s[rows]):
        raise ValueError("Future behavior sample in target support")
    return {
        "partition": partition,
        "eligible_rows": len(rows),
        "support": "all neural/behavioral support within retained partition and segment; no future target sample",
    }


def parameter_digest(model, kind) -> str:
    digest = hashlib.sha256()
    if kind == "gru":
        arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    elif kind == "ridge":
        arrays = {"weights": model.weights, "intercept": model.intercept}
    else:
        arrays = {
            k: getattr(model, k)
            for k in (
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
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    digest.update(scaler_sha256(model.scaler).encode())
    return digest.hexdigest()


def evaluate_prepared(
    targets, windows, specification, timing, models, mean_model, tolerances
):
    """Predict all five fixed models, verify identical replay rows, then score once."""
    partition = windows.partition
    support = verify_support(targets, windows, specification, timing, partition)
    labels = targets_from_windows(windows)
    index = labels.index
    scaler = models["ridge"].scaler
    before = {name: parameter_digest(model, name) for name, model in models.items()}
    mean_before = mean_model.mean_mm_s.copy()
    offline = {"zero": predict_zero(index), "mean": mean_model.predict(index)}
    offline["ridge"] = models["ridge"].predict(ridge_inputs(windows, scaler))
    offline["gru"] = models["gru"].predict(gru_inputs(windows, scaler), batch_size=1)
    filtered = filter_observations(
        models["kalman"],
        current_bin_observations(
            windows,
            scaler,
            specification["source"]["session"],
            specification["bin_grid"]["origin_s"],
            timing.bin_s,
        ),
    )
    offline["kalman"] = filtered.predictions
    sequential = {"zero": predict_zero(index), "mean": mean_model.predict(index)}
    equivalence, resets = {}, {}
    # Each stream sees one packet at a time; movement-valued labels are absent.
    for name, model in models.items():
        stream = SequentialDecoder(
            model, specification["bin_grid"]["origin_s"], timing.bin_s
        )
        results = [
            r
            for p in iter_replay_bins(targets, specification, timing)
            if (r := stream.step(p)) is not None
        ]
        if len(results) != len(index.bin_ids):
            raise ValueError("Model failed to emit the common eligible population")
        for row, result in enumerate(results):
            if (
                result.bin_id != index.bin_ids[row]
                or result.time_s != index.time_s[row]
                or result.partition != partition
                or result.segment_id != index.segment_ids[row]
                or result.source_bin_ids != tuple(windows.source_bin_end_indices[row])
            ):
                raise ValueError("Replay/offline row or source identity mismatch")
        sequential[name] = VelocitySeries(
            np.array([r.velocity_mm_s for r in results]), index
        )
        resets[name] = [r.bin_id for r in results if r.reset]
        if name == "kalman":
            np.testing.assert_array_equal([r.reset for r in results], filtered.reset)
    scores, independent = {}, {}
    for name in MODEL_ORDER:
        atol, rtol = tolerances[name]["atol_mm_s"], tolerances[name]["rtol"]
        expected, observed = offline[name].values_mm_s, sequential[name].values_mm_s
        difference = float(np.abs(expected - observed).max())
        if not np.allclose(expected, observed, atol=atol, rtol=rtol):
            raise ValueError(f"Offline/replay integrity failure: {name}")
        equivalence[name] = {
            "maximum_absolute_difference_mm_s": difference,
            "rows": len(observed),
            "atol_mm_s": atol,
            "rtol": rtol,
        }
        scores[name] = evaluate_velocity(labels, sequential[name])
        independent[name] = check_metrics(labels.values_mm_s, observed, scores[name])
    after = {name: parameter_digest(model, name) for name, model in models.items()}
    if before != after or not np.array_equal(mean_before, mean_model.mean_mm_s):
        raise ValueError("Frozen model parameters/transform mutated during inference")
    return {
        "labels": labels,
        "offline": offline,
        "sequential": sequential,
        "metrics": scores,
        "integrity": {
            "support": support,
            "equivalence": equivalence,
            "independent_metrics": independent,
            "parameter_digests_before": before,
            "parameter_digests_after": after,
            "resets": resets,
        },
    }
