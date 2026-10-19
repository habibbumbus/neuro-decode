"""Synthetic parameter estimation and temporal-boundary tests; no filter loop."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from neurodecode.data.features import (
    FrozenCountScaler,
    fit_count_scaler,
    prepare_partition,
)
from neurodecode.data.loaders import Recording
from neurodecode.evaluation import ExampleIndex, VelocitySeries
from neurodecode.models.kalman import (
    CurrentBinObservations,
    KalmanFitConfig,
    consecutive_training_pairs,
    covariance_diagnostics,
    current_bin_observations,
    fit_kalman_parameters,
    normalization_sha256,
)
from neurodecode.splits import TimingConfig


def synthetic(n=2400):
    rng = np.random.default_rng(781)
    A = np.array([[0.75, 0.08], [-0.06, 0.65]])
    a = np.array([0.2, -0.1])
    H = np.array([[1.2, -0.4], [0.3, 0.9], [-0.8, 0.5]])
    d = np.array([0.3, -0.2, 1.1])
    q = np.array([0.04, 0.09])
    r = np.array([0.01, 0.02, 0.03])
    states = np.zeros((n + 400, 2))
    for row in range(1, len(states)):
        states[row] = A @ states[row - 1] + a + rng.normal(size=2) * np.sqrt(q)
    states = states[400:]
    neural = states @ H.T + d + rng.normal(size=(n, 3)) * np.sqrt(r)
    ids = np.arange(1, n + 1, dtype=np.int64)
    identity = (("source_sha256", "synthetic"),)
    channels = ("M1 001", "M1 002", "M1 003")
    scaler = FrozenCountScaler(
        np.zeros(3),
        np.ones(3),
        np.zeros(3, dtype=bool),
        channels,
        ((0,),) * 3,
        1,
        identity,
        n,
        "synthetic",
        "synthetic",
    )
    index = ExampleIndex(
        ids * 0.02,
        ids,
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=bool),
        "train",
        channels,
        scaler.unit_indices,
        identity,
    )
    observation = CurrentBinObservations(
        neural,
        index,
        (ids - 1) * 0.02,
        ("synthetic_session",) * n,
        0.0,
        0.02,
        normalization_sha256(scaler),
    )
    return (
        observation,
        VelocitySeries(states, index),
        scaler,
        TimingConfig(history_bins=1),
        (A, a, H, d, q, r),
    )


def fit_synthetic(n=2400):
    observations, states, scaler, timing, known = synthetic(n)
    result = fit_kalman_parameters(
        observations, states, scaler, timing, KalmanFitConfig()
    )
    return result, observations, states, scaler, timing, known


def test_known_linear_dynamical_system_parameter_recovery():
    p, _, _, _, _, (A, a, H, d, q, r) = fit_synthetic()
    np.testing.assert_allclose(p.A, A, atol=0.05)
    np.testing.assert_allclose(p.a, a, atol=0.04)
    np.testing.assert_allclose(p.H, H, atol=0.05)
    np.testing.assert_allclose(p.d, d, atol=0.05)
    np.testing.assert_allclose(np.diag(p.Q), q, atol=0.01)
    np.testing.assert_allclose(np.diag(p.R), r, atol=0.003)
    assert p.A.shape == p.Q.shape == (2, 2)
    assert p.H.shape == (3, 2) and p.R.shape == (3, 3)
    assert p.d.shape == (3,) and p.a.shape == (2,)
    assert p.diagnostics["eligible_training_transitions"] == 2399
    assert p.diagnostics["transition_fit"]["centered_predictor_rank"] == 2


@pytest.mark.parametrize("case", ["missing_bin", "segment", "session", "invalid_row"])
def test_transition_pairs_never_cross_discontinuities(case):
    observations, states, scaler, timing, _ = synthetic(20)
    index = observations.index
    starts = observations.bin_start_s.copy()
    sessions = observations.session_ids
    if case == "missing_bin":
        ids = index.bin_ids.copy()
        ids[10:] += 1
        index = replace(index, bin_ids=ids, time_s=ids * 0.02)
        starts = (ids - 1) * 0.02
    elif case == "segment":
        segments = index.segment_ids.copy()
        segments[10:] = 1
        index = replace(index, segment_ids=segments)
    elif case == "session":
        sessions = ("session_a",) * 10 + ("session_b",) * 10
    else:
        valid = index.score_mask.copy()
        valid[10] = False
        index = replace(index, score_mask=valid)
    observations = replace(
        observations, index=index, bin_start_s=starts, session_ids=sessions
    )
    states = replace(states, index=index)
    observations.validate(scaler)
    pairs = consecutive_training_pairs(observations)
    assert not np.any(np.all(pairs == [9, 10], axis=1))
    if case == "invalid_row":
        assert not np.any(np.all(pairs == [10, 11], axis=1))
    result = fit_kalman_parameters(
        observations, states, scaler, timing, KalmanFitConfig()
    )
    assert result.diagnostics["eligible_training_transitions"] == len(pairs)


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_nontraining_fit_is_rejected(partition):
    observations, states, scaler, timing, _ = synthetic(20)
    index = replace(observations.index, partition=partition)
    with pytest.raises(ValueError, match="training partition"):
        fit_kalman_parameters(
            replace(observations, index=index),
            replace(states, index=index),
            scaler,
            timing,
            KalmanFitConfig(),
        )


def test_changing_excluded_row_does_not_change_any_fitted_parameter():
    observations, states, scaler, timing, _ = synthetic(120)
    valid = observations.index.score_mask.copy()
    valid[60] = False
    index = replace(observations.index, score_mask=valid)
    observations, states = (
        replace(observations, index=index),
        replace(states, index=index),
    )
    first = fit_kalman_parameters(
        observations, states, scaler, timing, KalmanFitConfig()
    )
    changed_neural, changed_velocity = (
        observations.values.copy(),
        states.values_mm_s.copy(),
    )
    changed_neural[60] += 1e6
    changed_velocity[60] += 1e6
    second = fit_kalman_parameters(
        replace(observations, values=changed_neural),
        replace(states, values_mm_s=changed_velocity),
        scaler,
        timing,
        KalmanFitConfig(),
    )
    for name in ("A", "a", "H", "d", "Q", "R", "initial_mean", "initial_covariance"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_prior_uses_training_state_distribution_only():
    parameters, _, states, *_ = fit_synthetic(120)
    np.testing.assert_array_equal(
        parameters.initial_mean, states.values_mm_s.mean(axis=0)
    )
    np.testing.assert_allclose(
        parameters.initial_covariance, np.cov(states.values_mm_s, rowvar=False, ddof=0)
    )
    assert parameters.diagnostics["initial_covariance"]["diagonal_jitter_added"] == 0


def test_silent_channel_is_kept_with_explicit_positive_R_floor():
    observations, states, scaler, timing, _ = synthetic(300)
    values = observations.values.copy()
    values[:, 2] = 0
    p = fit_kalman_parameters(
        replace(observations, values=values), states, scaler, timing, KalmanFitConfig()
    )
    np.testing.assert_allclose(p.H[2], [0, 0], atol=1e-15)
    assert p.d[2] == 0
    assert p.R.shape == (3, 3)
    assert p.diagnostics["R"]["floored_channels"] == ["M1 003"]
    assert p.R[2, 2] == p.diagnostics["R"]["variance_floor"]
    assert p.diagnostics["R"]["raw_full_residual_covariance"]["numerical_rank"] == 2


def test_exact_affine_transition_regularizes_zero_residual_Q():
    observations, states, scaler, timing, _ = synthetic(120)
    t = np.arange(120) * 0.02
    velocity = np.column_stack((np.sin(t), np.cos(t)))
    states = replace(states, values_mm_s=velocity)
    p = fit_kalman_parameters(observations, states, scaler, timing, KalmanFitConfig())
    assert p.diagnostics["Q"]["diagonal_jitter_added"] > 0
    assert np.linalg.eigvalsh(p.Q)[0] > 0


def test_fitted_covariances_finite_symmetric_and_positive():
    p, *_ = fit_synthetic(120)
    for name in ("Q", "R", "initial_covariance"):
        m = getattr(p, name)
        assert np.isfinite(m).all()
        np.testing.assert_array_equal(m, m.T)
        assert np.linalg.eigvalsh(m).min() > 0
        assert m.flags.writeable is False
    np.testing.assert_array_equal(p.R, np.diag(np.diag(p.R)))


@pytest.mark.parametrize(
    "case", ["time", "channel", "transform", "shape", "nonfinite", "session_length"]
)
def test_mismatched_or_invalid_observations_fail(case):
    observations, states, scaler, timing, _ = synthetic(20)
    if case == "time":
        times = observations.index.time_s.copy()
        times[5] = np.nextafter(times[5], np.inf)
        observations = replace(
            observations, index=replace(observations.index, time_s=times)
        )
    elif case == "channel":
        observations = replace(
            observations,
            index=replace(
                observations.index, channel_ids=observations.index.channel_ids[::-1]
            ),
        )
    elif case == "transform":
        observations = replace(observations, normalization_sha256="wrong")
    elif case == "shape":
        observations = replace(observations, values=observations.values[:, :-1])
    elif case == "session_length":
        observations = replace(observations, session_ids=("only_one",))
    else:
        values = observations.values.copy()
        values[2, 0] = np.nan
        observations = replace(observations, values=values)
    with pytest.raises(ValueError):
        fit_kalman_parameters(observations, states, scaler, timing, KalmanFitConfig())


def test_target_mask_identity_mismatch_fails_without_intersection():
    observations, states, scaler, timing, _ = synthetic(20)
    valid = states.index.score_mask.copy()
    valid[5] = False
    states = replace(states, index=replace(states.index, score_mask=valid))
    with pytest.raises(ValueError, match="score_mask"):
        fit_kalman_parameters(observations, states, scaler, timing, KalmanFitConfig())


def test_rank_deficient_velocity_and_insufficient_data_fail():
    observations, states, scaler, timing, _ = synthetic(20)
    with pytest.raises(ValueError, match="rank-2"):
        fit_kalman_parameters(
            observations,
            replace(states, values_mm_s=np.ones((20, 2))),
            scaler,
            timing,
            KalmanFitConfig(),
        )
    observations, states, scaler, timing, _ = synthetic(4)
    with pytest.raises(ValueError, match="at least four"):
        fit_kalman_parameters(observations, states, scaler, timing, KalmanFitConfig())


@pytest.mark.parametrize(
    "matrix",
    [np.diag([1.0, -1.0]), np.array([[1.0, 2.0], [0.0, 1.0]]), np.full((2, 2), np.nan)],
)
def test_invalid_covariance_diagnostics_fail(matrix):
    with pytest.raises(ValueError):
        covariance_diagnostics(matrix)


def test_saved_artifact_contains_parameters_prior_and_context_only(tmp_path):
    p, *_ = fit_synthetic(120)
    path = tmp_path / "kalman_parameters.json"
    p.save(path)
    saved = json.loads(path.read_text())
    assert saved["model"] == "kalman_parameters_only"
    assert saved["initial_state_policy"]["held_out_behavior_input"] is False
    assert saved["normalization"] == p.scaler.to_metadata()
    np.testing.assert_array_equal(saved["matrices"]["A"], p.A)
    assert "filter_state" not in saved and "predictions" not in saved
    with pytest.raises(FileExistsError):
        p.save(path)


def prepared_inputs():
    rng = np.random.default_rng(91)
    timing = TimingConfig(history_bins=2)
    time = 1 + np.arange(1001) * 0.004
    position = np.column_stack(
        (
            10 * np.sin(3 * time) + 2 * np.cos(11 * time),
            7 * np.cos(5 * time) + 1.5 * np.sin(7 * time),
        )
    )
    recording = Recording(
        time,
        position,
        ("M1 001", "M1 002"),
        tuple((np.sort(rng.uniform(1, 5, 1400)),) for _ in range(2)),
        {"spike_range_s": [1.0, 5.0]},
    )
    spec = {
        "schema_version": 1,
        "source": {"sha256": "synthetic"},
        "configuration": {"sha256": "synthetic"},
        "timing": asdict(timing),
        "overlap": {"start_s": 1.0, "end_s": 5.0},
        "bin_grid": {"origin_s": 1.0, "width_s": timing.bin_s},
        "partitions": [
            {
                "name": name,
                "retained": {"start_s": a, "end_s": b},
                "continuous_segments": [{"start_s": a, "end_s": b}],
            }
            for name, a, b in (
                ("train", 1.0, 3.0),
                ("validation", 3.2, 4.0),
                ("test", 4.2, 5.0),
            )
        ],
    }
    return recording, spec, timing


def fit_prepared(recording, spec, timing):
    aligned, windows = prepare_partition(recording, spec, timing, "train")
    scaler = fit_count_scaler(aligned, spec, timing)
    observations = current_bin_observations(
        windows, scaler, "synthetic", 1.0, timing.bin_s
    )
    from neurodecode.evaluation import targets_from_windows

    fitted = fit_kalman_parameters(
        observations, targets_from_windows(windows), scaler, timing, KalmanFitConfig()
    )
    return fitted, observations, windows, aligned, scaler


def test_current_bin_factory_matches_original_count_row_and_ignores_labels():
    recording, spec, timing = prepared_inputs()
    _, observations, windows, aligned, scaler = fit_prepared(recording, spec, timing)
    raw = aligned.bins.counts[windows.target_rows]
    np.testing.assert_array_equal(raw, windows.X_raw[:, -1, :])
    np.testing.assert_array_equal(
        observations.values, (raw - scaler.means) / scaler.scales
    )
    np.testing.assert_array_equal(
        observations.index.time_s, aligned.bins.bin_end_s[windows.target_rows]
    )
    np.testing.assert_array_equal(
        observations.bin_start_s, aligned.bins.bin_start_s[windows.target_rows]
    )
    assert not np.array_equal(raw, windows.X_raw[:, 0, :])
    changed = replace(windows, y=np.full_like(windows.y, np.nan))
    without_labels = current_bin_observations(
        changed, scaler, "synthetic", 1.0, timing.bin_s
    )
    np.testing.assert_array_equal(observations.values, without_labels.values)


@pytest.mark.parametrize("interval", [(3.2, 4.0), (4.2, 5.0)])
def test_heldout_raw_activity_and_movement_cannot_change_training_fit(interval):
    recording, spec, timing = prepared_inputs()
    original, _, _, _, original_scaler = fit_prepared(recording, spec, timing)
    a, b = interval
    positions = recording.cursor_position_mm.copy()
    positions[
        (recording.behavior_times_s >= a) & (recording.behavior_times_s <= b)
    ] += [10000, -20000]
    trains = []
    for units in recording.spike_times_s:
        events = units[0]
        keep = events[(events < a) | (events > b)]
        changed_events = np.linspace(a + 0.001, b - 0.001, 3500)
        trains.append((np.sort(np.concatenate((keep, changed_events))),))
    changed = replace(
        recording, cursor_position_mm=positions, spike_times_s=tuple(trains)
    )
    fitted, _, _, _, fitted_scaler = fit_prepared(changed, spec, timing)
    assert fitted_scaler.to_metadata() == original_scaler.to_metadata()
    for name in ("A", "a", "H", "d", "Q", "R", "initial_mean", "initial_covariance"):
        np.testing.assert_array_equal(getattr(original, name), getattr(fitted, name))
