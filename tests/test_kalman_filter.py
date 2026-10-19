"""Hand arithmetic, causal filtering, numerical stability and identity checks."""

import json
from dataclasses import replace

import numpy as np
import pytest

from neurodecode.data.features import FrozenCountScaler, HistoryWindows
from neurodecode.evaluation import ExampleIndex, VelocitySeries, evaluate_velocity
from neurodecode.models.kalman import (
    CurrentBinObservations,
    KalmanFitConfig,
    KalmanParameters,
    current_bin_observations,
    filter_observations,
    kalman_step,
    normalization_sha256,
)


def system(n=200):
    scaler = FrozenCountScaler(
        np.zeros(2),
        np.ones(2),
        np.zeros(2, dtype=bool),
        ("M1 001", "M1 002"),
        ((0,), (0,)),
        10,
        (("source_sha256", "synthetic"),),
        100,
        "synthetic",
        "synthetic",
    )
    p = KalmanParameters(
        np.eye(2) * 0.9,
        np.array([0.1, -0.2]),
        np.eye(2),
        np.array([0.3, -0.1]),
        np.eye(2) * 0.1,
        np.eye(2) * 0.5,
        np.array([1.0, -1.0]),
        np.eye(2) * 2,
        scaler,
        KalmanFitConfig(),
        {"grid_origin_s": 0.0, "bin_width_s": 0.02},
    )
    rng = np.random.default_rng(87)
    states = np.empty((n, 2))
    states[0] = p.initial_mean
    for row in range(1, n):
        states[row] = p.A @ states[row - 1] + p.a + rng.normal(size=2) * np.sqrt(0.1)
    values = states @ p.H.T + p.d + rng.normal(size=(n, 2)) * np.sqrt(0.5)
    ids = np.arange(1, n + 1, dtype=np.int64)
    index = ExampleIndex(
        ids * 0.02,
        ids,
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=bool),
        "validation",
        scaler.channel_ids,
        scaler.unit_indices,
        scaler.identity,
    )
    observations = CurrentBinObservations(
        values,
        index,
        (ids - 1) * 0.02,
        ("synthetic",) * n,
        0.0,
        0.02,
        normalization_sha256(scaler),
    )
    return p, observations, states


def test_hand_calculated_predict_update_and_joseph_covariance():
    p, _, _ = system(2)
    p = replace(
        p,
        A=np.eye(2),
        a=np.zeros(2),
        H=np.eye(2),
        d=np.zeros(2),
        Q=np.eye(2),
        R=np.eye(2),
    )
    step = kalman_step(p, np.array([8.0, -4.0]), np.eye(2) * 3, np.array([10.0, -2.0]))
    np.testing.assert_allclose(step.predicted_mean, [8, -4])
    np.testing.assert_allclose(step.predicted_covariance, np.eye(2) * 4)
    np.testing.assert_allclose(step.innovation, [2, 2])
    np.testing.assert_allclose(step.gain, np.eye(2) * 0.8)
    np.testing.assert_allclose(step.mean, [9.6, -2.4])
    np.testing.assert_allclose(step.covariance, np.eye(2) * 0.8)
    assert step.solve_relative_residual < 1e-15


def test_reset_updates_training_prior_without_extra_transition():
    p, obs, _ = system(3)
    result = filter_observations(p, obs)
    first = kalman_step(
        p, p.initial_mean, p.initial_covariance, obs.values[0], predict=False
    )
    np.testing.assert_array_equal(result.predictions.values_mm_s[0], first.mean)
    np.testing.assert_array_equal(result.traces[0][1].predicted_mean, p.initial_mean)
    assert result.reset.tolist() == [True, False, False]
    assert result.steps_since_reset.tolist() == [1, 2, 3]


def test_small_system_finite_symmetric_and_better_than_noisy_observation():
    p, obs, states = system(1000)
    result = filter_observations(p, obs)
    naive = obs.values - p.d
    assert np.mean((result.predictions.values_mm_s - states) ** 2) < np.mean(
        (naive - states) ** 2
    )
    assert np.isfinite(result.predictions.values_mm_s).all()
    np.testing.assert_array_equal(
        result.covariance, result.covariance.transpose(0, 2, 1)
    )
    assert np.linalg.eigvalsh(result.covariance).min() > 0
    for _, step in result.traces:
        assert (
            np.linalg.eigvalsh(step.predicted_covariance - step.covariance).min()
            > -1e-10
        )


def test_missing_observation_is_prediction_only_not_zero_measurement():
    p, _, _ = system(2)
    missing = kalman_step(p, p.initial_mean, p.initial_covariance, None)
    zero = kalman_step(p, p.initial_mean, p.initial_covariance, np.zeros(2))
    np.testing.assert_array_equal(missing.mean, p.A @ p.initial_mean + p.a)
    np.testing.assert_allclose(
        missing.covariance, p.A @ p.initial_covariance @ p.A.T + p.Q
    )
    assert missing.innovation is None and missing.gain is None
    assert not np.array_equal(missing.mean, zero.mean)


def test_changing_future_neural_observations_preserves_earlier_predictions():
    p, obs, _ = system()
    values = obs.values.copy()
    values[80:] += [100, -200]
    original = filter_observations(p, obs)
    changed = filter_observations(p, replace(obs, values=values))
    np.testing.assert_array_equal(
        original.predictions.values_mm_s[:80], changed.predictions.values_mm_s[:80]
    )
    np.testing.assert_array_equal(original.covariance, changed.covariance)
    assert not np.array_equal(
        original.predictions.values_mm_s[80:], changed.predictions.values_mm_s[80:]
    )


@pytest.mark.parametrize("case", ["gap", "segment", "session", "invalid"])
def test_resets_at_discontinuity_prevent_state_carry(case):
    p, obs, _ = system(20)
    index, starts, sessions = obs.index, obs.bin_start_s, obs.session_ids
    if case == "gap":
        ids = index.bin_ids.copy()
        ids[10:] += 5
        index = replace(index, bin_ids=ids, time_s=ids * 0.02)
        starts = (ids - 1) * 0.02
    elif case == "segment":
        segments = index.segment_ids.copy()
        segments[10:] = 1
        index = replace(index, segment_ids=segments)
    elif case == "session":
        sessions = ("first",) * 10 + ("second",) * 10
    else:
        mask = index.score_mask.copy()
        mask[9] = False
        index = replace(index, score_mask=mask)
    obs = replace(obs, index=index, bin_start_s=starts, session_ids=sessions)
    result = filter_observations(p, obs)
    expected = kalman_step(
        p, p.initial_mean, p.initial_covariance, obs.values[10], predict=False
    )
    assert result.reset[10] and result.steps_since_reset[10] == 1
    np.testing.assert_array_equal(result.predictions.values_mm_s[10], expected.mean)
    changed = obs.values.copy()
    changed[:10] += 1000
    again = filter_observations(p, replace(obs, values=changed))
    np.testing.assert_array_equal(
        result.predictions.values_mm_s[10:], again.predictions.values_mm_s[10:]
    )
    if case == "invalid":
        assert result.status[9] == "not_emitted_invalid"
        assert not result.predictions.index.score_mask[9]
        assert result.steps_since_reset[9] == 0


def test_partition_calls_start_fresh_and_validation_labels_only_affect_scores():
    p, obs, states = system()
    first = filter_observations(
        p, replace(obs, index=replace(obs.index, partition="train"))
    )
    validation = filter_observations(p, obs)
    np.testing.assert_array_equal(
        first.predictions.values_mm_s, validation.predictions.values_mm_s
    )
    targets = VelocitySeries(states, obs.index)
    changed = replace(targets, values_mm_s=states + 1000)
    original_score = evaluate_velocity(targets, validation.predictions)
    changed_score = evaluate_velocity(changed, validation.predictions)
    assert original_score != changed_score
    repeated = filter_observations(p, obs)
    np.testing.assert_array_equal(
        validation.predictions.values_mm_s, repeated.predictions.values_mm_s
    )


def test_save_load_identical_predictions_and_prior(tmp_path):
    p, obs, _ = system()
    artifact = tmp_path / "kalman.json"
    p.save(artifact)
    restored = KalmanParameters.load(artifact, expected_scaler=p.scaler)
    before, after = filter_observations(p, obs), filter_observations(restored, obs)
    np.testing.assert_array_equal(
        before.predictions.values_mm_s, after.predictions.values_mm_s
    )
    np.testing.assert_array_equal(before.covariance, after.covariance)
    np.testing.assert_array_equal(restored.initial_mean, p.initial_mean)
    assert not restored.initial_mean.flags.writeable
    with pytest.raises(FileExistsError):
        p.save(artifact)


@pytest.mark.parametrize(
    "case", ["channels", "transform", "grid", "order", "nonfinite"]
)
def test_reject_incompatible_observations(case):
    p, obs, _ = system(5)
    if case == "channels":
        obs = replace(
            obs, index=replace(obs.index, channel_ids=obs.index.channel_ids[::-1])
        )
    elif case == "transform":
        obs = replace(obs, normalization_sha256="wrong")
    elif case == "grid":
        obs = replace(obs, grid_origin_s=1.0)
    elif case == "order":
        obs = replace(obs, index=replace(obs.index, time_s=obs.index.time_s[::-1]))
    else:
        values = obs.values.copy()
        values[0, 0] = np.nan
        obs = replace(obs, values=values)
    with pytest.raises(ValueError):
        filter_observations(p, obs)


@pytest.mark.parametrize("case", ["policy", "hash", "channels", "covariance"])
def test_reject_incompatible_saved_parameters(tmp_path, case):
    p, _, _ = system(5)
    artifact = tmp_path / "kalman.json"
    p.save(artifact)
    metadata = json.loads(artifact.read_text())
    if case == "policy":
        metadata["initial_state_policy"]["held_out_behavior_input"] = True
    elif case == "hash":
        metadata["normalization_sha256"] = "wrong"
    elif case == "channels":
        metadata["normalization"]["channel_ids"].reverse()
    else:
        metadata["matrices"]["Q"][0][0] = -1
    artifact.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        KalmanParameters.load(artifact, expected_scaler=p.scaler)


def test_silent_channel_with_tiny_R_is_retained_and_has_zero_gain():
    p, obs, _ = system()
    H = p.H.copy()
    H[1] = 0
    p = replace(p, H=H, R=np.diag([0.5, 1e-8]))
    result = filter_observations(p, obs)
    assert result.predictions.index.channel_ids == p.scaler.channel_ids
    for _, step in result.traces:
        np.testing.assert_array_equal(step.gain[:, 1], 0)
    assert result.diagnostics["maximum_relative_innovation_solve_residual"] < 1e-12


def test_observation_grid_cannot_change_from_fitted_model():
    p, obs, _ = system(5)
    p = replace(p, diagnostics={"grid_origin_s": 0.0, "bin_width_s": 0.04})
    with pytest.raises(ValueError, match="fitted Kalman timing"):
        filter_observations(p, obs)


def test_nonsquare_observation_matrix_matches_independent_state_space_solve():
    p, _, _ = system(2)
    scaler = replace(
        p.scaler,
        means=np.zeros(3),
        scales=np.ones(3),
        constant_channels=np.zeros(3, dtype=bool),
        channel_ids=("M1 001", "M1 002", "M1 003"),
        unit_indices=((0,),) * 3,
    )
    p = replace(
        p,
        scaler=scaler,
        H=np.array([[1.0, 0.2], [-0.4, 0.8], [0.7, -1.0]]),
        d=np.array([0.1, 0.3, -0.2]),
        R=np.diag([0.2, 0.4, 1e-8]),
    )
    z = np.array([2.0, -1.0, 3.0])
    step = kalman_step(p, p.initial_mean, p.initial_covariance, z)
    precision = np.linalg.solve(step.predicted_covariance, np.eye(2))
    expected_precision = precision + p.H.T @ np.linalg.solve(p.R, p.H)
    expected = np.linalg.solve(
        expected_precision,
        precision @ step.predicted_mean + p.H.T @ np.linalg.solve(p.R, z - p.d),
    )
    np.testing.assert_allclose(step.mean, expected, atol=1e-8)
    assert step.gain.shape == (2, 3)
    assert np.linalg.eigvalsh(step.covariance).min() > 0


def test_filter_path_from_windows_never_reads_validation_velocity_values():
    p, _, _ = system(20)
    n, length, channels = 20, 10, 2
    ids = np.arange(length, length + n, dtype=np.int64)
    source_ids = ids[:, None] - np.arange(length - 1, -1, -1)
    windows = HistoryWindows(
        np.random.default_rng(6).poisson(1.0, size=(n, length, channels)),
        np.zeros((n, 2)),
        source_ids - 1,
        source_ids,
        (source_ids - 1) * 0.02,
        source_ids * 0.02,
        ids - 1,
        ids,
        ids * 0.02,
        np.zeros(n, dtype=np.int64),
        p.scaler.channel_ids,
        p.scaler.unit_indices,
        "validation",
        p.scaler.identity,
        {},
    )
    original = current_bin_observations(windows, p.scaler, "synthetic", 0.0, 0.02)
    # A deliberately unusable label array proves the inference path ignores it.
    changed = current_bin_observations(
        replace(windows, y=np.full((n, 2), np.nan)), p.scaler, "synthetic", 0.0, 0.02
    )
    np.testing.assert_array_equal(
        filter_observations(p, original).predictions.values_mm_s,
        filter_observations(p, changed).predictions.values_mm_s,
    )
