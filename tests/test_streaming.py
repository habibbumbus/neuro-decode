"""Synthetic histories, causality, strict identity and offline/replay agreement."""

from dataclasses import replace

import numpy as np
import pytest

from neurodecode.data.features import build_history_windows, fit_count_scaler
from neurodecode.evaluation import targets_from_windows
from neurodecode.models.gru import GRUDecoder, gru_inputs
from neurodecode.models.kalman import current_bin_observations, filter_observations
from neurodecode.models.ridge import RidgeDecoder, ridge_inputs
from neurodecode.streaming import HistoryBuffer, SequentialDecoder, iter_replay_bins
from neurodecode.training import cpu_training_runtime
from tests.test_features import inputs
from tests.test_kalman_filter import system


def fixture(kind, gap=False, invalid=False):
    train, spec, timing = inputs(gap=gap)
    targets, _, _ = inputs("validation", gap=gap)
    if gap:
        # The old helper's 0.2+0.1 start exceeds grid 15*0.02 by one ULP.
        # Specify this synthetic segment at its actual grid edge; production
        # preparation continues to reject such a mismatch without snapping.
        start = float(targets.bins.bin_start_s[3])
        spec["partitions"][1]["continuous_segments"][1]["start_s"] = start
        metadata = list(targets.bins.segment_metadata)
        metadata[1] = {**metadata[1], "start_s": start}
        targets = replace(
            targets, bins=replace(targets.bins, segment_metadata=tuple(metadata))
        )
    scaler = fit_count_scaler(train, spec, timing)
    training = build_history_windows(train, spec, timing)
    if kind == "ridge":
        model = RidgeDecoder.fit(
            ridge_inputs(training, scaler), targets_from_windows(training), scaler, 1.0
        )
    elif kind == "gru":
        with cpu_training_runtime(7):
            model = GRUDecoder(scaler)
    else:
        p, _, _ = system(2)
        model = replace(p, scaler=scaler)
    mask = np.ones(6, dtype=bool)
    if invalid:
        mask[2] = False
    windows = build_history_windows(targets, spec, timing, mask)
    packets = list(iter_replay_bins(targets, spec, timing, mask))
    return model, targets, spec, timing, windows, packets


def replay(model, packets):
    stream = SequentialDecoder(model, 0.0, 0.02)
    return [result for p in packets if (result := stream.step(p)) is not None]


@pytest.mark.parametrize("kind", ["ridge", "gru", "kalman"])
@pytest.mark.parametrize("case", ["ordinary", "gap", "invalid"])
def test_replay_matches_offline_with_exact_rows_order_warmup_and_resets(kind, case):
    model, _, _, _, windows, packets = fixture(
        kind, gap=case == "gap", invalid=case == "invalid"
    )
    with cpu_training_runtime(7):
        result = replay(model, packets)
        if kind == "ridge":
            expected = model.predict(ridge_inputs(windows, model.scaler)).values_mm_s
        elif kind == "gru":
            expected = model.predict(gru_inputs(windows, model.scaler)).values_mm_s
        else:
            filtered = filter_observations(
                model,
                current_bin_observations(windows, model.scaler, "synthetic", 0.0, 0.02),
            )
            expected = filtered.predictions.values_mm_s
            assert [r.reset for r in result] == filtered.reset.tolist()
    assert [r.bin_id for r in result] == windows.target_bin_end_indices.tolist()
    assert [r.time_s for r in result] == windows.target_time_s.tolist()
    assert [
        list(r.source_bin_ids) for r in result
    ] == windows.source_bin_end_indices.tolist()
    np.testing.assert_allclose(
        [r.velocity_mm_s for r in result], expected, atol=1e-6, rtol=1e-6
    )
    assert len(result) == {"ordinary": 4, "gap": 2, "invalid": 1}[case]


def test_history_order_frozen_scaler_and_caller_count_mutation():
    model, _, _, _, _, packets = fixture("ridge")
    history = HistoryBuffer(model.scaler, 0.0, 0.02)
    assert history.push(packets[0]) is None
    packets[0].counts[:] = 999
    assert history.push(packets[1]) is None
    window = history.push(packets[2])
    np.testing.assert_array_equal(window.X_raw[0], [[0, 1], [2, 3], [4, 5]])
    expected = (window.X_raw - model.scaler.means) / model.scaler.scales
    np.testing.assert_array_equal(model.scaler.transform(window), expected)
    assert np.isnan(window.y).all()


@pytest.mark.parametrize("kind", ["ridge", "gru", "kalman"])
def test_changing_future_packets_cannot_change_earlier_outputs(kind):
    model, _, _, _, _, packets = fixture(kind)
    changed = [
        replace(p, counts=p.counts + 100) if i >= 4 else p
        for i, p in enumerate(packets)
    ]
    with cpu_training_runtime(7):
        first, second = replay(model, packets), replay(model, changed)
    np.testing.assert_array_equal(
        [r.velocity_mm_s for r in first[:2]], [r.velocity_mm_s for r in second[:2]]
    )
    assert not np.array_equal(
        [r.velocity_mm_s for r in first[2:]], [r.velocity_mm_s for r in second[2:]]
    )


@pytest.mark.parametrize("kind", ["ridge", "gru", "kalman"])
@pytest.mark.parametrize("boundary", ["partition", "session", "segment", "missing_bin"])
def test_boundaries_discard_history_and_kalman_state(kind, boundary):
    model, _, _, _, _, packets = fixture(kind)
    prefix = packets[:3]
    suffix = packets[3:]
    if boundary == "partition":
        suffix = [replace(p, partition="train") for p in suffix]
    elif boundary == "session":
        suffix = [replace(p, session="another") for p in suffix]
    elif boundary == "segment":
        suffix = [replace(p, segment_id=1) for p in suffix]
    else:
        suffix = [
            replace(
                p,
                bin_id=p.bin_id + 2,
                start_s=(p.bin_id + 1) * 0.02,
                end_s=(p.bin_id + 2) * 0.02,
            )
            for p in suffix
        ]
    with cpu_training_runtime(7):
        combined = replay(model, prefix + suffix)
        fresh = replay(model, suffix)
    assert len(combined) == 2 and len(fresh) == 1
    assert combined[-1].reset
    np.testing.assert_array_equal(combined[-1].velocity_mm_s, fresh[0].velocity_mm_s)


@pytest.mark.parametrize(
    "change",
    ["channels", "units", "identity", "grid", "negative", "nonfinite", "order"],
)
def test_bad_packet_rejected_without_silent_sorting_or_repair(change):
    model, _, _, _, _, packets = fixture("ridge")
    stream = SequentialDecoder(model, 0.0, 0.02)
    packet = packets[0]
    if change == "channels":
        packet = replace(packet, channel_ids=packet.channel_ids[::-1])
    elif change == "units":
        packet = replace(packet, unit_indices=packet.unit_indices[::-1])
    elif change == "identity":
        packet = replace(packet, identity=(("source", "wrong"),))
    elif change == "grid":
        packet = replace(packet, end_s=packet.end_s + 1e-10)
    elif change == "negative":
        packet = replace(packet, counts=np.array([-1, 0]))
    elif change == "nonfinite":
        packet = replace(packet, counts=np.array([np.nan, 0]))
    else:
        stream.step(packet)
    with pytest.raises(ValueError):
        stream.step(packet)


def test_zero_counts_are_valid_and_movement_values_never_enter_packets():
    model, targets, spec, timing, _, packets = fixture("ridge")
    mutated = replace(
        targets, velocity_mm_s=np.nan_to_num(targets.velocity_mm_s) + 1000000
    )
    changed = list(iter_replay_bins(mutated, spec, timing))
    np.testing.assert_array_equal(
        [r.velocity_mm_s for r in replay(model, packets)],
        [r.velocity_mm_s for r in replay(model, changed)],
    )
    zero = [replace(p, counts=np.zeros_like(p.counts)) for p in packets]
    result = replay(model, zero)
    assert len(result) == 4
    assert all(np.isfinite(r.velocity_mm_s).all() for r in result)


def test_timing_boundaries_are_nested_without_changing_estimates():
    model, _, _, _, _, packets = fixture("ridge")
    stream = SequentialDecoder(model, 0.0, 0.02)
    measured = [
        result for p in packets if (result := stream.step(p, measure=True)) is not None
    ]
    ordinary = replay(model, packets)
    np.testing.assert_array_equal(
        [r.velocity_mm_s for r in measured], [r.velocity_mm_s for r in ordinary]
    )
    assert all(r.preparation_model_ns >= r.model_ns > 0 for r in measured)
