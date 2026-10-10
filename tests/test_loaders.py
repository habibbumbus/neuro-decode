"""Synthetic HDF5 fixtures only; no real recordings enter this test suite."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from neurodecode.data.loaders import RecordingFormatError, load_recording


def make_recording(path: Path) -> Path:
    """Write the observed reversed dimensions, cell references, and empty encoding."""
    with h5py.File(path, "w") as file:
        refs = file.create_group("#refs#")
        for name, values in {
            "t": [[10.0, 10.004, 10.008]],
            "cursor_pos": [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
        }.items():
            dataset = file.create_dataset(name, data=values, dtype="f8")
            dataset.attrs["MATLAB_class"] = np.bytes_("double")
        names = file.create_dataset("chan_names", (1, 2), dtype=h5py.ref_dtype)
        names.attrs["MATLAB_class"] = np.bytes_("cell")
        for c, name in enumerate(["M1 002", "M1 001"]):
            value = refs.create_dataset(
                f"name{c}",
                data=np.frombuffer(name.encode("utf-16-le"), dtype="<u2")[:, None],
            )
            value.attrs["MATLAB_class"] = np.bytes_("char")
            names[0, c] = value.ref
        spikes = file.create_dataset("spikes", (3, 2), dtype=h5py.ref_dtype)
        spikes.attrs["MATLAB_class"] = np.bytes_("cell")
        trains = [
            [[9.9, 10.001, 10.1], [], [10.002]],
            [[10.004, 10.004], [10.008], []],
        ]
        for c, channel in enumerate(trains):
            for u, train in enumerate(channel):
                if train:
                    value = refs.create_dataset(
                        f"spike{c}_{u}", data=[train], dtype="f8"
                    )
                else:
                    value = refs.create_dataset(
                        f"spike{c}_{u}", data=[0, 0], dtype="u8"
                    )
                    value.attrs["MATLAB_empty"] = np.uint8(1)
                value.attrs["MATLAB_class"] = np.bytes_("double")
                spikes[u, c] = value.ref
        # A broken waveform link makes unwanted waveform access fail immediately.
        file["wf"] = h5py.ExternalLink("unavailable-waveforms.h5", "/wf")
    return path


def test_extracts_orientation_identity_empty_trains_and_coverage(
    tmp_path: Path,
) -> None:
    recording = load_recording(make_recording(tmp_path / "synthetic.mat"))
    np.testing.assert_array_equal(recording.behavior_times_s, [10, 10.004, 10.008])
    np.testing.assert_array_equal(
        recording.cursor_position_mm, [[1, -1], [2, -2], [3, -3]]
    )
    assert recording.channel_ids == ("M1 002", "M1 001")
    np.testing.assert_array_equal(recording.spike_times_s[0][0], [9.9, 10.001, 10.1])
    np.testing.assert_array_equal(recording.spike_times_s[0][2], [10.002])
    np.testing.assert_array_equal(recording.spike_times_s[1][1], [10.008])
    assert recording.spike_times_s[0][1].shape == (0,)
    assert recording.diagnostics["spike_counts_by_channel_unit"] == [
        [3, 0, 1],
        [2, 1, 0],
    ]
    assert recording.diagnostics["events_before_behavior"] == 1
    assert recording.diagnostics["events_after_behavior"] == 1
    assert recording.diagnostics["spike_duplicate_events"] == [
        {"channel_id": "M1 001", "unit_index": 0, "count": 1}
    ]


def test_reports_behavior_gap_without_repair(tmp_path: Path) -> None:
    path = make_recording(tmp_path / "synthetic.mat")
    with h5py.File(path, "r+") as file:
        file["t"][0] = [10, 10.004, 10.020]
    recording = load_recording(path)
    assert recording.diagnostics["gap_after_sample_indices"] == [1]
    np.testing.assert_array_equal(recording.behavior_times_s, [10, 10.004, 10.020])


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([10, 10, 10.008], "duplicate"),
        ([10, 9, 11], "descending"),
        ([10, np.nan, 11], "nonfinite"),
    ],
)
def test_rejects_invalid_behavior_times(
    tmp_path: Path, values: list, message: str
) -> None:
    path = make_recording(tmp_path / "synthetic.mat")
    with h5py.File(path, "r+") as file:
        file["t"][0] = values
    with pytest.raises(RecordingFormatError, match=message):
        load_recording(path)


@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (3, 2)])
def test_rejects_wrong_behavior_position_dimensions(
    tmp_path: Path, shape: tuple
) -> None:
    path = make_recording(tmp_path / "synthetic.mat")
    with h5py.File(path, "r+") as file:
        del file["cursor_pos"]
        dataset = file.create_dataset("cursor_pos", shape, dtype="f8")
        dataset.attrs["MATLAB_class"] = np.bytes_("double")
    with pytest.raises(RecordingFormatError, match="cursor_pos.*matching t"):
        load_recording(path)


def test_rejects_nonfinite_positions(tmp_path: Path) -> None:
    path = make_recording(tmp_path / "synthetic.mat")
    with h5py.File(path, "r+") as file:
        file["cursor_pos"][0, 1] = np.inf
    with pytest.raises(RecordingFormatError, match="cursor_pos.*nonfinite"):
        load_recording(path)


@pytest.mark.parametrize(
    ("values", "message"),
    [([11, 10, 12], "descending"), ([10, np.nan, 12], "nonfinite")],
)
def test_rejects_invalid_spike_times(
    tmp_path: Path, values: list, message: str
) -> None:
    path = make_recording(tmp_path / "synthetic.mat")
    with h5py.File(path, "r+") as file:
        file[file["spikes"][0, 0]][0] = values
    with pytest.raises(RecordingFormatError, match=f"spikes.*{message}"):
        load_recording(path)


def test_rejects_null_reference_as_distinct_from_empty_train(tmp_path: Path) -> None:
    path = make_recording(tmp_path / "synthetic.mat")
    with h5py.File(path, "r+") as file:
        file["spikes"][0, 0] = h5py.Reference()
    with pytest.raises(RecordingFormatError, match="null reference"):
        load_recording(path)


def test_rejects_channel_spike_mismatch(tmp_path: Path) -> None:
    path = make_recording(tmp_path / "synthetic.mat")
    with h5py.File(path, "r+") as file:
        del file["spikes"]
        dataset = file.create_dataset("spikes", (3, 1), dtype=h5py.ref_dtype)
        dataset.attrs["MATLAB_class"] = np.bytes_("cell")
    with pytest.raises(RecordingFormatError, match="spikes.*matching chan_names"):
        load_recording(path)


def test_rejects_malformed_empty_encoding(tmp_path: Path) -> None:
    path = make_recording(tmp_path / "synthetic.mat")
    with h5py.File(path, "r+") as file:
        file[file["spikes"][1, 0]][:] = [1, 2]
    with pytest.raises(RecordingFormatError, match="empty encoding"):
        load_recording(path)


def test_rejects_missing_field(tmp_path: Path) -> None:
    path = make_recording(tmp_path / "synthetic.mat")
    with h5py.File(path, "r+") as file:
        del file["t"]
    with pytest.raises(RecordingFormatError, match="t: required dataset"):
        load_recording(path)


def test_rejects_duplicate_channel_identity(tmp_path: Path) -> None:
    path = make_recording(tmp_path / "synthetic.mat")
    with h5py.File(path, "r+") as file:
        file["chan_names"][0, 1] = file["chan_names"][0, 0]
    with pytest.raises(RecordingFormatError, match="duplicate channel"):
        load_recording(path)


def test_rejects_unsupported_mat_format(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.mat"
    path.write_bytes(b"MATLAB 5.0 MAT-file: visibly synthetic unsupported input")
    with pytest.raises(RecordingFormatError, match="earlier MAT formats unsupported"):
        load_recording(path)


def test_accepts_all_empty_trains_without_inventing_coverage(tmp_path: Path) -> None:
    path = make_recording(tmp_path / "synthetic.mat")
    with h5py.File(path, "r+") as file:
        empty = file["spikes"][1, 0]
        file["spikes"][:] = empty
    recording = load_recording(path)
    assert recording.diagnostics["total_spike_events"] == 0
    assert recording.diagnostics["empty_spike_trains"] == 6
    assert recording.diagnostics["spike_range_s"] is None


def test_rejects_numeric_array_in_place_of_cell_references(tmp_path: Path) -> None:
    path = make_recording(tmp_path / "synthetic.mat")
    with h5py.File(path, "r+") as file:
        del file["spikes"]
        dataset = file.create_dataset("spikes", (3, 2), dtype="f8")
        dataset.attrs["MATLAB_class"] = np.bytes_("cell")
    with pytest.raises(RecordingFormatError, match="object-reference cell array"):
        load_recording(path)
