"""
This module provides unit tests for the functions provided in utils2p.synchronization.
"""

import os.path

import numpy as np
import pytest
import h5py

import utils2p.synchronization

@pytest.fixture
def h5_file(tmpdir):
    """
    This pytest factory constructs a fake Thor Sync h5 file, saves it
    to file and returns the path.
    """
    default_dict = {
            "AI": {"Piezo Monitor": np.zeros((1000, 1), dtype=np.dtype("<f8"))},
            "DI": {
                "Frame Counter": np.zeros((1000, 1), dtype=np.dtype("<u4"))
            }, 
            "CI": {
                "Cameras": np.zeros((1000, 1), dtype=np.dtype("<u4")),
                "CO2": np.zeros((1000, 1), dtype=np.dtype("<u4")),
                "Capture On": np.zeros((1000, 1), dtype=np.dtype("<u4")),
                "OpFlow": np.zeros((1000, 1), dtype=np.dtype("<u4")),
                "Frame out": np.zeros((1000, 1), dtype=np.dtype("<u4"))
            },
            "Global": {
                "GCtr": np.zeros((1000, 1), dtype=np.dtype("<u8"))
            }
        }
    
    def _dict_to_h5(h5_file, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                subgroup = h5_file.create_group(key)
                _dict_to_h5(subgroup, value)
            elif isinstance(value, np.ndarray):
                h5_file.create_dataset(key, data=value, dtype=value.dtype)
            else:
                raise ValueError("Can only encode numpy arrays.")

    def _h5_file(structure=default_dict):
        path = os.path.join(tmpdir, "Episode001.h5")
        with h5py.File(path, "w") as f:
            _dict_to_h5(f, structure)
        return path

    return _h5_file


@pytest.mark.parametrize("length", [20, 1000000])
def test_get_time(length):
    times = utils2p.synchronization.get_times(length, 1 / 30000)
    assert len(times) == length
    assert np.isclose(times[0], 0)


def test_get_lines_from_h5_file(h5_file):
    expected_frame_counter = np.ones((10, 1), dtype=np.dtype("<u4"))
    expected_cam_line = np.ones((1000, 1), dtype=np.dtype("<u4")) * 2
    expected_stim_line = np.ones((1000, 1), dtype=np.dtype("<u4")) * 3
    expected_capture_on = np.ones((1000, 1), dtype=np.dtype("<u4")) * 4
    content = {
            "DI": {
                "Frame Counter": expected_frame_counter,
            }, 
            "CI": {
                "Cameras": expected_cam_line,
                "CO2": expected_stim_line,
                "Capture On": expected_capture_on,
            },
        }
    path = h5_file(content)
    cam_line, frame_counter, stimulus_line, capture_on = utils2p.synchronization.get_lines_from_h5_file(path, ["Cameras", "Frame Counter", "CO2", "Capture On"])
    assert cam_line.ndim == 1
    assert np.all(expected_cam_line == cam_line)
    assert frame_counter.ndim == 1
    assert np.all(expected_frame_counter == frame_counter)
    assert stimulus_line.ndim == 1
    assert np.all(expected_stim_line == stimulus_line)
    assert capture_on.ndim == 1
    assert np.all(expected_capture_on == capture_on)


def test_edges():
    line = np.array([0, 1, 1, 2, 2, 2, 0, -1, -1, -1, 5, 5])

    expected = np.array([1, 3, 10])
    assert np.all(expected == utils2p.synchronization.edges(line, (0, np.inf))[0])
    
    expected = np.array([6, 7])
    assert np.all(expected == utils2p.synchronization.edges(line, (np.NINF, 0)))
    
    expected = np.array([10,])
    assert np.all(expected == utils2p.synchronization.edges(line, 2))


def test_get_start_times():
    line = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0, 3])
    times = np.arange(len(line))
    assert np.allclose(np.array([2, 5, 9]), utils2p.synchronization.get_start_times(line, times))


def test_process_cam_line():
    #line = np.array([
    raise NotImplementedError


def test_process_frame_counter():
    line = np.array([0, 0, 0, 3, 3, 3, 4, 4, 8, 8, 8, 10, 10, 10, 11, 11, 15, 15, 15, 15,])
    
    expected = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5,])
    assert np.allclose(expected, utils2p.synchronization.process_frame_counter(line))
    
    expected = np.array([-1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,])
    assert np.allclose(expected, utils2p.synchronization.process_frame_counter(line, steps_per_frame=2))


def test_process_stimulus_line():
    line = np.array([0, 0, -1, 1, 4, 0, 0, -1, 0, 4])
    expected = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1])
    assert np.allclose(expected, utils2p.synchronization.process_stimulus_line(line))


def test_crop_lines():
    mask = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=np.bool)
    line1 = np.arange(len(mask))
    line2 = line1[::-1].copy()
    cropped_line1 = np.arange(3, 7)
    cropped_line2 = np.arange(5, 1, -1)
    cropped_lines = utils2p.synchronization.crop_lines(mask, [line1, line2])
    assert np.allclose(cropped_line1, cropped_lines[0])
    assert np.allclose(cropped_line2, cropped_lines[1])
    
    mask[4] = False
    cropped_lines = utils2p.synchronization.crop_lines(mask, [line1, line2])
    assert np.allclose(cropped_line1, cropped_lines[0])
    assert np.allclose(cropped_line2, cropped_lines[1])


def test_process_optical_flow_line():
    raw_line = np.array([0, 0, 16, 16, 16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 0])
    expected = np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    processed = utils2p.synchronization.process_optical_flow_line(raw_line)
    assert np.allclose(expected, processed)


def test_beh_idx_to_2p_idx():
    beh_indices = np.array([2, 4])
    cam_line = np.array([-1, -1, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4])
    frame_counter = np.array([-1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    expected = np.array([1, 2])
    np.allclose(expected, utils2p.synchronization.beh_idx_to_2p_idx(beh_indices, cam_line, frame_counter))


def test_reduce_during_2p_frame():
    frame_counter = np.array([0, 1, 1, 1, 2, 2, 3, 3, 3, 3])
    values = np.arange(len(frame_counter))
    
    output_mean = utils2p.synchronization.reduce_during_2p_frame(frame_counter, values, np.mean)
    expected_result_mean = np.array([0, 2, 4.5, 7.5])
    assert np.allclose(output_mean, expected_result_mean)

    output_max = utils2p.synchronization.reduce_during_2p_frame(frame_counter, values, np.max)
    expected_result_max = np.array([0, 3, 5, 9])
    assert np.allclose(output_max, expected_result_max)

    frame_counter = np.array([-1, 0, 1, 1, 2, 2, 3, 3, 3, 3])
    output_mean = utils2p.synchronization.reduce_during_2p_frame(frame_counter, values, np.mean)
    expected_result_mean = np.array([1, 2.5, 4.5, 7.5])
    assert np.allclose(output_mean, expected_result_mean)

    with pytest.raises(ValueError):
        utils2p.synchronization.reduce_during_2p_frame(np.zeros(3), np.zeros(4), np.mean)
