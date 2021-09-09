"""
This module provides unit tests for the functions provided in utils2p.synchronization.
"""

import os.path
import json

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


@pytest.fixture
def seven_camera_metadata(tmpdir):
    """
    This pytest factory produces a json metadata file
    for the seven camera setup.
    """
    path = os.path.join(tmpdir, "capture_metadata.json")
    def _seven_camera_metadata(n_frames, dropped_frames=None):
        """
        Generates a json file with seven camera capture metadata.

        Parameters
        ----------
        n_frames : list of integers
            Number of frames for each camera.
        dropped_frames : list of list of integers
            Frames that were dropped for each camera.
            Default is None which means no frames where
            dropped.
        """
        capture_info = utils2p.synchronization._capture_metadata(n_frames, dropped_frames)
        with open(path, "w") as f:
            json.dump(capture_info, f)
        return path
    return _seven_camera_metadata


@pytest.mark.parametrize("length", [20, 1000000])
def test_get_time(length):
    times = utils2p.synchronization.get_times(length, 1 / 30000)
    assert len(times) == length
    assert np.isclose(times[0], 0)


def test_get_lines_from_sync_file(h5_file):
    expected_frame_counter = np.ones((10, 1), dtype=np.dtype("<u4"))
    expected_cam_line = np.ones((1000, 1), dtype=np.dtype("<u4")) * 2
    expected_stim_line = np.ones((1000, 1), dtype=np.dtype("<u4")) * 3
    expected_capture_on = np.ones((1000, 1), dtype=np.dtype("<u4")) * 4
    expected_piezo_monitor = np.sin(np.linspace(0, 10 * np.pi, 1000, dtype=np.float64))[:, np.newaxis]
    content = {
            "DI": {
                "Frame Counter": expected_frame_counter,
            }, 
            "CI": {
                "Cameras": expected_cam_line,
                "CO2": expected_stim_line,
                "Capture On": expected_capture_on,
            },
            "AI": {
                "Piezo Monitor": expected_piezo_monitor,
            },
        }
    path = h5_file(content)
    cam_line, frame_counter, stimulus_line, capture_on, piezo_monitor = utils2p.synchronization.get_lines_from_sync_file(path, ["Cameras", "Frame Counter", "CO2", "Capture On", "Piezo Monitor"])
    assert cam_line.ndim == 1
    assert np.all(expected_cam_line == cam_line)
    assert frame_counter.ndim == 1
    assert np.all(expected_frame_counter == frame_counter)
    assert stimulus_line.ndim == 1
    assert np.all(expected_stim_line == stimulus_line)
    assert capture_on.ndim == 1
    assert np.all(expected_capture_on == capture_on)
    assert piezo_monitor.ndim == 1
    assert np.allclose(expected_piezo_monitor.squeeze(), piezo_monitor)

    with pytest.raises(KeyError):
        utils2p.synchronization.get_lines_from_sync_file(path, ["Some none existing line",])


def test_edges():
    line = np.array([0, 1, 1, 2, 2, 2, 0, 0, -1, -1, -1, 5, 5])

    expected = np.array([1, 3, 11])
    assert np.all(expected == utils2p.synchronization.edges(line, (0, np.inf))[0])
    
    expected = np.array([6, 8])
    assert np.all(expected == utils2p.synchronization.edges(line, (np.NINF, 0)))
    
    expected = np.array([11,])
    assert np.all(expected == utils2p.synchronization.edges(line, 2))


def test_correct_split_edges():
    # rising
    line = np.array([0, 0, 0, 1, 2, 2, 4, 4, 4, 5, 6, 6])
    expected = np.array([0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6])
    assert np.all(expected == utils2p.synchronization.correct_split_edges(line))
    
    # falling
    line = line[::-1]
    expected = expected[::-1]
    assert np.all(expected == utils2p.synchronization.correct_split_edges(line))

    # Wider spread
    line = np.array([0, 0, 0, 0, 1, 2, 3, 3, 6, 6, 6, 8, 9, 9, 9])
    expected = np.array([0, 0, 0, 0, 3, 3, 3, 3, 6, 6, 6, 9, 9, 9, 9])
    assert np.all(expected == utils2p.synchronization.correct_split_edges(line))

    assert np.all(expected[::-1] == utils2p.synchronization.correct_split_edges(line[::-1]))

    # Falling and rising
    line = np.array([0, 0, 0, 1, 2, 3, 3, 3, 2, 1, 0, 0, 0])
    expected = np.array([0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0])
    assert np.all(expected == utils2p.synchronization.correct_split_edges(line))


def test_get_start_times():
    line = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0, 3])
    times = np.arange(len(line))
    assert np.allclose(np.array([2, 5, 9]), utils2p.synchronization.get_start_times(line, times))
    
    # Test zero_based_counter
    line = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    assert np.allclose(np.array([2, 5, 7]), utils2p.synchronization.get_start_times(line, times))
    assert np.allclose(np.array([0, 2, 5, 7]), utils2p.synchronization.get_start_times(line, times, zero_based_counter=True))


def test__capture_metadata():
    capture_info = utils2p.synchronization._capture_metadata([2, 3])
    assert capture_info == {"Frame Counts": {"0": {"0": 0, "1": 1}, "1": {"0": 0, "1": 1, "2": 2}}}
    
    capture_info = utils2p.synchronization._capture_metadata([3,], [[1,]])
    assert capture_info == {"Frame Counts": {"0": {"0": 0, "1": 2, "2": 3},}}
    
    capture_info = utils2p.synchronization._capture_metadata([6,], [[1, 4]])
    assert capture_info == {"Frame Counts": {"0": {"0": 0, "1": 2, "2": 3, "3": 5, "4": 6, "5": 7},}}


def test_process_cam_line(seven_camera_metadata):
    line = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1])
    expected = np.array([-9223372036854775808, -9223372036854775808, -9223372036854775808, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2])
    result = utils2p.synchronization.process_cam_line(line, None)
    assert np.allclose(result, expected)

    metadata = seven_camera_metadata([2,])
    expected = np.array([-9223372036854775808, -9223372036854775808, -9223372036854775808, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -9223372036854775808])
    result = utils2p.synchronization.process_cam_line(line, metadata)
    assert np.allclose(result, expected)

    line = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    expected = np.array([-9223372036854775808, -9223372036854775808, -9223372036854775808, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, -9223372036854775808])
    metadata = seven_camera_metadata([4,])
    result = utils2p.synchronization.process_cam_line(line, metadata)
    assert np.allclose(result, expected)
    
    line = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    expected = np.array([-9223372036854775808, -9223372036854775808, -9223372036854775808, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, -9223372036854775808])
    metadata = seven_camera_metadata([3,], dropped_frames=[[2,],])
    result = utils2p.synchronization.process_cam_line(line, metadata)
    assert np.allclose(result, expected)
    
    line = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, ])
    expected = np.array([-9223372036854775808, -9223372036854775808, -9223372036854775808, 0, 0, 0, 0, 0, 0, 1,])
    result = utils2p.synchronization.process_cam_line(line, None)
    assert np.allclose(result, expected)

    # Non-binary line argument
    line = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        utils2p.synchronization.process_cam_line(line, None)

    # Inconsistent number of frames across cameras
    metadata = seven_camera_metadata([4, 5])
    line = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    with pytest.raises(utils2p.synchronization.SynchronizationError):
        utils2p.synchronization.process_cam_line(line, metadata)

    # More frames in metadata than ticks in line
    metadata = seven_camera_metadata([5,])
    line = np.array([0, 1, 0, 1, 0])
    with pytest.raises(ValueError):
        utils2p.synchronization.process_cam_line(line, metadata)


def test_process_frame_counter(metadata_obj):
    line = np.array([0, 0, 0, 3, 3, 3, 4, 4, 8, 8, 8, 10, 10, 10, 11, 11, 15, 15, 15, 15,])
    
    expected = np.array([-9223372036854775808, -9223372036854775808, -9223372036854775808, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5,])
    assert np.allclose(expected, utils2p.synchronization.process_frame_counter(line, steps_per_frame=1))
    
    metadata = metadata_obj(flyback_frames=0, average_num=1, n_z=1) 
    assert np.allclose(expected, utils2p.synchronization.process_frame_counter(line, metadata))
    
    expected = np.array([-9223372036854775808, -9223372036854775808, -9223372036854775808, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,])
    assert np.allclose(expected, utils2p.synchronization.process_frame_counter(line, steps_per_frame=2))

    metadata = metadata_obj(flyback_frames=1, average_num=1, n_z=1) 
    assert np.allclose(expected, utils2p.synchronization.process_frame_counter(line, metadata))


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
    expected = np.array([-9223372036854775808, -9223372036854775808, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    processed = utils2p.synchronization.process_optical_flow_line(raw_line)
    assert np.allclose(expected, processed)


def test_beh_idx_to_2p_idx():
    beh_indices = np.array([2, 4])
    cam_line = np.array([-9223372036854775808, -9223372036854775808, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4])
    frame_counter = np.array([-9223372036854775808, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
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

    frame_counter = np.array([-9223372036854775808, 0, 1, 1, 2, 2, 3, 3, 3, 3])
    output_mean = utils2p.synchronization.reduce_during_2p_frame(frame_counter, values, np.mean)
    expected_result_mean = np.array([1, 2.5, 4.5, 7.5])
    assert np.allclose(output_mean, expected_result_mean)

    with pytest.raises(ValueError):
        utils2p.synchronization.reduce_during_2p_frame(np.zeros(3), np.zeros(4), np.mean)


def test_reduce_during_frame():
    frame_counter = np.array([0, 1, 1, 1, 2, 2, 3, 3, 3, 3])
    values = np.arange(len(frame_counter))
    
    output_mean = utils2p.synchronization.reduce_during_frame(frame_counter, values, np.mean)
    expected_result_mean = np.array([0, 2, 4.5, 7.5])
    assert np.allclose(output_mean, expected_result_mean)

    output_max = utils2p.synchronization.reduce_during_frame(frame_counter, values, np.max)
    expected_result_max = np.array([0, 3, 5, 9])
    assert np.allclose(output_max, expected_result_max)

    frame_counter = np.array([-9223372036854775808, 0, 1, 1, 2, 2, 3, 3, 3, 3])
    output_mean = utils2p.synchronization.reduce_during_frame(frame_counter, values, np.mean)
    expected_result_mean = np.array([1, 2.5, 4.5, 7.5])
    assert np.allclose(output_mean, expected_result_mean)

    # Missing frames
    frame_counter = np.array([-9223372036854775808, 0, 1, 1, 3, 3, 7, 7, 7, 7])
    output_mean = utils2p.synchronization.reduce_during_frame(frame_counter, values, np.mean)
    expected_result_mean = np.array([1, 2.5, 4.5, 7.5])
    assert np.allclose(output_mean, expected_result_mean)

    with pytest.raises(ValueError):
        utils2p.synchronization.reduce_during_frame(np.zeros(3), np.zeros(4), np.mean)


    # Reduce array of strings
    frame_counter = np.array([0, 1, 1, 1, 2, 2, 3, 3, 3, 3])
    values = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
    concat = lambda x: " ".join(x)
    output = utils2p.synchronization.reduce_during_frame(frame_counter, values, concat)
    expected_result = np.array(["a", "b c d", "e f", "g h i j"])
    assert np.all(output == expected_result)


def test_epoch_length_filter():
    binary_sequence = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    
    results = utils2p.synchronization.epoch_length_filter(binary_sequence, 3)
    expected_result = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    assert np.all(results == expected_result)
    
    results = utils2p.synchronization.epoch_length_filter(binary_sequence, 4)
    expected_result = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    assert np.all(results == expected_result)


def test_process_odor_line():
    freq = 30000
    line = np.zeros(freq * 10)
    dutycycle = 64
    for i in range(dutycycle):
        line[60000 + i : 90000 + i : 255] = 5
    dutycycle = 128
    for i in range(dutycycle):
        line[150000 + i : 210000 + i : 255] = 5
    result = utils2p.synchronization.process_odor_line(line, freq=freq, arduino_commands=("None", "One", "Two"), step_size=1.25)
    expected_result = np.array(["None",] * 10 * freq)
    expected_result[59902 : 89997] = "One"
    expected_result[150307 : 209746] = "Two"
    assert np.all(result == expected_result)


def test_event_based_frame_indices():
    event_indicator = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    expected_event_based_indices = np.array([-2, -1, 0, 1, 2, -4, -3 , -2 , -1, 0, 1, -1, 0, -3, -2, -1, 0, 1, 2], dtype=int)
    expected_event_number = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4], dtype=int)

    event_based_indices, event_number = utils2p.synchronization.event_based_frame_indices(event_indicator)
    assert np.all(event_based_indices == expected_event_based_indices)
    assert np.all(event_number == expected_event_number)
    
    # Test stop after last event
    event_indicator = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0], dtype=bool)
    expected_event_based_indices = np.array([-2, -1, 0, 1, 2, -4, -3 , -2 , -1, 0, 1, -1, 0, -3, -2, -1, 0, 1, 2, -2, -1], dtype=int)
    expected_event_number = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, -1, -1], dtype=int)

    event_based_indices, event_number = utils2p.synchronization.event_based_frame_indices(event_indicator)
    assert np.all(event_based_indices == expected_event_based_indices)
    assert np.all(event_number == expected_event_number)
    
    # Test start with event
    event_indicator = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0], dtype=bool)
    expected_event_based_indices = np.array([0, 1, 2, 3, 4, -4, -3 , -2 , -1, 0, 1, -1, 0, -3, -2, -1, 0, 1, 2, -2, -1], dtype=int)
    expected_event_number = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, -1, -1], dtype=int)

    event_based_indices, event_number = utils2p.synchronization.event_based_frame_indices(event_indicator)
    assert np.all(event_based_indices == expected_event_based_indices)
    assert np.all(event_number == expected_event_number)
