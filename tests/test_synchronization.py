"""
This module provides unit tests for the functions provided in utils2p.synchronization.
"""

import numpy as np

import utils2p.synchronization

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
