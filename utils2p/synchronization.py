"""
Synchronization module
======================

This module provides functions to process the synchronization data
acquired with Thor Sync during imaging.
"""

import numpy as np
import h5py
import json

import utils2p.main as main

class SynchronizationError(Exception):
    """The input data is not consistent with synchronization assumption."""

    pass


def get_lines_from_h5_file(file_path, line_names):
    """
    This function returns the values of the requested lines save in
    an h5 generated by ThorSync.

    Parameters
    ----------
    file_path : string
        Path to h5 file.
    line_names : list of strings
        List of the ThorSync line names to be returned.

    Returns
    -------
    lines : tuple
        Line arrays in the same order as given in line_names.

    Examples
    --------
    >>> import utils2p
    >>> import utils2p.synchronization
    >>> h5_file = utils2p.find_sync_file("data/mouse_kidney_z_stack")
    >>> line_names = ["Frame Counter", "Capture On"]
    >>> frame_counter, capture_on = utils2p.synchronization.get_lines_from_h5_file(h5_file, line_names)
    >>> type(frame_counter)
    <class 'numpy.ndarray'>
    >>> frame_counter.shape
    (54000,)
    >>> type(capture_on)
    <class 'numpy.ndarray'>
    >>> capture_on.shape
    (54000,)
    """
    lines = []
    
    with h5py.File(file_path, "r") as f:
        for name in line_names:
            try:
                try:
                    lines.append(f["DI"][name][:].squeeze())
                except KeyError:
                    lines.append(f["CI"][name][:].squeeze())
            except KeyError:
                DI_keys = list(f["DI"].keys())
                CI_keys = list(f["CI"].keys())
                raise KeyError(f"No line named '{name}' exists. The digital lines are {DI_keys} and the continuous lines are {CI_keys}.")
    return tuple(lines)


def get_times(length, freq):
    """
    This function returns the time point of each tick
    for a given sequence length and tick frequency.

    Parameters
    ----------
    length : int
        Length of sequence.
    freq : float
        Frequency in Hz.

    Returns
    -------
    times : array
        Times in seconds.

    Examples
    --------
    >>> import utils2p.synchronization
    >>> utils2p.synchronization.get_times(5, 20)
    array([0.  , 0.05, 0.1 , 0.15, 0.2 ])
    """
    times = np.arange(0, length / freq, 1 / freq)
    return times


def edges(line, size=0):
    """
    Returns the indices of edges in a line. An
    edge is change in value of the line. A size
    argument can be specified to filter for changes
    of specific magnitude. By default only rising
    edges (increases in value) are returned.

    Parameters
    ----------
    line : numpy array
        Line signal from h5 file.
    size : float or tuple
        Size of the rising edge. If float it is used as minimum.
        Tuples specify a range. To get falling edges use negative values.
        Only one boundary can be applied using np.inf as on of the values.
        All boundaries are exclusive the specified value.
    Returns
    -------
    indices : list
        Indices of the rising edges.

    Examples
    --------
    >>> import utils2p.synchronization
    >>> import numpy as np
    >>> binary_line = np.array([0, 1, 1, 0, 1, 1])
    >>> utils2p.synchronization.edges(binary_line)
    (array([1, 4]),)
    >>> utils2p.synchronization.edges(binary_line, size=2)
    (array([], dtype=int64),)
    >>> utils2p.synchronization.edges(binary_line, size=(-np.inf, np.inf))
    (array([1, 3, 4]),)
    >>> continuous_line = np.array([0, 0, 3, 3, 3, 5, 5, 8, 8, 10, 10, 10])
    >>> utils2p.synchronization.edges(continuous_line)
    (array([2, 5, 7, 9]),)
    >>> utils2p.synchronization.edges(continuous_line, size=2)
    (array([2, 7]),)
    >>> utils2p.synchronization.edges(continuous_line, size=(-np.inf, 3))
    (array([5, 9]),)
    """
    diff = np.diff(line.astype(np.float64))
    if type(size) == tuple:
        zero_elements = np.isclose(diff, np.zeros_like(diff))
        edges_in_range = np.logical_and(diff > size[0], diff < size[1])
        valid_edges = np.logical_and(edges_in_range,  np.logical_not(zero_elements))
        indices = np.where(valid_edges)
    else:
        indices = np.where(diff > size)
    indices = tuple([i + 1 for i in indices])
    return indices


def get_start_times(line, times):
    """
    Get the start times of a digital signal,
    i.e. the times of the rising edges.

    Parameters
    ----------
    line : numpy array
        Line signal from h5 file.
    times : numpy array
        Times returned by :func:`utils2p.synchronization.get_times`

    Returns
    -------
    time_points : list
        List of the start times.

    Examples
    --------
    >>> import utils2p.synchronization
    >>> import numpy as np
    >>> binary_line = np.array([0, 1, 1, 0, 1, 1])
    >>> times = utils2p.synchronization.get_times(len(binary_line), freq=20)
    >>> times
    array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25])
    >>> utils2p.synchronization.get_start_times(binary_line, times)
    array([0.05, 0.2 ])
    """
    indices = edges(line, size=(0, np.inf))
    time_points = times[indices]
    return time_points


def _capture_metadata(n_frames, dropped_frames=None):
    """
    Returns a dictionary as it is usually saved by the seven
    camera setup in the "capture_metadata.json" file.
    It assumes that no frames where dropped.

    Parameters
    ----------
    n_frames : list of integers
        Number of frames for each camera.
    dropped_frames : list of list of integers
        Frames that were dropped for each camera.
        Default is None which means no frames where
        dropped.

    Returns
    -------
    capture_info : dict
        Default metadata dictionary for the seven camera
        system.
    """
    if dropped_frames is None:
        dropped_frames = [[] for i in range(len(n_frames))]
    capture_info = {"Frame Counts": {}}
    for cam_idx, n in enumerate(n_frames):
        frames_dict = {}
        current_frame = 0
        for i in range(n):
            while current_frame in dropped_frames[cam_idx]:
                current_frame += 1
            frames_dict[str(i)] = current_frame
            current_frame += 1
        capture_info["Frame Counts"][str(cam_idx)] = frames_dict
    return capture_info


def process_cam_line(line, capture_json):
    """
    Remove superfluous signals and use frame numbers in array.
    The cam line signal form the h5 file is a binary sequence.
    Rising edges mark the acquisition of a new frame.
    The setup keeps producing rising edges after the acquisition of the
    last frame. These rising edges are ignored.
    This function converts it to frame numbers using the information
    stored in the metadata file of the seven camera setup.
    In the metadata file the keys are the indices of the file names
    and the values are the grabbed frame numbers. Suppose the 3
    frame was dropped. Then the entries in the dictionary will
    be as follows:
    "2": 2
    "3": 4
    "4": 5

    Parameters
    ----------
    line : numpy array
        Line signal from h5 file.
    capture_json : string
        Path to the json file save by our camera software.
        This file is usually located in the same folder as the frames
        and is called 'capture_metadata.json'. If None, it is assumed
        that no frames were dropped.

    Returns
    -------
    processed_line : numpy array
        Array with frame number for each time point.
        If no frame is available for a given time the value is -1.

    Examples
    --------
    >>> import utils2p
    >>> import utils2p.synchronization
    >>> import numpy as np
    >>> h5_file = utils2p.find_sync_file("data/mouse_kidney_raw")
    >>> capture_json = utils2p.find_seven_camera_metadata_file("data/mouse_kidney_raw")
    >>> line_names = ["Basler"]
    >>> (cam_line,) = utils2p.synchronization.get_lines_from_h5_file(h5_file, line_names)
    >>> set(np.diff(cam_line))
    {0, 8, 4294967288}
    >>> processed_cam_line = utils2p.synchronization.process_cam_line(cam_line, capture_json)
    >>> set(np.diff(processed_cam_line))
    {0, 1, -60}
    >>> cam_line = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0])
    >>> utils2p.synchronization.process_cam_line(cam_line, capture_json=None)
    array([-1,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1])
    """
    # Check that sequence is binary
    if len(set(line)) > 2:
        raise ValueError("Invalid line argument. Sequence is not binary.")

    # Find indices of the start of each frame acquisition
    rising_edges = edges(line, (0, np.inf))[0]

    # Load capture metadata or generate default
    if capture_json is not None:
        with open(capture_json, "r") as f:
            capture_info = json.load(f)
    else:
        capture_info = _capture_metadata([len(rising_edges),])

    # Find the number of frames for each camera
    n_frames = []
    for cam_idx in capture_info["Frame Counts"].keys():
        max_in_json = max(capture_info["Frame Counts"][cam_idx].values())
        n_frames.append(max_in_json + 1)

    # Ensure all cameras acquired the same number of frames
    if len(np.unique(n_frames)) > 1:
        raise SynchronizationError("The frames across cameras are not synchronized.")

    # Last rising edge that corresponds to a frame
    last_tick = max(n_frames)

    # check that there is a rising edge for every frame
    if len(rising_edges) < last_tick:
        raise ValueError("The provided cam line and metadata are inconsistent. cam line has less frame acquisitions than metadata.")

    # Ensure correct handling if no rising edges are present after last frame
    if len(rising_edges) == int(last_tick):
        average_frame_length = int(np.mean(np.diff(rising_edges)))
        last_rising_edge = rising_edges[-1]
        additional_edge = last_rising_edge + average_frame_length
        if additional_edge > len(line):
            additional_edge = len(line)
        rising_edges = list(rising_edges)
        rising_edges.append(additional_edge)
        rising_edges = np.array(rising_edges)

    processed_line = np.ones_like(line) * -1

    current_frame = 0
    first_camera_used = sorted(list(capture_info["Frame Counts"].keys()))[0]
    for i, (start, stop) in enumerate(
        zip(rising_edges[: last_tick], rising_edges[1 : last_tick + 1])
    ):
        if capture_info["Frame Counts"][first_camera_used][str(current_frame + 1)] <= i:
            current_frame += 1
        processed_line[start:stop] = current_frame
    return processed_line.astype(np.int)


def process_frame_counter(line, metadata=None, steps_per_frame=None):
    """
    Converts the frame counter line to an array with frame numbers for each
    time point.

    Parameters
    ----------
    line : numpy array
        Line signal from h5 file.
    metadata : :class:`utils2p.Metadata`
        :class:`utils2p.Metadata` object that holding the 2p imaging
        metadata for the experiment. Optional. If metadata is not
        given steps_per_frame has to be set.
    steps_per_frame : int
        Number of steps the frame counter takes per frame.
        This includes fly back frame and averaging, i.e. if you
        acquire one frame and flyback frames is set to 3 this number
        should be 4.

    Returns
    -------
    processed_frame_counter : numpy array
        Array with frame number for each time point.
        If no frame was recorded at a time point the value is -1.

    Examples
    --------
    >>> import utils2p
    >>> import utils2p.synchronization
    >>> h5_file = utils2p.find_sync_file("data/mouse_kidney_z_stack")
    >>> line_names = ["Frame Counter",]
    >>> (frame_counter,) = utils2p.synchronization.get_lines_from_h5_file(h5_file, line_names)
    >>> set(frame_counter)
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}
    >>> metadata_file = utils2p.find_metadata_file("data/mouse_kidney_z_stack")
    >>> metadata = utils2p.Metadata(metadata_file)
    >>> processed_frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, metadata)
    >>> set(processed_frame_counter)
    {0, -1}
    >>> steps_per_frame = metadata.get_n_z() * metadata.get_n_averaging()
    >>> steps_per_frame
    30
    >>> processed_frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, steps_per_frame=steps_per_frame)
    >>> set(processed_frame_counter)
    {0, -1}
    
    By default the function treat volumes as frames.
    If you want to treat every slice of the volume as a separate frame,
    you can do so by `steps_per_frame`. The example has three steps in z.
    >>> steps_per_frame = metadata.get_n_averaging()
    >>> steps_per_frame
    10
    >>> processed_frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, steps_per_frame=steps_per_frame)
    >>> set(processed_frame_counter)
    {0, 1, 2, -1}
    """
    if metadata is not None and steps_per_frame is not None:
        warnings.warn("metadata argument will be ignored because steps_per_frame argument was set.")
    if metadata is not None and type(metadata) != main.Metadata:
        raise TypeError("metadata argument must be of type utils2p.Metadata or None.")
    if steps_per_frame is not None and type(steps_per_frame) != int:
        raise TypeError("steps_per_frame has to be of type int")

    if metadata is not None and steps_per_frame is None:
        if metadata.get_value("Streaming", "zFastEnable") == "0":
            steps_per_frame = 1
        else:
            steps_per_frame = metadata.get_n_z() 
            if metadata.get_value("Streaming", "enable") == "1":
                steps_per_frame += metadata.get_n_flyback_frames()
        if metadata.get_value("LSM", "averageMode") == "1" and metadata.get_area_mode() not in ["line", "kymograph"]:
            steps_per_frame = steps_per_frame * metadata.get_n_averaging()
    elif steps_per_frame is None:
        raise ValueError("If no metadata object is given, the steps_per_frame argument has to be set.")

    processed_frame_counter = np.ones_like(line) * -1
    rising_edges = edges(line, (0, np.inf))[0]
    for i, index in enumerate(
        range(0, len(rising_edges) - steps_per_frame, steps_per_frame)
    ):
        processed_frame_counter[
            rising_edges[index] : rising_edges[index + steps_per_frame]
        ] = i
    processed_frame_counter[rising_edges[-steps_per_frame] :] = (
        processed_frame_counter[rising_edges[-steps_per_frame] - 1] + 1
    )
    return processed_frame_counter.astype(np.int)


def process_stimulus_line(line):
    """
    This function converts the stimulus line to an array with
    0s and 1s for stimulus off and on respectively. The raw
    stimulus line can contain values larger than 1.

    Parameters
    ----------
    line : numpy array
        Line signal from h5 file.

    Returns
    -------
    processed_frame_counter : numpy array
        Array with binary stimulus state for each time point.

    Examples
    --------
    >>> import utils2p
    >>> import utils2p.synchronization
    >>> import numpy as np
    >>> h5_file = utils2p.find_sync_file("data/mouse_kidney_raw")
    >>> line_names = ["CO2_Stim"]
    >>> (stimulus_line,) = utils2p.synchronization.get_lines_from_h5_file(h5_file, line_names)
    >>> set(stimulus_line)
    {0, 4}
    >>> processed_stimulus_line = utils2p.synchronization.process_stimulus_line(stimulus_line)
    >>> set(processed_stimulus_line)
    {0, 1}
    """
    processed_stimulus_line = np.zeros_like(line)
    indices = np.where(line > 0)
    processed_stimulus_line[indices] = 1
    return processed_stimulus_line.astype(np.int)


def process_optical_flow_line(line):
    """
    This function converts the optical flow line
    into a step function. The value corresponds
    to the index of optical flow value at this
    time point. If the value is -1, no optical flow
    value was recorded for this time point.

    Note: Due to the time it take to transfer the data
    from the Arduino to the computer it is possible that
    the last optical flow data point is missing, i.e.
    the processed optical flow line indicates one more
    data point than the text file contains. This can be
    solved by cropping all lines before the acquisition
    of the last optical flow data point. Lines can be
    cropped with :func:`crop_lines`.

    Parameters
    ----------
    line : numpy array
        Line signal for h5 file.

    Returns
    -------
    processed_optical_flow_line : numpy array
        Array with monotonically increasing step
        function.

    Examples
    --------
    >>> import utils2p
    >>> import utils2p.synchronization
    >>> import numpy as np
    >>> h5_file = utils2p.find_sync_file("data/mouse_kidney_raw")
    >>> line_names = ["OpFlow"]
    >>> (optical_flow_line,) = utils2p.synchronization.get_lines_from_h5_file(h5_file, line_names)
    >>> set(optical_flow_line)
    {0, 16}
    >>> processed_optical_flow_line = utils2p.synchronization.process_optical_flow_line(optical_flow_line)
    >>> len(set(processed_optical_flow_line))
    1409
    """
    processed_optical_flow_line = np.ones_like(line) * -1
    rising_edges = edges(line, (0, np.inf))[0]
    for i in range(0, len(rising_edges) - 1):
        processed_optical_flow_line[
            rising_edges[i] : rising_edges[i + 1]
        ] = i
    processed_optical_flow_line[rising_edges[-1] :] = (
        processed_optical_flow_line[rising_edges[-1] - 1] + 1
    )
    return processed_optical_flow_line.astype(np.int)


def crop_lines(mask, lines):
    """
    This function crops all lines based on a binary signal/mask.
    The 'Capture On' line of the h5 file can be used as a mask.

    Parameters
    ----------
    mask : numpy array
        Mask that is used for cropping.
    lines : list of numpy arrays
        List of the lines that should be cropped.

    Returns
    -------
    cropped_lines : tuple of numpy arrays
        Tuple of cropped lines in same order as in input list.

    Examples
    --------
    >>> import utils2p
    >>> import utils2p.synchronization
    >>> import numpy as np
    >>> h5_file = utils2p.find_sync_file("data/mouse_kidney_raw")
    >>> line_names = ["Frame Counter", "Capture On", "CO2_Stim", "OpFlow"]
    >>> (frame_counter, capture_on, stimulus_line, optical_flow_line,) = utils2p.synchronization.get_lines_from_h5_file(h5_file, line_names)
    >>> frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, steps_per_frame=4)
    >>> len(frame_counter), len(capture_on), len(stimulus_line), len(optical_flow_line)
    (117000, 117000, 117000, 117000)
    >>> mask = np.logical_and(frame_counter >= 0, capture_on)
    >>> np.sum(mask)
    105869
    >>> (frame_counter, capture_on, stimulus_line, optical_flow_line,) = utils2p.synchronization.crop_lines(mask, (frame_counter, capture_on, stimulus_line, optical_flow_line,))
    >>> len(frame_counter), len(capture_on), len(stimulus_line), len(optical_flow_line)
    (105869, 105869, 105869, 105869)
    >>> line = np.arange(10)
    >>> mask = np.ones(10, dtype=np.bool)
    >>> mask[0] = False
    >>> mask[-1] = False
    >>> mask[4] = False
    >>> utils2p.synchronization.crop_lines(mask, (line,))
    (array([1, 2, 3, 4, 5, 6, 7, 8]),)
    """
    indices = np.where(mask)[0]
    first_idx = indices[0]
    last_idx = indices[-1]
    cropped_lines = []
    for line in lines:
        cropped_lines.append(line[first_idx : last_idx + 1])
    return tuple(cropped_lines)


def beh_idx_to_2p_idx(beh_indices, cam_line, frame_counter):
    """
    This functions converts behaviour frame numbers into the corresponding
    2p frame numbers.

    Parameters
    ----------
    beh_indices : numpy array
        Indices of the behaviour frames to be converted.
    cam_line : numpy array
        Processed cam line.
    frame_counter : numpy array
        Processed frame counter.

    Returns
    -------
    indices_2p : numpy array
        Corresponding 2p frame indices.

    Examples
    --------
    >>> import utils2p
    >>> import utils2p.synchronization
    >>> import numpy as np
    >>> h5_file = utils2p.find_sync_file("data/mouse_kidney_raw")
    >>> line_names = ["Frame Counter", "Basler"]
    >>> (frame_counter, cam_line,) = utils2p.synchronization.get_lines_from_h5_file(h5_file, line_names)
    >>> frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, steps_per_frame=4)
    >>> capture_json = utils2p.find_seven_camera_metadata_file("data/mouse_kidney_raw")
    >>> cam_line = utils2p.synchronization.process_cam_line(cam_line, capture_json)
    >>> utils2p.synchronization.beh_idx_to_2p_idx(np.array([0,]), cam_line, frame_counter)
    array([-1])
    >>> utils2p.synchronization.beh_idx_to_2p_idx(np.array([10,]), cam_line, frame_counter)
    array([0])
    >>> utils2p.synchronization.beh_idx_to_2p_idx(np.arange(30), cam_line, frame_counter)
    array([-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1])
    """
    thor_sync_indices = edges(cam_line)[0]

    indices_2p = np.ones(len(beh_indices), dtype=np.int) * -1

    for i, frame_num in enumerate(beh_indices):
        thor_sync_index = thor_sync_indices[frame_num]
        beh_frame_num = cam_line[thor_sync_index]
        indices_2p[i] = frame_counter[thor_sync_index]

    return indices_2p.astype(np.int)


def reduce_during_2p_frame(frame_counter, values, function):
    """
    Reduces all values occuring during the acquisition of a
    2-photon frame to a single value using the `function`
    given by the user.

    Parameters
    ----------
    frame_counter : numpy array
        Processed frame counter.
    values : numpy array
        Values upsampled to the frequency of ThorSync,
        i.e. 1D numpy array of the same length as
        `frame_counter`.
    function : function
        Function used to reduce the value,
        e.g. np.mean.

    Returns
    -------
    reduced : numpy array
        Numpy array with value for each 2p frame.

    Examples
    --------
    >>> import utils2p
    >>> import utils2p.synchronization
    >>> import numpy as np
    >>> h5_file = utils2p.find_sync_file("data/mouse_kidney_raw")
    >>> line_names = ["Frame Counter", "CO2_Stim"]
    >>> (frame_counter, stimulus_line,) = utils2p.synchronization.get_lines_from_h5_file(h5_file, line_names)
    >>> frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, steps_per_frame=1)
    >>> stimulus_line = utils2p.synchronization.process_stimulus_line(stimulus_line)
    >>> np.max(frame_counter)
    4
    >>> stimulus_during_2p_frames = utils2p.synchronization.reduce_during_2p_frame(frame_counter, stimulus_line, np.mean)
    >>> len(stimulus_during_2p_frames)
    5
    >>> np.max(stimulus_during_2p_frames)
    0.7136134613556422
    >>> stimulus_during_2p_frames = utils2p.synchronization.reduce_during_2p_frame(frame_counter, stimulus_line, np.max)
    >>> len(stimulus_during_2p_frames)
    5
    >>> set(stimulus_during_2p_frames)
    {0.0, 1.0}
    """
    if len(frame_counter) != len(values):
        raise ValueError("frame_counter and values need to have the same length.")
    
    reduced = np.ones(np.max(frame_counter) + 1) * np.nan
    thor_sync_indices = tuple(edges(frame_counter, (0, np.inf))[0])
    
    starts = thor_sync_indices
    stops = thor_sync_indices[1:] + (len(frame_counter),)
    
    if frame_counter[0] != -1:
        starts = (0,) + starts
        stops = (thor_sync_indices[0],) + stops

    for i, (start, stop) in enumerate(zip(starts, stops)):
        reduced[i] = function(values[start:stop])

    return reduced


class SyncMetadata(main._XMLFile):
    """
    Class for managing ThorSync metadata.
    Loads metadata file 'ThorRealTimeDataSettings.xml'
    and returns the root of an ElementTree.

    Parameters
    ----------
    path : string
        Path to xml file.

    Returns
    -------
    Instance of class Metadata
        Based on given xml file.

    Examples
    --------
    >>> import utils2p.synchronization
    >>> metadata = utils2p.synchronization.SyncMetadata("data/mouse_kidney_raw/2p/Sync-025/ThorRealTimeDataSettings.xml")
    >>> type(metadata)
    <class 'utils2p.synchronization.SyncMetadata'>
    """

    def get_active_devices(self):
        active_devices = []
        for device in self.get_value("DaqDevices", "AcquireBoard"):
            if device.attrib["active"] == "1":
                active_devices.append(device)
        return active_devices

    
    def get_freq(self):
        """
        Returns the frequency of the ThorSync
        value acquisition, i.e. the sample rate.

        Returns
        -------
        freq : integer
            Sample frequency in Hz.

        Examples
        --------
        >>> import utils2p.synchronization
        >>> metadata = utils2p.synchronization.SyncMetadata("data/mouse_kidney_raw/2p/Sync-025/ThorRealTimeDataSettings.xml")
        >>> metadata.get_freq()
        30000
        """
        sample_rate = -1
        for device in self.get_active_devices():
            set_for_device = False
            for element in device.findall("SampleRate"):
                if element.attrib["enable"] == "1":
                    if set_for_device:
                        raise ValueError("Invalid metadata file. Multiple sample rates are enabled for device {device.type}")
                    if sample_rate != -1:
                        raise ValueError("Multiple devices are enabled.")
                    sample_rate = int(element.attrib["rate"])
                    set_for_device = True
        return sample_rate 


def processed_lines(sync_file, sync_metadata_file, metadata_2p_file, seven_camera_metadata_file=None):
    """
    This function extracts all the standard lines and processes them.
    It works for both microscopes.

    Parameters
    ----------
    sync_file : str
        Path to the synchronization file.
    sync_metadata_file : str
        Path to the synchronization metadata file.
    metadata_2p_file : str
        Path to the ThorImage metadata file.
    seven_camera_metadata_file : str
        Path to the metadata file of the 7 camera system.

    Returns
    -------
    processed_lines : dictionary
        Dictionary with all processed lines.
    
    Examples
    --------
    >>> import utils2p
    >>> import utils2p.synchronization
    >>> experiment_dir = "data/mouse_kidney_raw/"
    >>> sync_file = utils2p.find_sync_file(experiment_dir)
    >>> metadata_file = utils2p.find_metadata_file(experiment_dir)
    >>> sync_metadata_file = utils2p.find_sync_metadata_file(experiment_dir)
    >>> seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(experiment_dir)
    >>> processed_lines = utils2p.synchronization.processed_lines(sync_file, sync_metadata_file, metadata_file, seven_camera_metadata_file)
    """
    processed_lines = {}
    processed_lines["Capture On"], processed_lines["Frame Counter"] = get_lines_from_h5_file(sync_file, ["Capture On", "Frame Counter"])

    try:
        # For microscope 1
        processed_lines["CO2"], processed_lines["Cameras"], processed_lines["Optical flow"] = get_lines_from_h5_file(sync_file, ["CO2_Stim", "Basler", "OpFlow",])
    except KeyError:
        # For microscope 2
        processed_lines["CO2"], processed_lines["Cameras"] = get_lines_from_h5_file(sync_file, ["CO2", "Cameras",])


    processed_lines["Cameras"] = process_cam_line(processed_lines["Cameras"], seven_camera_metadata_file)

    metadata_2p = main.Metadata(metadata_2p_file)
    processed_lines["Frame Counter"] = process_frame_counter(processed_lines["Frame Counter"], metadata_2p)

    processed_lines["CO2"] = process_stimulus_line(processed_lines["CO2"])
        
    if "Optical flow" in processed_lines.keys():
        processed_lines["Optical flow"] = process_optical_flow_line(processed_lines["Optical flow"])

    mask = np.logical_and(processed_lines["Capture On"], processed_lines["Frame Counter"] >= 0)

    # Make sure the clipping start just before the acquisition of the first frame
    indices = np.where(mask)[0]
    mask[max(0, indices[0] - 1)] = True

    for line_name, line in processed_lines.items():
        processed_lines[line_name] = crop_lines(mask, [processed_lines[line_name],])[0]
    
    # Get times of ThorSync ticks
    metadata = SyncMetadata(sync_metadata_file)
    freq = metadata.get_freq()
    times = get_times(len(processed_lines["Frame Counter"]), freq)
    processed_lines["Times"] = times

    return processed_lines
