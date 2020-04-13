"""
Core module
===========

The functions in this module are essentials that can be directly imported using `import utils2p`.
"""
import math
import os
import glob
import xml.etree.ElementTree as ET
import array
from pathlib import Path

import numpy as np

from . import synchronization
from .external import tifffile

package_directory = os.path.dirname(os.path.abspath(__file__))


class InvalidValueInMetaData(Exception):
    """This error should be raised when an invalid value
    is encountered in an 'Experiement.xml' file."""

    pass


class Metadata:
    """
    Class for managing ThorImage metadata.
    """

    def __init__(self, path):
        """
        Loads metadata file 'Experiment.xml' and returns the root of an ElementTree.

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
        >>> import utils2p
        >>> metadata = utils2p.Metadata("data/mouse_kidney_z_stack/Experiment.xml")
        >>> type(metadata)
        <class 'utils2p.main.Metadata'>
        """
        self.path = path
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()

    def get_metadata_value(self, *args):
        """
        This function returns a value from the metadata file 'Experiment.xml'.
    
        Parameters
        ----------
        args : strings
            Arbitrary number of strings of tag from the xml file in the correct order.
            See examples.
    
        Returns
        -------
        attribute or node : string or ElementTree node
            If the number of strings given in args leads to a leaf of the tree, the attribute,
            usually a dictionary is returned. Otherwise the node is returned.
    
        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_metadata_value('Timelapse','timepoints')
        '3'
        >>> metadata.get_metadata_value('LSM','pixelX')
        '128'
        >>> metadata.get_metadata_value('LSM','pixelY')
        '128'
        """
        node = self.root.find(args[0])
        for key in args[1:-1]:
            node = node.find(key)
        if len(list(node)) == 0:
            return node.attrib[args[-1]]
        else:
            return node

    def get_n_time_points(self):
        """
        Returns the number of time points for a given experiment metadata.
     
        Returns
        -------
        n_time_points : int
            Number of time points.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_n_time_points()
        3
        """
        return int(self.get_metadata_value("Timelapse", "timepoints"))

    def get_num_x_pixels(self):
        """
        Returns the image width for a given experiment metadata,
        i.e. the number of pixels in the x direction.
    
        Returns
        -------
        width : int
            Width of image.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_num_x_pixels()
        128
        """
        return int(self.get_metadata_value("LSM", "pixelX"))

    def get_num_y_pixels(self):
        """
        Returns the image height for a given experiment metadata,
        i.e. the number of pixels in the y direction.
    
        Returns
        -------
        height : int
            Width of image.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_num_y_pixels()
        128
        """
        return int(self.get_metadata_value("LSM", "pixelY"))

    def get_area_mode(self):
        """
        Returns the area mode of a given experiment metadata, e.g.
        square, rectangle, line, kymograph.
    
        Returns
        -------
        area_mode : string
            Area mode of experiment.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_area_mode()
        'square'
        """
        int_area_mode = int(self.get_metadata_value("LSM", "areaMode"))
        if int_area_mode == 0:
            return "square"
        elif int_area_mode == 1:
            return "rectangle"
        elif int_area_mode == 2:
            return "kymograph"
        elif int_area_mode == 3:
            return "line"
        else:
            raise InvalidValueInMetaData(
                f"{int_area_mode} is not a valid value for areaMode."
            )

    def get_n_z(self):
        """
        Returns the number for z slices for a given experiment metadata.
    
        Returns
        -------
        n_z : int
            Number of z layers of image.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_n_z()
        3
        """
        return int(self.get_metadata_value("ZStage", "steps"))

    def get_n_channels(self):
        """
        Returns the number of channels for a given experiment metadata.
    
        Returns
        -------
        n_channels : int
            Number of channels in raw data file.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_n_channels()
        2
        """
        return len(self.get_metadata_value("Wavelengths")) - 1

    def get_channels(self):
        """
        Retruns a tuple with the names of all channels.
    
        Returns
        -------
        channels : tuple of strings
            Names of channels.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_channels()
        ('ChanA', 'ChanB')
        """
        wavelengths_node = self.get_metadata_value("Wavelengths")
        channels = []
        for node in wavelengths_node.findall("Wavelength"):
            channels += [node.attrib["name"]]
        return tuple(channels)

    def get_pixel_size(self):
        """
        Returns the pixel size for a given experiment metadata.
    
        Returns
        -------
        pixel_size : float
            Size of one pixel in um in x and y direction.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_pixel_size()
        0.593
        """
        return float(self.get_metadata_value("LSM", "pixelSizeUM"))

    def get_z_step_size(self):
        """
        Returns the z step size for a given experiment metadata.
    
        Returns
        -------
        z_step_size : float
            Distance covered in um along z direction.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_z_step_size()
        15.0
        """
        return float(self.get_metadata_value("ZStage", "stepSizeUM"))

    def get_z_pixel_size(self):
        """
        Returns the pixel size in z direction for a given experiment metadata.
        This function is meant for "kymograph" and "line" recordings.
        For these recordings the pixel size in z direction is not 
        equal to the step size, unless the number of pixels equals the number
        of steps.
        For all other types of recordings it is equivalent to :func:`get_z_step_size`.
    
        Returns
        -------
        z_pixel_size : float
            Distance covered in um along z direction.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_z_pixel_size()
        15.0
        """
        area_mode = self.get_area_mode()
        if area_mode == "line" or area_mode == "kymograph":
            return (
                float(self.get_metadata_value("ZStage", "stepSizeUM"))
                * self.get_n_z()
                / self.get_num_y_pixels()
            )
        return float(self.get_metadata_value("ZStage", "stepSizeUM"))

    def get_dwell_time(self):
        """
        Returns the dwell time for a given experiment metadata.
    
        Returns
        -------
        dwell_time : float
            Dwell time for a pixel.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_dwell_time()
        0.308199306062498
        """
        return float(self.get_metadata_value("LSM", "dwellTime"))

    def get_n_flyback_frames(self):
        """
        Returns the number of flyback frames.

        Returns
        -------
        n_flyback : int
            Number of flyback frames.
        """
        n_flyback = int(self.get_metadata_value("Streaming", "flybackFrames"))
        return n_flyback

    def get_frame_rate(self):
        """
        Returns the frame rate for a given experiment metadata.
        When the frame rate is calculated flyback frames and
        steps in z are not considered frames.
    
        Returns
        -------
        frame_rate : float
            Frame rate of the experiment.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_frame_rate()
        10.0145
        """
        frame_rate_without_flybacks = float(self.get_metadata_value("LSM", "frameRate"))
        flyback_frames = self.get_n_flyback_frames()
        number_of_slices = self.get_n_z()
        return frame_rate_without_flybacks / (flyback_frames + number_of_slices)

    def get_width(self):
        """
        Returbns the image with in um for a given experiment metadata.
    
        Returns
        -------
        width : float
            Width of FOV in um..

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_width()
        75.88
        """
        return float(self.get_metadata_value("LSM", "widthUM"))

    def get_power_reg1_start(self):
        """
        Returns the starting position of power regulator 1 for a given
        experiment metadata. Unless a gradient is defined, this
        value is the power value for the entire experiment.
    
        Returns
        -------
        reg1_start : float
            Starting position of power regulator 1.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_power_reg1_start()
        1.0
        """
        return float(self.get_metadata_value("PowerRegulator", "start"))

    def get_gainA(self):
        """
        Returns the gain of channel A for a given experiment metadata.
    
        Returns
        -------
        gainA : int
            Gain of channel A.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_gainA()
        20.0
        """
        return float(self.get_metadata_value("PMT", "gainA"))

    def get_gainB(self):
        """
        Returns the gain of channel B for a given experiment metadata.
    
        Returns
        -------
        gainB : int
            Gain of channel B.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_gainB()
        30.0
        """
        return float(self.get_metadata_value("PMT", "gainB"))

    def get_date_time(self):
        """
        Returns the date and time of an experiment for a given experiment metadata.
    
        Returns
        -------
        date_time : string
            Date and time of an experiment.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_date_time()
        '11/21/2019 11:15:18'
        """
        return self.get_metadata_value("Date", "date")


def load_img(path):
    """
    This functions loads an image from file and returns as a numpy array.
    
    Parameters
    ----------
    path : string
        Path to image file.

    Returns
    -------
    numpy.array
        Image in form of numpy array.

    Examples
    --------
    >>> import utils2p
    >>> img = utils2p.load_img("data/chessboard_GRAY_U16.tif")
    >>> type(img)
    <class 'numpy.ndarray'>
    >>> img.shape
    (200, 200)
    """
    path = os.path.expanduser(os.path.expandvars(path))
    return tifffile.imread(path)


def load_raw(path, metadata):
    """
    This function loads a raw image generated by ThorImage as a numpy array.

    Parameters
    ----------
    path : string
        Path to raw file.
    metadata : ElementTree root
        Can be obtained with :func:`get_metadata`.

    Returns
    -------
    stacks : tuple of numpy arrays
        Number of numpy arrays depends on the number of channels recoded during the experiment.
        Has the following dimensions: TZYX or TYX for planar images.

    Examples
    --------
    >>> import utils2p
    >>> metadata = utils2p.Metadata('data/mouse_kidney_raw/Experiment.xml')
    >>> stack1, stack2 = utils2p.load_raw('data/mouse_kidney_raw/Image_0001_0001.raw',metadata)
    >>> type(stack1), type(stack2)
    (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>)
    >>> utils2p.save_img('stack1.tif',stack1)
    >>> utils2p.save_img('stack2.tif',stack2)
    """
    path = os.path.expanduser(os.path.expandvars(path))
    n_time_points = metadata.get_n_time_points()
    width = metadata.get_num_x_pixels()
    height = metadata.get_num_y_pixels()
    n_channels = metadata.get_n_channels()
    byte_size = os.stat(path).st_size

    assert not byte_size % 1, "File does not have an integer byte length."
    byte_size = int(byte_size)

    n_z = (
        byte_size / 2 / width / height / n_time_points / n_channels
    )  # divide by two because the values are of type short (16bit = 2byte)

    assert (
        not n_z % 1
    ), "Size given in metadata does not match the size of the raw file."
    n_z = int(n_z)

    # number of z slices from meta data can be different because of flyback frames
    meta_n_z = metadata.get_n_z()

    if n_z == 1:
        stacks = np.zeros((n_channels, n_time_points, height, width), dtype="uint16")
        image_size = width * height
        t_size = (
            width * height * n_channels
        )  # number of values stored for a given time point (this includes images for all channels)
        with open(path, "rb") as f:
            for t in range(n_time_points):
                # print('{}/{}'.format(t,n_time_points))
                a = array.array("H")
                a.fromfile(f, t_size)
                for c in range(n_channels):
                    stacks[c, t, :, :] = np.array(
                        a[c * image_size : (c + 1) * image_size]
                    ).reshape((height, width))
    elif n_z > 1:
        stacks = np.zeros(
            (n_channels, n_time_points, meta_n_z, height, width), dtype="uint16"
        )
        image_size = width * height
        t_size = (
            width * height * n_z * n_channels
        )  # number of values stored for a given time point (this includes images for all channels)
        with open(path, "rb") as f:
            for t in range(n_time_points):
                # print('{}/{}'.format(t,n_time_points))
                a = array.array("H")
                a.fromfile(f, t_size)
                a = np.array(a).reshape(
                    (-1, image_size)
                )  # each row is an image alternating between channels
                for c in range(n_channels):
                    stacks[c, t, :, :, :] = a[c::n_channels, :].reshape(
                        (n_z, height, width)
                    )[:meta_n_z, :, :]

    area_mode = metadata.get_area_mode()
    if (area_mode == "line" or area_mode == "kymograph") and meta_n_z > 1:
        concatenated = []
        for stack in stacks:
            concatenated.append(concatenate_z(stack))
        stacks = concatenated

    if len(stacks) == 1:
        return (np.squeeze(stacks[0]),)
    return tuple(np.squeeze(stacks))


def load_z_stack(path, metadata):
    """
    Loads tif files as saved when capturing a z-stack into a 3D numpy array.

    Parameters
    ----------
    path : string
        Path to directory of the z-stack.
    metadata : ElementTree root
        Can be obtained with :func:`get_metadata`.

    Returns
    -------
    stacks : tuple of numpy arrays
        Z-stacks for Channel A (green) and Channel B (red).
        
    Examples
    --------
    >>> import utils2p
    >>> metadata = utils2p.Metadata("data/mouse_kidney_z_stack/Experiment.xml")
    >>> z_stack_A, z_stack_B = utils2p.load_z_stack("data/mouse_kidney_z_stack/", metadata)
    >>> type(z_stack_A), type(z_stack_B)
    (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>)
    >>> z_stack_A.shape, z_stack_B.shape
    ((3, 128, 128), (3, 128, 128))
    """
    path = os.path.expanduser(os.path.expandvars(path))
    channels = metadata.get_channels()
    paths = sorted(glob.glob(os.path.join(path, channels[0]) + "*.tif"))
    stacks = load_img(paths[0])
    if stacks.ndim == 5:
        return tuple([stacks[:, :, 0, :, :], stacks[:, :, 1, :, :]])
    return tuple([stacks[:, 0, :, :], stacks[:, 1, :, :]])


def concatenate_z(stack):
    """
    Concatenate in z direction for area mode 'line' or 'kymograph',
    e.g. coronal section. This is necessary because z steps are
    otherwise treated as additional temporal frame, i.e. in Fiji
    the frames jump up and down between z positions.

    Parameters
    ----------
    stack : 4D or 6D numpy array
        Stack to be z concatenated.

    Returns
    -------
    stack : 3D or 5D numpy array
        Concatenated stack.

    Examples
    --------
    >>> import utils2p
    >>> import numpy as np
    >>> stack = np.zeros((100, 2, 64, 128))
    >>> concatenated = utils2p.concatenate_z(stack)
    >>> concatenated.shape
    (100, 128, 128)
    """
    res = np.concatenate(np.split(stack, stack.shape[-3], axis=-3), axis=-2)
    return np.squeeze(res)


def save_img(
    path, img, imagej=True, color=False, full_dynamic_range=True, metadata=None
):
    """
    Saves an image that is given as a numpy array to file.

    Parameters
    ----------
    path : string
        Path where the file is saved.
    img : numpy array
        Image or stack. For stacks, the first dimension is the stack index.
        For color images, the last dimension are the RGB channels.
    imagej : boolean
        Save imagej compatible stacks and hyperstacks.
    color : boolean, default = False
        Determines if image is RGB or gray scale.
        Will be converted to uint8.
    full_dynamic_range : boolean, default = True
        When an image is converted to uint8 for saving a color image the
        max value of the output image is the max of uint8,
        i.e. the image uses the full dynamic range available.

    Examples
    --------
    >>> import utils2p
    >>> import numpy as np
    >>> 
    """
    if img.dtype == np.bool:
        img = img.astype(np.uint8) * 255
    path = os.path.expanduser(os.path.expandvars(path))
    if color:
        if img.dtype != np.uint8:
            old_max = np.max(img, axis=tuple(range(img.ndim - 1)))
            if not full_dynamic_range:
                if np.issubdtype(img.dtype, np.integer):
                    old_max = np.iinfo(img.dtype).max * np.ones(3)
                elif np.issubdtype(img.dtype, np.floating):
                    old_max = np.finfo(img.dtype).max * np.ones(3)
                else:
                    raise ValueError(
                        f"img must be integer or float type not {img.dtype}"
                    )
            new_max = np.iinfo(np.uint8).max
            img = img / old_max * new_max
            img = img.astype(np.uint8)
        if imagej and img.ndim == 4:
            img = np.expand_dims(img, axis=1)
        if imagej and img.ndim == 3:
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=1)
    else:
        if imagej and img.ndim == 4:
            img = np.expand_dims(img, axis=2)
            img = np.expand_dims(img, axis=5)
        if imagej and img.ndim == 3:
            img = np.expand_dims(img, axis=1)
            img = np.expand_dims(img, axis=4)
    if img.dtype == np.float64:
        img = img.astype(np.float32)
    if metadata is None:
        tifffile.imsave(path, img, imagej=imagej)
    else:
        # TODO add meta data like metadata={'xresolution':'4.25','yresolution':'0.0976','PixelAspectRatio':'43.57'}
        # tifffile.imsave(path, img, imagej=imagej, metadata={})
        raise NotImplemented("Saving of metadata is not yet implemented")


def _find_file(directory, name, file_type):
    """
    This function finds a unique file with a given name in
    in the directory.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    name : str
        Name of the file.

    Returns
    -------
    path : str
        Path to file.
    """
    file_names = list(Path(directory).rglob("*" + name))
    if len(file_names) > 1:
        raise RuntimeError(
            f"Could not identify {file_type} file unambiguously. Discovered {len(file_names)} {file_type} files in {directory}."
        )
    elif len(file_names) == 0:
        raise FileNotFoundError(f"No {file_type} file found in {directory}")
    return str(file_names[0])


def find_metadata_file(directory):
    """
    This functions find the path to the metadata file
    "Experiment.xml" created by ThorImage and returns it.
    If multiple files with this name are found, it throws
    and exception.

    Parameters
    ----------
    directory : str
        Directory in which to search.

    Returns
    -------
    path : str
        Path to metadata file.

    Examples
    --------
    >>> import utils2p
    >>> utils2p.find_metadata_file("data/mouse_kidney_z_stack")
    'data/mouse_kidney_z_stack/Experiment.xml'
    """
    return _find_file(directory, "Experiment.xml", "metadata")


def find_sync_file(directory):
    """
    This functions find the path to the sync file
    "Episode001.h5" created by ThorSync and returns it.
    If multiple files with this name are found, it throws
    and exception.

    Parameters
    ----------
    directory : str
        Directory in which to search.

    Returns
    -------
    path : str
        Path to sync file.

    Examples
    --------
    >>> import utils2p
    >>> utils2p.find_sync_file("data/mouse_kidney_z_stack")
    'data/mouse_kidney_z_stack/Episode001.h5'
    """
    return _find_file(directory, "Episode001.h5", "synchronization")


def find_raw_file(directory):
    """
    This functions find the path to the raw file
    "Image_0001_0001.raw" created by ThorImage and returns it.
    If multiple files with this name are found, it throws
    and exception.

    Parameters
    ----------
    directory : str
        Directory in which to search.

    Returns
    -------
    path : str
        Path to raw file.

    Examples
    --------
    >>> import utils2p
    >>> utils2p.find_raw_file("data/mouse_kidney_raw")
    'data/mouse_kidney_raw/Image_0001_0001.raw'
    """
    return _find_file(directory, "Image_0001_0001.raw", "raw")


def load_optical_flow(
    path: str, gain_0_x: float, gain_0_y: float, gain_1_x: float, gain_1_y: float
):
    """
    This function loads the optical flow data from
    the file specified in path. By default it is
    directly converted into ball rotation. Gain values
    have to be determined with the calibration of the
    optical flow sensors.
    
    Parameters
    ----------
    path : str
        Path to file holding the optical flow data.
    gain_0_x: float
        Gain for the x direction of sensor 0.
    gain_0_y: float
        Gain for the y direction of sensor 0.
    gain_1_x: float
        Gain for the x direction of sensor 1.
    gain_1_y: float
        Gain for the y direction of sensor 1.

    Returns
    -------
    data : dictionary
        A dictionary with keys: 'sensor0', 'sensor1',
        'time_stamps', 'vel_pitch', 'vel_yaw', 'vel_roll'.

    Examples
    --------
    >>> import utils2p
    >>> gain_0_x = round(1 / 1.45, 2)
    >>> gain_0_y = round(1 / 1.41, 2)
    >>> gain_1_x = round(1 / 1.40, 2)
    >>> gain_1_y = round(1 / 1.36, 2)

    >>> optical_flow = utils2p.load_optical_flow("data/behData/OptFlowData/OptFlow.txt", gain_0_x, gain_0_y, gain_1_x, gain_1_y)
    >>> type(optical_flow)
    <class 'dict'>
    >>> optical_flow.keys()
    dict_keys(['sensor0', 'sensor1', 'time_stamps', 'vel_pitch', 'vel_yaw', 'vel_roll'])

    >>> type(optical_flow["time_stamps"])
    <class 'numpy.ndarray'>
    >>> optical_flow["time_stamps"].shape
    (1000,)

    >>> type(optical_flow["vel_pitch"])
    <class 'numpy.ndarray'>
    >>> optical_flow["vel_pitch"].shape
    (1000,)

    >>> type(optical_flow["vel_yaw"])
    <class 'numpy.ndarray'>
    >>> optical_flow["vel_yaw"].shape
    (1000,)

    >>> type(optical_flow["vel_roll"])
    <class 'numpy.ndarray'>
    >>> optical_flow["vel_roll"].shape
    (1000,)

    >>> type(optical_flow["sensor0"])
    <class 'dict'>
    >>> optical_flow["sensor0"].keys()
    dict_keys(['x', 'y', 'gain_x', 'gain_y'])
    """
    raw_data = np.genfromtxt(path, delimiter=",")
    data = {
        "sensor0": {
            "x": raw_data[:, 0],
            "y": raw_data[:, 1],
            "gain_x": gain_0_x,
            "gain_y": gain_0_y,
        },
        "sensor1": {
            "x": raw_data[:, 2],
            "y": raw_data[:, 3],
            "gain_x": gain_1_x,
            "gain_y": gain_1_y,
        },
        "time_stamps": raw_data[:, 4],
    }

    data["vel_pitch"] = -(
        data["sensor0"]["y"] * data["sensor0"]["gain_y"]
        + data["sensor1"]["y"] * data["sensor1"]["gain_y"]
    ) * np.cos(np.deg2rad(45))
    data["vel_yaw"] = (
        data["sensor0"]["x"] * data["sensor0"]["gain_x"]
        + data["sensor1"]["x"] * data["sensor1"]["gain_x"]
    ) / 2.0
    data["vel_roll"] = (
        data["sensor0"]["y"] * data["sensor0"]["gain_y"]
        - data["sensor1"]["y"] * data["sensor1"]["gain_y"]
    ) * np.sin(np.deg2rad(45))

    return data
