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

from .external import tifffile

package_directory = os.path.dirname(os.path.abspath(__file__))


class InvalidValueInMetaData(Exception):
    """This error should be raised when an invalid value
    is encountered in an 'Experiement.xml' file."""

    pass


def _node_crawler(node, *args):
    if len(args) == 0:
        return node
    elif len(args) == 1 and args[0] in node.attrib.keys():
        return node.attrib[args[0]]
    if len(node) == 0:
        raise ValueError(f"Hit dead end {node} has no children.")
    return [_node_crawler(child, *args[1:]) for child in node.findall(args[0])]


class _XMLFile:
    """
    Base class for xml based Metadata.
    """
    def __init__(self, path):
        self.path = path
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()

    def get_value(self, *args):
        node = self.root.find(args[0])
        values = _node_crawler(node, *args[1:])
        if len(values) == 1:
            return values[0]
        return values


class Metadata(_XMLFile):
    """
    Class for managing ThorImage metadata.
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
    def get_metadata_value(self, *args):
        """
        This function returns a value from the metadata file 'Experiment.xml'.

        Parameters
        ----------
        args : strings
            Arbitrary number of strings of tags from the xml file in the
            correct order. See examples.

        Returns
        -------
        attribute or node : string or ElementTree node
            If the number of strings given in args leads to a leaf of the tree,
            the attribute, usually a dictionary, is returned.
            Otherwise the node is returned.

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
        return self.get_value(*args)

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
        if int_area_mode == 1:
            return "rectangle"
        if int_area_mode == 2:
            return "kymograph"
        if int_area_mode == 3:
            return "line"
        raise InvalidValueInMetaData(
            f"{int_area_mode} is not a valid value for areaMode.")

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

    def get_n_averaging(self):
        """
        Returns the number of frames that are averaged.

        Returns
        -------
        n_averaging : int
            Number of averaged frames.

        Examples
        --------
        >>> import utils2p
        >>> metadata = Metadata('data/mouse_kidney_time_series_z_stack/Experiment.xml')
        >>> metadata.get_n_averaging()
        10
        """
        return int(self.get_value("LSM", "averageNum"))

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
        Returns a tuple with the names of all channels.

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
        channels = self.get_metadata_value("Wavelengths", "Wavelength", "name")
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
        if area_mode in ('line', 'kymograph'):
            return (float(self.get_metadata_value("ZStage", "stepSizeUM")) *
                    self.get_n_z() / self.get_num_y_pixels())
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
        frame_rate_without_flybacks = float(
            self.get_metadata_value("LSM", "frameRate"))
        flyback_frames = self.get_n_flyback_frames()
        number_of_slices = self.get_n_z()
        return frame_rate_without_flybacks / (flyback_frames +
                                              number_of_slices)

    def get_width(self):
        """
        Returns the image width in um for a given experiment metadata.

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

    def get_gain_a(self):
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
        >>> metadata.get_gain_a()
        20.0
        """
        return float(self.get_metadata_value("PMT", "gainA"))

    def get_gain_b(self):
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
        >>> metadata.get_gain_b()
        30.0
        """
        return float(self.get_metadata_value("PMT", "gainB"))

    def get_date_time(self):
        """
        Returns the date and time of an experiment
        for a given experiment metadata.

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


def load_img(path, memmap=False):
    """
    This functions loads an image from file and returns as a numpy array.

    Parameters
    ----------
    path : string
        Path to image file.
    memmap : bool
        If `True`, the image is not loaded into memory but remains on
        disk and a `numpy.memmap` object is returned. It can be indexed
        like a normal `numpy.array`. This option is useful when a stack
        does not fit into memory. It can also be used when opening many
        stacks simultaneously.

    Returns
    -------
    numpy.array or numpy.memmap
        Image in form of numpy array or numpy memmap.

    Examples
    --------
    >>> import utils2p
    >>> img = utils2p.load_img("data/chessboard_GRAY_U16.tif")
    >>> type(img)
    <class 'numpy.ndarray'>
    >>> img.shape
    (200, 200)
    >>> img = utils2p.load_img("data/chessboard_GRAY_U16.tif", memmap=True)
    >>> type(img)
    <class 'numpy.memmap'>
    """
    path = os.path.expanduser(os.path.expandvars(path))
    return tifffile.imread(path, memmap=memmap)


def load_stack_batches(path, batch_size):
    """
    This function loads a stack in several batches to make sure
    the system does not run out of memory. It returns a generator
    that yields consecutive chunks of `batch_size` frames of the stack.
    The remaining memory is freed up by the function until the generator
    is called again.

    Parameters
    ----------
    path : string
        Path to stack.
    batch_size : int
        Number of frames in one chunk.

    Returns
    -------
    generator
        Generator that yields chunks of `batch_size` frames of the
        stack.
    """
    stack = load_img(path, memmap=True)
    if stack.ndim < 3:
        raise ValueError(
            f"The path does not point to a stack. The shape is {stack.shape}.")
    n_batches = int(stack.shape[0] / batch_size) + 1
    for i in range(n_batches):
        substack = np.array(stack[i * batch_size:(i + 1) * batch_size])
        yield substack


def load_stack_patches(path, patch_size, padding=0, return_indices=False):
    """
    Returns a generator that yields patches of the stack of images.
    This is useful when multiple stacks should be processed but they
    don't fit into memory, e.g. when computing an overall fluorescence
    baseline for all trials of a fly.

    Parameters
    ----------
    path : string
       Path to stack.
    patch_size : tuple of two integers
       Size of the patch returned.
    padding : integer or tuple of two integers
       The amount of overlap between patches. Note that this increases
       the effective patch size. Default is 0. If tuple, different padding
       is used for the dimensions.
    return_indices : boolean
       If True, the indices necessary for slicing to generate the patch and
       the indices necessary for slicing to remove the padding from the
       returned patch are returned. Default is False.
       The values are retuned in the following form:
       ```
       indices = [[start_patch_dim_0, stop_patch_dim_0],
                 [start_patch_dim_1, stop_patch_dim_1],]
       patch_indices = [[start_after_padding_dim_0, stop_after_padding_dim_0],
                        [start_after_padding_dim_1, stop_after_padding_dim_1],]
       ```

    Returns
    -------
    patch : numpy array
        Patch of the stack.
    indices : tuples of integers, optional
        See description of the `return_indices` parameter above
        and examples below.
    patch_indices : tuples of integers, optinal
        See description of the `return_indices` parameter above
        and examples below.

    Examples
    --------
    >>> import numpy as np
    >>> import utils2p
    >>> metadata = utils2p.Metadata('data/mouse_kidney_raw/2p/Untitled_001/Experiment.xml')
    >>> stack1, stack2 = utils2p.load_raw('data/mouse_kidney_raw/2p/Untitled_001/Image_0001_0001.raw',metadata)
    >>> print(stack1.shape)
    (5, 256, 256)
    >>> utils2p.save_img('stack1.tif',stack1)
    >>> generator = utils2p.load_stack_patches('stack1.tif', (5, 4))
    >>> first_patch = next(generator)
    >>> print(first_patch.shape)
    (5, 5, 4)
    >>> generator = utils2p.load_stack_patches('stack1.tif', (15, 20), padding=3, return_indices=True)
    >>> first_patch, indices, patch_indices = next(generator)
    >>> print(first_patch.shape)
    (5, 18, 23)
    >>> print(patch_indices)
    [[0, 15], [0, 20]]
    >>> print(indices)
    [[0, 15], [0, 20]]
    >>> first_patch_without_padding = first_patch[:, patch_indices[0][0] : patch_indices[0][1], patch_indices[1][0] : patch_indices[1][1]]
    >>> print(first_patch_without_padding.shape)
    (5, 15, 20)
    >>> np.all(stack1[:, indices[0][0] : indices[0][1], indices[1][0] : indices[1][1]] == first_patch_without_padding)
    True

    Note that the patch has no padding at the edges.
    When looking at the second patch we see that it is padded on both side
    in the second dimension but still only on one side of the first dimension.

    >>> second_patch, indices, patch_indices = next(generator)
    >>> print(second_patch.shape)
    (5, 18, 26)
    >>> print(patch_indices)
    [[0, 15], [3, 23]]
    >>> print(indices)
    [[0, 15], [20, 40]]
    >>> second_patch_without_padding = second_patch[:, patch_indices[0][0] : patch_indices[0][1], patch_indices[1][0] : patch_indices[1][1]]
    >>> print(second_patch_without_padding.shape)
    (5, 15, 20)
    """
    stack = load_img(path, memmap=True)
    dims = stack.shape[1:]
    n_patches_0 = math.ceil(dims[0] / patch_size[0])
    n_patches_1 = math.ceil(dims[1] / patch_size[1])
    if isinstance(padding, int):
        padding = (padding, padding)
    for i in range(n_patches_0):
        for j in range(n_patches_1):
            indices = [
                [patch_size[0] * i, patch_size[0] * (i + 1)],
                [patch_size[1] * j, patch_size[1] * (j + 1)],
            ]
            start_dim_0 = max(indices[0][0] - padding[0], 0)
            start_dim_1 = max(indices[1][0] - padding[1], 0)
            stop_dim_0 = min(indices[0][1] + padding[0], dims[0])
            stop_dim_1 = min(indices[1][1] + padding[1], dims[1])
            patch = stack[:, start_dim_0:stop_dim_0,
                          start_dim_1:stop_dim_1].copy()
            del stack
            if not return_indices:
                yield patch
            else:
                offset_dim_0 = indices[0][0] - start_dim_0
                offset_dim_1 = indices[1][0] - start_dim_1
                patch_indices = [
                    [offset_dim_0, patch_size[0] + offset_dim_0],
                    [offset_dim_1, patch_size[1] + offset_dim_1],
                ]
                yield patch, indices, patch_indices
            stack = load_img(path)


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
        Number of numpy arrays depends on the number of channels recoded during
        the experiment. Has the following dimensions:
        TZYX or TYX for planar images.

    Examples
    --------
    >>> import utils2p
    >>> metadata = utils2p.Metadata('data/mouse_kidney_raw/2p/Untitled_001/Experiment.xml')
    >>> stack1, stack2 = utils2p.load_raw('data/mouse_kidney_raw/2p/Untitled_001/Image_0001_0001.raw',metadata)
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
        not n_z %
        1), "Size given in metadata does not match the size of the raw file."
    n_z = int(n_z)

    # number of z slices from meta data can be different
    # because of flyback frames
    meta_n_z = metadata.get_n_z()

    if n_z == 1:
        stacks = np.zeros((n_channels, n_time_points, height, width),
                          dtype="uint16")
        image_size = width * height
        # number of values stored for a given time point
        # (this includes images for all channels)
        t_size = (width * height * n_channels)
        with open(path, "rb") as f:
            for t in range(n_time_points):
                # print('{}/{}'.format(t,n_time_points))
                a = array.array("H")
                a.fromfile(f, t_size)
                for c in range(n_channels):
                    stacks[c, t, :, :] = np.array(a[c * image_size:(c + 1) *
                                                    image_size]).reshape(
                                                        (height, width))
    elif n_z > 1:
        stacks = np.zeros((n_channels, n_time_points, meta_n_z, height, width),
                          dtype="uint16")
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
                    (-1, image_size
                     ))  # each row is an image alternating between channels
                for c in range(n_channels):
                    stacks[c, t, :, :, :] = a[c::n_channels, :].reshape(
                        (n_z, height, width))[:meta_n_z, :, :]

    area_mode = metadata.get_area_mode()
    if area_mode in ('line', 'kymograph') and meta_n_z > 1:
        concatenated = []
        for stack in stacks:
            concatenated.append(concatenate_z(stack))
        stacks = concatenated

    if len(stacks) == 1:
        return (np.squeeze(stacks[0]), )
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


def save_img(path,
             img,
             imagej=True,
             color=False,
             full_dynamic_range=True,
             metadata=None):
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
                        f"img must be integer or float type not {img.dtype}")
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
        raise NotImplementedError("Saving of metadata is not yet implemented")


def _find_file(directory, name, file_type, most_recent=True):
    """
    This function finds a unique file with a given name in
    in the directory.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    name : str
        Name of the file.
    most_recent : bool
        If True, the file with the most recent change time
        is returned and no exception is raised if multiple
        files are present.

    Returns
    -------
    path : str
        Path to file.
    """
    file_names = list(Path(directory).rglob(name))
    if len(file_names) > 1:
        if most_recent:
            change_times = [os.stat(path).st_mtime for path in file_names]
            file_names = (file_names[np.argmax(change_times)], )
        else:
            raise RuntimeError(
                f"Could not identify {file_type} file unambiguously. " +
                f"Discovered {len(file_names)} {file_type} files in {directory}."
            )
    elif len(file_names) == 0:
        raise FileNotFoundError(f"No {file_type} file found in {directory}")
    return str(file_names[0])


def find_metadata_file(directory, most_recent=False):
    """
    This function finds the path to the metadata file
    "Experiment.xml" created by ThorImage and returns it.
    If multiple files with this name are found, it throws
    an exception unless `most_recent` is `True`, in which case
    the file with the most recent change time is returned.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    most_recent : bool
        If True, the file with the most recent change time
        is returned and no exception is raised if multiple
        files are present.

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
    return _find_file(directory,
                      "Experiment.xml",
                      "metadata",
                      most_recent=most_recent)


def find_seven_camera_metadata_file(directory, most_recent=False):
    """
    This function finds the path to the metadata file
    "capture_metadata.json" created by seven camera
    setup and returns it.
    If multiple files with this name are found, it throws
    an exception unless `most_recent` is `True`, in which case
    the file with the most recent change time is returned.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    most_recent : bool
        If True, the file with the most recent change time
        is returned and no exception is raised if multiple
        files are present.

    Returns
    -------
    path : str
        Path to capture metadata file.

    Examples
    --------
    >>> import utils2p
    >>> utils2p.find_seven_camera_metadata_file("data/mouse_kidney_raw")
    'data/mouse_kidney_raw/behData/images/capture_metadata.json'
    """
    return _find_file(directory,
                      "capture_metadata.json",
                      "seven camera capture metadata",
                      most_recent=most_recent)


def find_sync_file(directory, most_recent=False):
    """
    This function finds the path to the sync file
    "Episode001.h5" created by ThorSync and returns it.
    If multiple files with this name are found, it throws
    an exception unless `most_recent` is `True`, in which case
    the file with the most recent change time is returned.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    most_recent : bool
        If True, the file with the most recent change time
        is returned and no exception is raised if multiple
        files are present.

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
    return _find_file(directory,
                      "Episode001.h5",
                      "synchronization",
                      most_recent=most_recent)


def find_optical_flow_file(directory, most_recent=False):
    """
    This function finds the path to the optical flow file
    "OptFlow.txt" created by seven camera software and returns it.
    If multiple files with this name are found, it throws
    an exception unless `most_recent` is `True`,in which case
    the file with the most recent change time is returned.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    most_recent : bool
        If True, the file with the most recent change time
        is returned and no exception is raised if multiple
        files are present.

    Returns
    -------
    path : str
        Path to optical flow file.

    Examples
    --------
    >>> import utils2p
    >>> utils2p.find_optical_flow_file("data/mouse_kidney_raw")
    'data/mouse_kidney_raw/behData/OptFlowData/OptFlow.txt'
    """
    return _find_file(directory,
                      "OptFlow.txt",
                      "optical flow",
                      most_recent=most_recent)


def find_raw_file(directory, most_recent=False):
    """
    This function finds the path to the raw file
    "Image_0001_0001.raw" created by ThorImage and returns it.
    If multiple files with this name are found, it throws
    an exception unless `most_recent` is `True`, in which case
    the file with the most recent change time is returned.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    most_recent : bool
        If True, the file with the most recent change time
        is returned and no exception is raised if multiple
        files are present.

    Returns
    -------
    path : str
        Path to raw file.

    Examples
    --------
    >>> import utils2p
    >>> utils2p.find_raw_file("data/mouse_kidney_raw")
    'data/mouse_kidney_raw/2p/Untitled_001/Image_0001_0001.raw'
    """
    return _find_file(directory,
                      "Image_0001_0001.raw",
                      "raw",
                      most_recent=most_recent)


def find_sync_metadata_file(directory, most_recent=False):
    """
    This function finds the path to the synchronization
    metadata file "ThorRealTimeDataSettings.xml" created
    by ThorSync. If multiple files with this name are found,
    it throws an exception unless `most_recent` is `True`,
    in which case the file with the most recent change time
    is returned.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    most_recent : bool
        If True, the file with the most recent change time
        is returned and no exception is raised if multiple
        files are present.

    Returns
    -------
    path : str
        Path to synchronization metadata file.

    Examples
    --------
    >>> import utils2p
    >>> utils2p.find_sync_metadata_file("data/mouse_kidney_raw")
    'data/mouse_kidney_raw/2p/Sync-025/ThorRealTimeDataSettings.xml'

    """
    return _find_file(directory,
                      "ThorRealTimeDataSettings.xml",
                      "synchronization metadata",
                      most_recent=most_recent)


def find_fictrac_file(directory, camera=3, most_recent=False):
    """
    This function finds the path to the output file of
    fictrac of the form `camera_{cam}*.dat`, where
    `{cam}` is the values specified in the `camera`
    argument. If multiple files with this name are found,
    it throws an exception unless `most_recent` is `True`,
    in which case the file with the most recent change time
    is returned.

    Parameters
    ----------
    directory : str
        Directory in which to search.
    camera : int
        The camera used for fictrac.
    most_recent : bool
        If True, the file with the most recent change time
        is returned and no exception is raised if multiple
        files are present.

    Returns
    -------
    path : str
        Path to fictrac output file.

    Examples
    --------
    >>> import utils2p
    >>> utils2p.find_fictrac_file("data")
    Traceback (most recent call last):
     ...
    RuntimeError: Could not identify fictrac output file unambiguously. Discovered 2 fictrac output files in data.
    >>> utils2p.find_fictrac_file("data", most_recent=True)
    'data/camera_3-20210803_103010.dat'
    """
    return _find_file(directory,
                      f"camera_{camera}*.dat",
                      "fictrac output",
                      most_recent=most_recent)


def load_optical_flow(path: str,
                      gain_0_x: float,
                      gain_0_y: float,
                      gain_1_x: float,
                      gain_1_y: float,
                      smoothing_kernel=None):
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
    smoothing_kernel: numpy array, optional
        Default is None, in which case the sensor signals
        are not smoothed. The signal is convolved with this
        kernel. A reasonable choice seems to be a moving average
        filter of length 300: np.ones(300) / 300.

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

    >>> optical_flow_file = utils2p.find_optical_flow_file("data/mouse_kidney_raw")
    >>> optical_flow = utils2p.load_optical_flow(optical_flow_file, gain_0_x, gain_0_y, gain_1_x, gain_1_y)
    >>> type(optical_flow)
    <class 'dict'>
    >>> optical_flow.keys()
    dict_keys(['sensor0', 'sensor1', 'time_stamps', 'vel_pitch', 'vel_yaw', 'vel_roll'])

    >>> type(optical_flow["time_stamps"])
    <class 'numpy.ndarray'>
    >>> optical_flow["time_stamps"].shape
    (1408,)

    >>> type(optical_flow["vel_pitch"])
    <class 'numpy.ndarray'>
    >>> optical_flow["vel_pitch"].shape
    (1408,)

    >>> type(optical_flow["vel_yaw"])
    <class 'numpy.ndarray'>
    >>> optical_flow["vel_yaw"].shape
    (1408,)

    >>> type(optical_flow["vel_roll"])
    <class 'numpy.ndarray'>
    >>> optical_flow["vel_roll"].shape
    (1408,)

    >>> type(optical_flow["sensor0"])
    <class 'dict'>
    >>> optical_flow["sensor0"].keys()
    dict_keys(['x', 'y', 'gain_x', 'gain_y'])

    >>> optical_flow = utils2p.load_optical_flow(optical_flow_file, gain_0_x, gain_0_y, gain_1_x, gain_1_y, smoothing_kernel=np.ones(300) / 300)
    >>> optical_flow["vel_pitch"].shape
    (1408,)
    """
    raw_data = np.genfromtxt(path, delimiter=",")
    if smoothing_kernel is not None:
        if len(smoothing_kernel) >= raw_data.shape[0]:
            raise ValueError(
                f"smoothing_kernel of shape {smoothing_kernel.shape} " +
                f"is longer than optical flow data of shape {raw_data.shape}.")
        raw_data = np.apply_along_axis(
            lambda m: np.convolve(m, smoothing_kernel, mode="same"),
            axis=0,
            arr=raw_data)
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

    data["vel_pitch"] = -(data["sensor0"]["y"] * data["sensor0"]["gain_y"] +
                          data["sensor1"]["y"] *
                          data["sensor1"]["gain_y"]) * np.cos(np.deg2rad(45))
    data["vel_yaw"] = (data["sensor0"]["x"] * data["sensor0"]["gain_x"] +
                       data["sensor1"]["x"] * data["sensor1"]["gain_x"]) / 2.0
    data["vel_roll"] = (data["sensor0"]["y"] * data["sensor0"]["gain_y"] -
                        data["sensor1"]["y"] *
                        data["sensor1"]["gain_y"]) * np.sin(np.deg2rad(45))

    return data


def load_fictrac(path, ball_radius=5, fps=100):
    """
    This functions loads the fictrac data from file.

    Parameters
    ----------
    path : str
        Path to fictrac output file (.dat).
    ball_radius : int
        Radius of the spherical treadmill.
    fps : float
        Number of frames per second.

    Returns
    -------
    data : dictionary
        A dictionary with the following keys:
        Speed, x, y, forward_pos, side_pos, delta_rot_lab_side,
        delta_rot_lab_forward, delta_rot_lab_turn, integrated_forward_movement,
        integrated_side_movement, Time
        All speeds are in mm/s and all positions are in mm.
    """
    col_names = [
        "Frame_counter", "delta_rot_cam_right", "delta_rot_cam_down",
        "delta_rot_cam_forward", "delta_rot_error", "delta_rot_lab_side",
        "delta_rot_lab_forward", "delta_rot_lab_turn", "abs_rot_cam_right",
        "abs_rot_cam_down", "abs_rot_cam_forward", "abs_rot_lab_side",
        "abs_rot_lab_forward", "abs_rot_lab_turn", "integrated_lab_x",
        "integrated_lab_y", "integrated_lab_heading",
        "animal_movement_direction_lab", "animal_movement_speed",
        "integrated_forward_movement", "integrated_side_movement", "timestamp",
        "seq_counter", "delta_time", "alt_time"
    ]

    dat_table = np.genfromtxt(path, delimiter=",")
    data = {}
    for i, col in enumerate(col_names):
        data[col] = dat_table[:, i]
    data["Speed"] = data["animal_movement_speed"] * ball_radius * fps
    data["x"] = data["integrated_lab_x"] * ball_radius
    data["y"] = data["integrated_lab_y"] * ball_radius
    data["forward_pos"] = data["integrated_forward_movement"] * ball_radius
    data["side_pos"] = data["integrated_side_movement"] * ball_radius
    data["delta_rot_lab_side"] = data["delta_rot_lab_side"] * ball_radius * fps
    data["delta_rot_lab_forward"] = data[
        "delta_rot_lab_forward"] * ball_radius * fps
    data["delta_rot_lab_turn"] = data[
        "delta_rot_lab_turn"] / 2 / np.pi * 360 * fps
    data["integrated_forward_movement"] = data[
        "integrated_forward_movement"] * ball_radius
    data["integrated_side_movement"] = data[
        "integrated_side_movement"] * ball_radius
    data["Time"] = data["Frame_counter"] / fps

    return data
