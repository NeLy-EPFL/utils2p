"""
This module provides unit test for the functions provided in utils2p.main.
"""

import os.path
import struct
from pathlib import Path

import pytest
import numpy as np

import utils2p
import utils2p.errors
from utils2p.external import tifffile


data_dir = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def metadata_xml(tmpdir):
    """
    This pytest factory constructs a fake metadata xml file, saves it to file
    and returns the path.
    """

    def _metadata_xml(
        timepoints=0,
        x_pixels=10,
        y_pixels=10,
        n_z=3,
        area_mode=2,
        channels=("ChanA", "ChanB"),
        pixel_size=1,
        z_step_size=1,
        dwell_time=1,
        frame_rate=10,
        width=1,
        gain_a=1,
        gain_b=1,
        flyback_frames=3,
        power=1,
        date="12/20/2018 18:33:52",
    ):
        wavelength_strings = ""
        for channel in channels:
            wavelength_strings = (
                wavelength_strings
                + f"""<Wavelength name="{channel}" exposuretimeMS="0" />"""
            )
        content = f"""<?xml version="1.0"?>
        <ThorImageExperiment>
            <Date date="{date}" />
            <Wavelengths nyquistExWavelengthNM="488" nyquistEmWavelengthNM="488">
                {wavelength_strings}
                <ChannelEnable Set="3" />
            </Wavelengths>
            <ZStage steps="{n_z}" stepSizeUM="{z_step_size}" />
            <Timelapse timepoints="{timepoints}" />
            <LSM pixelX="{x_pixels}" pixelY="{y_pixels}" areaMode="{area_mode}" pixelSizeUM="{pixel_size}" dwellTime="{dwell_time}" frameRate="{frame_rate}" widthUM="{width}" />
            <PMT gainA="{gain_a}" gainB="{gain_b}" />
            <Streaming flybackFrames="{flyback_frames}" />
            <PowerRegulator start="{power}" />
        </ThorImageExperiment>
        """
        path = tmpdir.join("Experiment.xml")
        path.write(content)
        return path

    return _metadata_xml


@pytest.fixture
def metadata_obj(metadata_xml):
    """
    This pytest factory returns an instance of the :class:`utils2p.Metadata` class
    based on the file given in `metadata_xml`. For the purpose of the test in
    this module metadata_xml it the pytest fixture :function:`.metadata_xml`.

    Parameters
    ----------
    metadata_xml : string
        Path to Experiment xml file.
    """

    def _metadata_obj(
        timepoints=0,
        x_pixels=10,
        y_pixels=10,
        n_z=3,
        area_mode=2,
        channels=("ChanA", "ChanB"),
        pixel_size=1,
        z_step_size=1,
        dwell_time=1,
        frame_rate=10,
        width=1,
        gain_a=1,
        gain_b=1,
        flyback_frames=3,
        power=1,
        date="12/20/2018 18:33:52",
    ):
        path = metadata_xml(
            timepoints=timepoints,
            x_pixels=x_pixels,
            y_pixels=y_pixels,
            n_z=n_z,
            area_mode=area_mode,
            channels=channels,
            pixel_size=pixel_size,
            z_step_size=z_step_size,
            dwell_time=dwell_time,
            frame_rate=frame_rate,
            width=width,
            gain_a=gain_a,
            gain_b=gain_b,
            flyback_frames=flyback_frames,
            power=power,
            date=date,
        )
        return utils2p.Metadata(str(path))

    return _metadata_obj


@pytest.fixture
def random_stack():
    """
    This pytest fixture generates a random array of given
    size with uint16 values up to :math:`2^{14}` as generated
    by the Aalazar card of the microscope.

    Parameters
    ----------
    shape : tuple
        Shape of array. Default is (10, 50, 60).

    Returns
    -------
    img_stack : numpy array
        Random array.
    """

    def _random_stack(shape=(10, 50, 60)):
        img_stack = np.random.randint(0, 2 ** 14, shape, dtype=np.uint16)
        return img_stack

    return _random_stack


@pytest.fixture
def random_tif_file(tmpdir, random_stack):
    """
    This pytest factory writes a random stack retuned
    by :func:`.random_stack` to file a tif file and returns
    the path and the matrix.

    Parameters
    ----------
    shape : tuple
        Shape of stack. Default is (10, 50, 60).

    Returns
    -------
    file_path : PosixPath
        Path to tif file.
    random_stack : numpy array
        Matrix of the stack.
    """

    def _random_tif_file(shape=(10, 50, 60)):
        file_path = tmpdir.join("img_stack.tif")
        img_stack = random_stack(shape)
        tifffile.imsave(str(file_path), img_stack)
        return file_path, img_stack

    return _random_tif_file


@pytest.fixture
def random_raw_file(tmpdir, random_stack, metadata_obj):
    """
    This pytest factory writes as random stack to a binary file as it is
    saved by ThorImage in raw data capture mode.

    Parameters
    ----------
    area_mode : int
        Area mode of the acquisition.
    shape : tuple of int
        Shape of a single x-y frame.
    timepoints : int
        Number of timepoints.
    channels : tuple of strings
        Names of the channels.
    n_z : int
        Number of steps in z direction.
    flyback_frames : int
        Number of flyback frames.

    Returns
    -------
    file_path : PosixPath
        Path to binary file.
    metadata_for_raw : Metadata object
        Metadata object with the set parameters.
    images : tuple of numpy arrays
        Each element of the tuple is the acquired stack for a different channel.
    """

    def _random_raw_file(
        area_mode=0,
        shape=(50, 60),
        timepoints=10,
        channels=("ChanA", "ChanB"),
        n_z=1,
        flyback_frames=0,
    ):
        if area_mode == 2 or area_mode == 3:
            new_n_y_pixels = shape[0] / n_z
            assert (
                not new_n_y_pixels % 1
            ), "Invalid test parameter. For kymograph and line the y values has to be divisible by the number of z steps."
            new_n_y_pixels = int(new_n_y_pixels)
            shape = (new_n_y_pixels, shape[1])

        img_stacks = np.zeros(
            (timepoints, n_z + flyback_frames, len(channels)) + shape, dtype=np.uint16
        )
        for t in range(timepoints):
            for i in range(len(channels)):
                img_stacks[t, :n_z, i] = random_stack((n_z,) + shape)
        metadata_for_raw = metadata_obj(
            timepoints=timepoints,
            x_pixels=shape[1],
            y_pixels=shape[0],
            channels=channels,
            n_z=n_z,
            area_mode=area_mode,
        )
        file_path = tmpdir.join("Image_0001_0001.raw")
        sequence = img_stacks.flatten()
        out = bytearray(len(sequence) * 2)
        struct.pack_into(f"<{len(sequence)}H", out, 0, *sequence)
        with open(file_path, "xb") as f:
            f.write(out)
        return (
            file_path,
            metadata_for_raw,
            tuple((np.squeeze(img_stacks[:, :n_z, i]) for i in range(len(channels)))),
        )

    return _random_raw_file


@pytest.fixture
def random_z_stack(tmpdir, random_stack, metadata_obj):
    """
    """

    def _random_z_stack(
        shape=(50, 60), timepoints=10, channels=("ChanA", "ChanB"), n_z=5
    ):
        z_stack = random_stack((timepoints, n_z, len(channels)) + shape)
        for c, channel in enumerate(channels):
            for step in range(n_z):
                for t in range(timepoints):
                    name = f"{channel}_0001_0001_{step + 1:04}_{t + 1:04}.tif"
                    path = tmpdir.join(name)
                    tifffile.imsave(str(path), z_stack[t, step, c])
        metadata = metadata_obj(
            y_pixels=shape[0],
            x_pixels=shape[1],
            timepoints=timepoints,
            channels=channels,
            n_z=n_z,
        )
        return tmpdir, metadata, z_stack

    return _random_z_stack


@pytest.mark.parametrize("timepoints", [0, 1, 10])
def test_get_n_time_points(metadata_obj, timepoints):
    assert metadata_obj(timepoints=timepoints).get_n_time_points() == timepoints


@pytest.mark.parametrize("x_pixels", [0, 1, 10])
def test_get_num_x_pixels(metadata_obj, x_pixels):
    assert metadata_obj(x_pixels=x_pixels).get_num_x_pixels() == x_pixels


@pytest.mark.parametrize("y_pixels", [0, 1, 10])
def test_get_num_y_pixels(metadata_obj, y_pixels):
    assert metadata_obj(y_pixels=y_pixels).get_num_y_pixels() == y_pixels


@pytest.mark.parametrize(
    "area_mode,result", [(0, "square"), (1, "rectangle"), (2, "kymograph"), (3, "line")]
)
def test_get_are_mode(metadata_obj, area_mode, result):
    assert metadata_obj(area_mode=area_mode).get_area_mode() == result


@pytest.mark.parametrize("n_z", [0, 1, 10])
def test_get_n_z(metadata_obj, n_z):
    assert metadata_obj(n_z=n_z).get_n_z() == n_z


@pytest.mark.parametrize(
    "channels,result", [(("ChanA", "ChanB"), 2), (("ChanA", "ChanB", "ChanC"), 3)]
)
def test_get_channels(metadata_obj, channels, result):
    assert metadata_obj(channels=channels).get_n_channels() == result


@pytest.mark.parametrize("channels", [("ChanA", "ChanB"), ("ChanA", "ChanC")])
def test_get_channels(metadata_obj, channels):
    assert metadata_obj(channels=channels).get_channels() == channels


@pytest.mark.parametrize("pixel_size", [0, 1, 10])
def test_get_pixel_size(metadata_obj, pixel_size):
    assert metadata_obj(pixel_size=pixel_size).get_pixel_size() == pixel_size


@pytest.mark.parametrize("z_step_size", [0, 1, 10])
def test_get_z_step_size(metadata_obj, z_step_size):
    assert metadata_obj(z_step_size=z_step_size).get_z_step_size() == z_step_size


@pytest.mark.parametrize(
    "area_mode,z_step_size,n_z,y_pixels,result",
    [(0, 1, 3, 4, 1), (1, 3, 1, 4, 3), (2, 0.5, 8, 2, 2), (3, 0.1, 200, 10, 2)],
)
def test_get_z_pixel_size(metadata_obj, area_mode, z_step_size, n_z, y_pixels, result):
    assert np.isclose(
        metadata_obj(
            area_mode=area_mode, z_step_size=z_step_size, n_z=n_z, y_pixels=y_pixels
        ).get_z_pixel_size(),
        result,
    )


@pytest.mark.parametrize("dwell_time", [0, 1, 10])
def test_get_dwell_time(metadata_obj, dwell_time):
    assert metadata_obj(dwell_time=dwell_time).get_dwell_time() == dwell_time


@pytest.mark.parametrize(
    "frame_rate,flyback_frames,n_z,result", [(5.5, 3, 2, 1.1), (8.3, 4, 1, 1.66)]
)
def test_get_frame_rate(metadata_obj, frame_rate, flyback_frames, n_z, result):
    assert np.isclose(
        metadata_obj(
            frame_rate=frame_rate, flyback_frames=flyback_frames, n_z=n_z
        ).get_frame_rate(),
        result,
    )


@pytest.mark.parametrize("width", [0, 1, 10])
def test_get_width(metadata_obj, width):
    assert metadata_obj(width=width).get_width() == width


@pytest.mark.parametrize("power", [0, 1, 10])
def test_get_power(metadata_obj, power):
    assert metadata_obj(power=power).get_power_reg1_start() == power


@pytest.mark.parametrize("gain_a", [0, 1, 10])
def test_get_gain_a(metadata_obj, gain_a):
    assert metadata_obj(gain_a=gain_a).get_gainA() == gain_a


@pytest.mark.parametrize("gain_b", [0, 1, 10])
def test_get_gain_b(metadata_obj, gain_b):
    assert metadata_obj(gain_b=gain_b).get_gainB() == gain_b


@pytest.mark.parametrize("date", ["12/20/2018 18:33:52", "05/04/2018 07:05:34"])
def test_get_date(metadata_obj, date):
    assert metadata_obj(date=date).get_date_time() == date


def test_load_img(random_tif_file):
    file_path, img_stack = random_tif_file()
    assert np.allclose(utils2p.load_img(file_path), img_stack)


@pytest.mark.parametrize(
    "area_mode,shape,timepoints,channels,n_z,flyback_frames",
    [
        (0, (50, 60), 1, ("ChanA",), 1, 0),
        (0, (50, 60), 1, ("ChanA",), 1, 3),
        (0, (50, 60), 1, ("ChanA",), 2, 0),
        (0, (50, 60), 1, ("ChanA",), 2, 3),
        (0, (50, 60), 1, ("ChanA", "ChanB"), 1, 0),
        (0, (50, 60), 1, ("ChanA", "ChanB"), 1, 3),
        (0, (50, 60), 1, ("ChanA", "ChanB"), 2, 0),
        (0, (50, 60), 1, ("ChanA", "ChanB"), 2, 3),
        (0, (50, 60), 8, ("ChanA",), 1, 0),
        (0, (50, 60), 8, ("ChanA",), 1, 3),
        (0, (50, 60), 8, ("ChanA",), 2, 0),
        (0, (50, 60), 8, ("ChanA",), 2, 3),
        (0, (50, 60), 8, ("ChanA", "ChanB"), 1, 0),
        (0, (50, 60), 8, ("ChanA", "ChanB"), 1, 3),
        (0, (50, 60), 8, ("ChanA", "ChanB"), 2, 0),
        (0, (50, 60), 8, ("ChanA", "ChanB"), 2, 3),
        (1, (50, 60), 1, ("ChanA",), 1, 0),
        (1, (50, 60), 1, ("ChanA",), 1, 3),
        (1, (50, 60), 1, ("ChanA",), 2, 0),
        (1, (50, 60), 1, ("ChanA",), 2, 3),
        (1, (50, 60), 1, ("ChanA", "ChanB"), 1, 0),
        (1, (50, 60), 1, ("ChanA", "ChanB"), 1, 3),
        (1, (50, 60), 1, ("ChanA", "ChanB"), 2, 0),
        (1, (50, 60), 1, ("ChanA", "ChanB"), 2, 3),
        (1, (50, 60), 8, ("ChanA",), 1, 0),
        (1, (50, 60), 8, ("ChanA",), 1, 3),
        (1, (50, 60), 8, ("ChanA",), 2, 0),
        (1, (50, 60), 8, ("ChanA",), 2, 3),
        (1, (50, 60), 8, ("ChanA", "ChanB"), 1, 0),
        (1, (50, 60), 8, ("ChanA", "ChanB"), 1, 3),
        (1, (50, 60), 8, ("ChanA", "ChanB"), 2, 0),
        (1, (50, 60), 8, ("ChanA", "ChanB"), 2, 3),
        (2, (50, 60), 1, ("ChanA",), 1, 0),
        (2, (50, 60), 1, ("ChanA",), 1, 3),
        (2, (50, 60), 1, ("ChanA",), 2, 0),
        (2, (50, 60), 1, ("ChanA",), 2, 3),
        (2, (50, 60), 1, ("ChanA", "ChanB"), 1, 0),
        (2, (50, 60), 1, ("ChanA", "ChanB"), 1, 3),
        (2, (50, 60), 1, ("ChanA", "ChanB"), 2, 0),
        (2, (50, 60), 1, ("ChanA", "ChanB"), 2, 3),
        (2, (50, 60), 8, ("ChanA",), 1, 0),
        (2, (50, 60), 8, ("ChanA",), 1, 3),
        (2, (50, 60), 8, ("ChanA",), 2, 0),
        (2, (50, 60), 8, ("ChanA",), 2, 3),
        (2, (50, 60), 8, ("ChanA", "ChanB"), 1, 0),
        (2, (50, 60), 8, ("ChanA", "ChanB"), 1, 3),
        (2, (50, 60), 8, ("ChanA", "ChanB"), 2, 0),
        (2, (50, 60), 8, ("ChanA", "ChanB"), 2, 3),
        (3, (50, 60), 1, ("ChanA",), 1, 0),
        (3, (50, 60), 1, ("ChanA",), 1, 3),
        (3, (50, 60), 1, ("ChanA",), 2, 0),
        (3, (50, 60), 1, ("ChanA",), 2, 3),
        (3, (50, 60), 1, ("ChanA", "ChanB"), 1, 0),
        (3, (50, 60), 1, ("ChanA", "ChanB"), 1, 3),
        (3, (50, 60), 1, ("ChanA", "ChanB"), 2, 0),
        (3, (50, 60), 1, ("ChanA", "ChanB"), 2, 3),
        (3, (50, 60), 8, ("ChanA",), 1, 0),
        (3, (50, 60), 8, ("ChanA",), 1, 3),
        (3, (50, 60), 8, ("ChanA",), 2, 0),
        (3, (50, 60), 8, ("ChanA",), 2, 3),
        (3, (50, 60), 8, ("ChanA", "ChanB"), 1, 0),
        (3, (50, 60), 8, ("ChanA", "ChanB"), 1, 3),
        (3, (50, 60), 8, ("ChanA", "ChanB"), 2, 0),
        (3, (50, 60), 8, ("ChanA", "ChanB"), 2, 3),
    ],
)
def test_load_raw(
    random_raw_file, area_mode, shape, timepoints, channels, n_z, flyback_frames
):
    file_path, metadata, img_stacks = random_raw_file(
        area_mode=area_mode,
        shape=shape,
        timepoints=timepoints,
        channels=channels,
        n_z=n_z,
        flyback_frames=flyback_frames,
    )
    loaded_stacks = utils2p.load_raw(file_path, metadata)
    for i in range(len(channels)):
        # For Kymograph recordings and Line recordings the z steps have to be concatenated along the vertical image axis.
        if (area_mode == 2 or area_mode == 3) and n_z > 1:
            if timepoints > 1:
                list_of_arrays = [
                    img_stacks[i][:, j] for j in range(img_stacks[i].shape[1])
                ]
            else:
                list_of_arrays = [
                    img_stacks[i][j] for j in range(img_stacks[i].shape[0])
                ]
            img_stack = np.concatenate(list_of_arrays, axis=-2)
        else:
            img_stack = img_stacks[i]
        assert np.allclose(
            loaded_stacks[i], img_stack
        ), f"Failed with parameters area_mode={area_mode}, shape={shape}, timepoints={timepoints}, channels={channels}, n_z={n_z}, flyback_frames={flyback_frames}."


def test_load_z_stack():
    metadata = utils2p.Metadata(data_dir / "mouse_kidney_z_stack/Experiment.xml")
    loaded_z_stack = utils2p.load_z_stack(data_dir / "mouse_kidney_z_stack", metadata)
    z_stack = np.load(data_dir / "mouse_kidney_z_stack/z_stack.npy")
    assert np.allclose(
        loaded_z_stack, z_stack
    ), "Failed to load mouse kidney z-stack correctly."

    metadata = utils2p.Metadata(
        data_dir / "mouse_kidney_time_series_z_stack/Experiment.xml"
    )
    loaded_z_stack = utils2p.load_z_stack(
        data_dir / "mouse_kidney_time_series_z_stack", metadata
    )
    z_stack = np.load(data_dir / "mouse_kidney_time_series_z_stack/z_stack.npy")
    assert np.allclose(
        loaded_z_stack, z_stack
    ), "Failed to load mouse kidney time of z-stacks correctly."


@pytest.mark.parametrize(
    "shape",
    [(2, 3, 4), (3, 4, 2, 4), (3, 4, 5, 3, 4), (5, 6, 7, 4, 5, 3), (4, 1, 3, 4)],
)
def test_concatenate_z(random_stack, shape):
    stack = random_stack(shape)
    concatenated = utils2p.concatenate_z(stack)
    assert concatenated.shape[-2] == stack.shape[-2] * stack.shape[-3]
    assert concatenated.ndim == stack.ndim - 1


@pytest.mark.parametrize(
    "shape",
    [(2, 3, 4), (3, 4, 2, 4), (3, 4, 5, 3, 4), (5, 6, 7, 4, 5, 1), (4, 1, 3, 4)],
)
def test_save_img(tmpdir, random_stack, shape):
    stack = random_stack(shape)
    utils2p.save_img(tmpdir / "stack.tif", stack)
    assert os.path.isfile(tmpdir / "stack.tif")
