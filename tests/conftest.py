import pytest

import utils2p

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
