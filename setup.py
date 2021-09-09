from setuptools import setup, find_packages
from distutils.core import Extension

try:
    import numpy as np
except ImportError or ModuleNotFoundError:
    raise ImportError(
        "Numpy has to be installed in order to install utils2p.\n"
        + "Please refer to the Numpy website for instructions on how to install it.\n"
    )


with open("README.rst", "r") as fh:
    long_description = fh.read()


external = Extension(
    "utils2p.external.tifffile._tifffile",
    sources=["utils2p/external/tifffile/tifffile.c"],
    include_dirs=[np.get_include()],
)


setup(
    name="utils2p",
    version="0.1.0",
    packages=["utils2p", "utils2p.external", "utils2p.external.tifffile"],
    author="Florian Aymanns",
    author_email="florian.ayamnns@epfl.ch",
    description="Basic utility functions for 2 photon image data generated using ThorImage and ThorSync.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/NeLy-EPFL/utils2p",
    setup_requires=["numpy"],
    install_requires=["numpy",
                      "pytest",
                      "scipy",
                      "h5py",
                     ],
    ext_modules=[external],
)
