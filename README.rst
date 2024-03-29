utils2p
=======

.. image:: https://travis-ci.com/NeLy-EPFL/utils2p.svg?token=snip9q1Tczja5zRZ7RGJ&branch=master
.. image:: https://codecov.io/gh/NeLy-EPFL/utils2p/branch/master/graph/badge.svg?token=Y5TJHHYYFT
  :target: https://codecov.io/gh/NeLy-EPFL/utils2p
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5501119.svg
  :target: https://doi.org/10.5281/zenodo.5501119

.. contents Topics

Overview
--------
This module provides utility functions necessary for loading and processing
2-photon imaging data acquired with ThorLabs microscopes. It includes function
to read files generated by ThorImage and ThorSync.

Documentation
-------------
The full documentation is hosted on `github pages <https://nely-epfl.github.io/utils2p/>`_.

Installation
------------
Note that you will need an installation of `numpy <https://numpy.org/>`_ to install this package.
You can install numpy with the following command:

.. code-block:: shell

    pip install numpy

The package can be installed using `pip <https://pypi.org/project/pip/>`_ with:

.. code-block:: shell

    pip install utils2p

If you need the latest version or you want to make changes to the code, you should install
utils2p from source by following the steps below.

To do so first create a copy of the source code on your machine by cloning the repository:

.. code-block:: shell

    git clone https://github.com/NeLy-EPFL/utils2p.git

Then install it using the following command:

.. code-block:: shell

    pip install -e utils2p

Dependencies
------------
- python >3.6
- `pip <https://pypi.org/project/pip/>`_
- `numpy <https://numpy.org/>`_
- `h5py <https://www.h5py.org/>`_
- `scipy <https://scipy.org/>`_
