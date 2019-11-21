"""
This module contains all the possible error raised by utils2p.
"""


class InputError(Exception):
    """This error should be raised if the input given to a function
    is not correct."""

    pass


class InvalidValueInMetaData(Exception):
    """This error should be raised when an invalid value
    is encountered in an 'Experiement.xml' file."""

    pass


class SynchronizationError(Exception):
    """The input data is not consistent with synchronization assumption."""

    pass
