#!/usr/bin/env python3
"""
utils.py

Some Python utilities.
"""

import errno
import numpy as np
import os

from decimal import Decimal


def quant_str(number, precision="0.1"):
    """Correctly round a number to a given number of decimal places."""
    return str(Decimal(number).quantize(Decimal(precision)))


def ensure_dir(path):
    """Ensure the string is a directory, with a slash as its last
        character."""
    if path != '':
        if path[-1] != '/':
            path += '/'
    return path


def ensure_exists(path):
    """Ensure the path exists; if not, make the directory."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_home_dir():
    """Specify the location of my home directory, just convenient.

    Note:
        Usage is as follows:
        from sys import platform as _platform
        home_dir = get_home_dir(_platform)
    """
    from sys import platform as _platform

    if _platform == "darwin":  # OS X
        home_dir = "/Users/koma/"

    elif _platform == "linux" or _platform == "linux2":  # Linux
        home_dir = "/home/koma/"

    else:
        raise ValueError("Platform unrecognized?")

    return home_dir


def find_nearest(array, value, forcefloor=False):
    """Return the index of ARRAY closest to VALUE."""
    idx = (np.abs(array - value)).argmin()
    if forcefloor:
        if array[idx] > value:
            idx -= 1
    return idx


def norm(array):
    """Normalize array. Might make more complex in the future."""
    return array / np.nanmax


def smooth(x, window_len=11, window='hanning'):
    """Smooth a 1-D array, from scipy? TODO: Find reference!"""
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector must be bigger than window size.")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', \
            'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[2 * x[0] - x[window_len - 1::-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')

    return y[window_len:-window_len + 1]


def to_fwhm(sigma):
    """Convert sigma to FWHM."""
    constant = 2 * np.sqrt(2 * np.log(2))
    return constant * sigma


def to_sigma(fwhm):
    """Convert FWHM to sigma."""
    constant = 2 * np.sqrt(2 * np.log(2))
    return fwhm / constant


def verify_dict_equality(dict1, dict2):
    """Ensure that the dictionary is unchnged after writing/reading.

    Args:
        dict1 (dict): Dictioanry 1.
        dict2 (dict): Dictionary 2.

    Returns:
        True if succesful.
    """

    # Test for equality.
    if set(dict1) == set(dict2):
        print('...verified.')
    else:
        print(dict1.keys())
        print(dict2.keys())
        raise KeyError("Dictionary error, integrity compromised.")

    return True


def verify_dataframe_equality(df1, df2):
    """Ensure that the dataframe is unchnged after writing/reading.

    Args:
        df1 (pd.DataFrame): Dataframe 1.
        df1 (pd.DataFrame): Dataframe 2.

    Returns:
        True if succesful.
    """
    if df1.equals(df2):
        print('...verified.')
    else:
        print(df1.shape)
        print(df2.shape)
        print("Dataframes not equal.")
        return False

    return True


def main():

    print('test')
    # filedir = 'spectra/run_lorentz/'
    # mydict = convert_txt_to_dict(filedir)
    # json_path = filedir + 'spectra_dict.json'

    # # Try writing it to disk.
    # write_dict_to_disk(json_path, mydict)

    # # Try reading it in again.
    # adict = read_dict_from_disk(json_path)

    # # Test for equality.
    # if set(mydict) == set(adict):
    #     print('JSON integrity verified.')
    # else:
    #     raise KeyError("JSON error, dictionary integrity compromised.")


if __name__ == '__main__':
    main()
