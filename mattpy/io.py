#!/usr/bin/env python3
"""
io.py

Some packaging tools for manipulating PAHdb run results.
"""

import json
import pandas as pd

from ast import literal_eval


def write_dataframe_to_pickle(fname, dataframe):
    """Use pickle to write dataframe to disk.

    Args:
        fname (str): Filename to save dataframe to.
        dataframe (pd.DataFrame): Containing flux, beta, ionfrac.

    Returns:
        True if successful.
    """

    try:
        dataframe.to_pickle(fname)
    except Exception as error:
        raise error

    print('Wrote dataframe to pickle: ', fname)

    return True


def read_dataframe_from_pickle(fname):
    """Read a dataframe from a pickle.

    Args:
        fname (str): Filename of the pickle.

    Returns:
        dataframe (pd.DataFrame): Containing flux, beta, ionfrac.
    """

    try:
        dataframe = pd.read_pickle(fname)
    except Exception as error:
        raise error

    print('Read dataframe from pickle: ', fname)

    return dataframe


def write_dict_to_json(fname, the_dict):
    """Write dictionary to JSON.

    Args:
        fname (str): Filename to save dictionary to.
        the_dict (dict): Should have tuple (beta, ionfrac) keys, using
            floats.
    """

    try:
        with open(fname, 'w') as f:
            json.dump({str(k): v for k, v in the_dict.items()}, f)
    except Exception as error:
        raise error

    print('Wrote dictionary to disk: ', fname)

    return


def read_dict_from_json(fname):
    """Read dictionary from JSON.

    Args:
        fname (str): Filename containing the dictionary.

    Returns:
        the_dict (dict): Should have tuple (beta, ionfrac) keys, using
            floats.
    """

    # Load JSON object.
    try:
        with open(fname, 'r') as f:
            obj = json.load(f)
    except IOError as error:
        raise error

    # Convert the keys back to tuples.
    the_dict = {literal_eval(k): v for k, v in obj.items()}

    print('Read dictionary from disk: ', fname)

    return the_dict


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
