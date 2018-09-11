#!/usr/bin/env python3
"""
data_io.py

Some packaging tools for manipulating PAHdb run results.
"""

import errno
import json
import os
import pandas as pd

from ast import literal_eval


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
        raise(error)

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
        raise(error)

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
        raise(error)

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
        raise(error)

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
