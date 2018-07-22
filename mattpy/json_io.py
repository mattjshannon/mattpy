#!/usr/bin/env python3
"""
tools.py

Some packaging tools for manipulating PAHdb run results.
"""

import glob
import json
import numpy as np
import pandas as pd

from ast import literal_eval
# from ipdb import set_trace as st


def ensure_dir(path):
    """Ensure the string is a directory, with a slash as its last
        character."""
    if path != '':
        if path[-1] != '/':
            path += '/'
    return path


def verify_dict_equality(dict1, dict2):
    """Ensure that the dictionary is unchnged after JSON write.

    Args:
        dict1 (dict): Dictioanry 1.
        dict2 (dict): Dictionary 2.

    Returns:
        None if succesful.
    """

    # Test for equality.
    if set(dict1) == set(dict2):
        print('JSON integrity verified.')
        print()
    else:
        print(dict1.keys())
        print(dict2.keys())
        raise KeyError("JSON error, dictionary integrity compromised.")

    return True


def parse_one_spectrum_file(fname):
    """Read a single PAHdb spectrum, return 'beta' and 'ionfrac',
    as well as the spectrum itself.

    Args:
        fname (str): File to parse.

    Returns:
        spec_dict (dict): Containing flux array, beta, ionfrac.
    """

    def parse_filename(fname):
        """Extract beta and ionfrac values from filename."""

        beta = fname.split('_beta_')[1].split('_')[0]
        ionfrac = fname.split('_ionfrac_')[-1].split('.txt')[0]

        return float(beta), float(ionfrac)

    beta, ionfrac = parse_filename(fname)

    try:
        dataframe = pd.read_csv(fname, sep=',', names=['flux', 'na'])
    except Exception as error:
        print(error)

    # Remove trivial column that arises from the csv format.
    del dataframe['na']

    spec_dict = dataframe.to_dict()
    spec_dict['beta'] = beta
    spec_dict['ionfrac'] = ionfrac

    return spec_dict


def convert_txt_to_dict(file_dir, search_str='spectra*.txt'):
    """Import all spectra*.txt files, combine them into a dictionary.

    Args:
        file_dir (str): location of the PAHdb spectra.

    Note:
        Utilizes pandas DataFrame for reading.
        Dataframe should contain columns for 'beta' and 'ionfrac'.
        FULL_DICT has keys in form (beta, ionfrac) as floats.
    """

    # Data to be combined.
    try:
        glob_files = glob.glob(ensure_dir(file_dir) + search_str)
    except IOError as error:
        print(error)

    spectra_files = np.sort(glob_files)

    # Dictionary for holding results.
    full_dict = {}

    # Iterate over all the spectra.
    for index, value in enumerate(spectra_files):

        # Parse the single file into a temporary dataframe.
        tmp_dataframe = parse_one_spectrum_file(value)

        # Create a new key, test that it's unique.
        new_key = (tmp_dataframe['beta'], tmp_dataframe['ionfrac'])
        if new_key in full_dict.keys():
            raise ValueError("key already defined!")
        else:
            full_dict[new_key] = tmp_dataframe['flux']

    return full_dict


def write_dict_to_disk(fname, the_dict):
    """Use JSON to write dictionary to disk.

    Args:
        fname (str): Filename to save dictionary to.
        the_dict (dict): Should have tuple (beta, ionfrac) keys, using
            floats.
    """

    # with open(fname, 'w') as f:
    #     json.dump(the_dict, f)

    try:
        with open(fname, 'w') as f:
            json.dump({str(k): v for k, v in the_dict.items()}, f)
    except IOError as error:
        print(error)

    print('Wrote dictionary to disk: ', fname)

    return


def read_dict_from_disk(fname):
    """Use JSON to read dictionary from disk.

    Args:
        fname (str): Filename containing the dictionary.

    Returns:
        the_dict (dict): Should have tuple (beta, ionfrac) keys, using
            floats.
    """

    # with open(fname) as f:
    #     the_dict = json.load(f)

    # load in two stages:#
    # (i) load json object
    try:
        with open(fname, 'r') as f:
            obj = json.load(f)
    except IOError as error:
        print(error)

    # (ii) convert loaded keys from string back to tuple
    the_dict = {literal_eval(k): v for k, v in obj.items()}

    print('Read dictionary from disk: ', fname)

    return the_dict


def dump_all_to_json(file_dir, search_str='spectra*.txt',
                     json_file='spectra_dict.json',
                     json_dir=None,
                     verify=True):
    """Shorthand for converting all .txt PAHdb files to JSON.

    Args:
        file_dir (str): Directory containg 'spectra*.txt'.
        search_str (str): Glob search string for files to include.
        json_fname (str): JSON filename.
        json_dir (str): Desired JSON output directory. If None,
            will default to filedir.

    Note:
        Places resulting JSON files in the same directory as the .txt
            PAHdb files.
    """

    file_dir = ensure_dir(file_dir)
    mydict = convert_txt_to_dict(file_dir, search_str)

    if json_dir:
        json_path = ensure_dir(json_dir) + json_file
    else:
        json_path = ensure_dir(file_dir) + json_file

    # Try writing it to disk.
    write_dict_to_disk(json_path, mydict)

    if verify:
        verify_dict_equality(mydict, read_dict_from_disk(json_path))

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
