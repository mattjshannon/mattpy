#!/usr/bin/env python3
"""
data_io.py

Some packaging tools for manipulating PAHdb run results.
"""

import glob
import json
import numpy as np
import pandas as pd

from ast import literal_eval


def ensure_dir(path):
    """Ensure the string is a directory, with a slash as its last
        character."""
    if path != '':
        if path[-1] != '/':
            path += '/'
    return path


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

    # Read the file as a pandas dataframe.
    try:
        dataframe = pd.read_csv(fname, sep=',', names=['flux', 'na'])
    except Exception as error:
        raise(error)

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
        raise(error)

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


def dump_all_to_disk(file_dir, search_str='spectra*.txt',
                     save_file='spectra_dict.json',
                     save_dir=None, method='pickle',
                     verify=True):
    """Shorthand for converting all .txt PAHdb files to JSON or df pickle.

    Args:
        file_dir (str): Directory containg 'spectra*.txt'.
        search_str (str): Glob search string for files to include.
        json_fname (str): JSON filename.
        json_dir (str): Desired JSON output directory. If None,
            will default to filedir.

    Note:
        Places resulting JSON files in the same directory as the .txt
            PAHdb files.
        Method can be either 'pickle' or 'json'.
    """

    file_dir = ensure_dir(file_dir)
    mydict = convert_txt_to_dict(file_dir, search_str)

    # Determine location for output.
    if save_dir:
        save_path = ensure_dir(save_dir) + save_file
    else:
        save_path = ensure_dir(file_dir) + save_file

    # Write to disk. If pickling, convert to dataframe first.
    if method == 'pickle':
        df = pd.DataFrame.from_dict(mydict)
        write_dataframe_to_pickle(save_path, df)
    elif method == 'json':
        write_dict_to_json(save_path, mydict)
    else:
        raise ValueError("Unknown method for saving to disk.")

    # Make sure the write/read operations leave the object unchanged.
    if verify:
        if method == 'pickle':
            # Test for dataframe equality.
            eq = verify_dataframe_equality(df, pd.read_pickle(save_path))
            if not eq:
                raise ValueError('Dataframes not equal.')
        elif method == 'json':
            # Test for dictionary equality.
            eq = verify_dict_equality(mydict, read_dict_from_json(save_path))
            if not eq:
                raise ValueError('Dicts not equal.')

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
