#!/usr/bin/env python3
"""
test_data_io.py

Test whether data_io.py behaves as expected."""


import os.path
import pandas as pd

from mattpy import data_io


def test_ensure_dir():
    assert data_io.ensure_dir('') == ''
    assert data_io.ensure_dir('test') == 'test/'
    assert data_io.ensure_dir('test/') == 'test/'
    return


def test_verify_dict_equality():
    """Test that a dictionary is unchnged after JSON write."""
    d1 = {}
    d2 = {}
    assert data_io.verify_dict_equality(d1, d2)

    d1 = {'test': 1}
    d2 = {'test': 1}
    assert data_io.verify_dict_equality(d1, d2)

    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 2, 'a': 1}
    assert data_io.verify_dict_equality(d1, d2)


def test_verify_dataframe_equality():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    assert data_io.verify_dataframe_equality(df1, df2)

    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    assert not data_io.verify_dataframe_equality(df, df2)

    return


def test_parse_one_spectrum_file():
    """Test that a single PAHdb spectrum can be parsed."""
    mock_fname = 'mattpy/tests/data/spectra/spectra_beta_2.5_ionfrac_0.45.txt'
    dict1 = data_io.parse_one_spectrum_file(mock_fname)
    assert dict1['beta'] == 2.5
    assert dict1['ionfrac'] == 0.45
    assert len(dict1['flux']) == 400


def test_convert_txt_to_dict():
    """Test that spectra can be read in and returned as a dict
        (with tuple keys)."""
    fdir = 'mattpy/tests/data/spectra/'
    dict1 = data_io.convert_txt_to_dict(fdir, search_str='spectra*.txt')
    assert list(dict1.keys()) == [(2.5, 0.45), (3.0, 0.05)]

    dict1 = data_io.convert_txt_to_dict(fdir, search_str='random*.txt')
    assert dict1 == {}


def write_dataframe_to_pickle():
    """Test writing a dataframe to pickle."""
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    fname = 'mattpy/tests/data/test_folder/df.pkl'
    data_io.write_dataframe_to_pickle(fname, df)
    assert os.path.isfile(fname)


def read_dataframe_from_pickle():
    return


def test_write_dict_to_json():
    """Test that a dict (with tuple keys) can be written to disk
        as JSON."""
    dict1 = {(0.5, 0.5): [1, 2, 3, 4, 5]}
    fname = 'mattpy/tests/data/test_folder/test_write.json'
    data_io.write_dict_to_json(fname, dict1)
    assert os.path.isfile(fname)


def test_read_dict_from_json():
    """Test that the JSON (tuple key) dict can be read back in."""
    dict1 = {(0.5, 0.5): [1, 2, 3, 4, 5]}
    fname = 'mattpy/tests/data/test_folder/test_write.json'
    dict2 = data_io.read_dict_from_json(fname)
    assert dict1 == dict2


def test_dump_all_to_disk():
    """Test that we can go straight from text files to the JSON."""
    fdir = 'mattpy/tests/data/'
    test = data_io.dump_all_to_disk(fdir, search_str='spectra*.txt',
                                    save_file='spectra_dict.json',
                                    save_dir=fdir + 'test_folder',
                                    method='json',
                                    verify=True)
    assert test
