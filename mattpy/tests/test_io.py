#!/usr/bin/env python3
"""
test_io.py

Test whether io.py behaves as expected."""


import os.path
import pandas as pd

from mattpy import io


def test_ensure_dir():
    assert io.ensure_dir('') == ''
    assert io.ensure_dir('test') == 'test/'
    assert io.ensure_dir('test/') == 'test/'
    return


def test_verify_dict_equality():
    """Test that a dictionary is unchnged after JSON write."""
    d1 = {}
    d2 = {}
    assert io.verify_dict_equality(d1, d2)

    d1 = {'test': 1}
    d2 = {'test': 1}
    assert io.verify_dict_equality(d1, d2)

    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 2, 'a': 1}
    assert io.verify_dict_equality(d1, d2)


def test_verify_dataframe_equality():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    assert io.verify_dataframe_equality(df1, df2)

    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    assert not io.verify_dataframe_equality(df, df2)

    return


def write_dataframe_to_pickle():
    """Test writing a dataframe to pickle."""
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    fname = 'mattpy/tests/data/test_folder/df.pkl'
    io.write_dataframe_to_pickle(fname, df)
    assert os.path.isfile(fname)


def read_dataframe_from_pickle():
    return


def test_write_dict_to_json():
    """Test that a dict (with tuple keys) can be written to disk
        as JSON."""
    dict1 = {(0.5, 0.5): [1, 2, 3, 4, 5]}
    fname = 'mattpy/tests/data/test_folder/test_write.json'
    io.write_dict_to_json(fname, dict1)
    assert os.path.isfile(fname)


def test_read_dict_from_json():
    """Test that the JSON (tuple key) dict can be read back in."""
    dict1 = {(0.5, 0.5): [1, 2, 3, 4, 5]}
    fname = 'mattpy/tests/data/test_folder/test_write.json'
    dict2 = io.read_dict_from_json(fname)
    assert dict1 == dict2
