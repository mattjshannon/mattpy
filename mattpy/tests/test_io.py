#!/usr/bin/env python3
"""
test_io.py

Test whether io.py behaves as expected."""


import os.path
import pandas as pd

from mattpy import io


def test_write_dataframe_to_pickle():
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
