#!/usr/bin/env python3
"""
test_utils.py

Test whether utils.py behaves as expected."""


import pandas as pd

from mattpy import utils


def test_ensure_dir():
    assert utils.ensure_dir('') == ''
    assert utils.ensure_dir('test') == 'test/'
    assert utils.ensure_dir('test/') == 'test/'
    return


def test_verify_dict_equality():
    """Test that a dictionary is unchnged after JSON write."""
    d1 = {}
    d2 = {}
    assert utils.verify_dict_equality(d1, d2)

    d1 = {'test': 1}
    d2 = {'test': 1}
    assert utils.verify_dict_equality(d1, d2)

    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 2, 'a': 1}
    assert utils.verify_dict_equality(d1, d2)


def test_verify_dataframe_equality():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    assert utils.verify_dataframe_equality(df1, df2)

    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    assert not utils.verify_dataframe_equality(df, df2)

    return
