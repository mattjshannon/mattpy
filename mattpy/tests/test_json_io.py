#!/usr/bin/env python3
"""
test_json_io.py

Test whether json_io.py behaves as expected."""


import os.path

from mattpy import json_io


def test_verify_dict_equality():
    """Test that a dictionary is unchnged after JSON write."""
    d1 = {}
    d2 = {}
    assert json_io.verify_dict_equality(d1, d2)

    d1 = {'test': 1}
    d2 = {'test': 1}
    assert json_io.verify_dict_equality(d1, d2)

    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 2, 'a': 1}
    assert json_io.verify_dict_equality(d1, d2)


def test_parse_one_spectrum_file():
    """Test that a single PAHdb spectrum can be parsed."""
    mock_fname = 'mattpy/tests/data/spectra_beta_2.5_ionfrac_0.45.txt'
    dict1 = json_io.parse_one_spectrum_file(mock_fname)
    assert dict1['beta'] == 2.5
    assert dict1['ionfrac'] == 0.45
    assert len(dict1['flux']) == 400


def test_convert_txt_to_dict():
    """Test that spectra can be read in and returned as a dict
        (with tuple keys)."""
    fdir = 'mattpy/tests/data/'
    dict1 = json_io.convert_txt_to_dict(fdir, search_str='spectra*.txt')
    assert list(dict1.keys()) == [(2.5, 0.45), (3.0, 0.05)]

    dict1 = json_io.convert_txt_to_dict(fdir, search_str='random*.txt')
    assert dict1 == {}


def test_write_dict_to_disk():
    """Test that a dict (with tuple keys) can be written to disk
        as JSON."""
    dict1 = {(0.5, 0.5): [1, 2, 3, 4, 5]}
    fname = 'mattpy/tests/data/test_write.json'
    json_io.write_dict_to_disk(fname, dict1)
    assert os.path.isfile(fname)


def test_read_dict_from_disk():
    """Test that the JSON (tuple key) dict can be read back in."""
    dict1 = {(0.5, 0.5): [1, 2, 3, 4, 5]}
    fname = 'mattpy/tests/data/test_write.json'
    dict2 = json_io.read_dict_from_disk(fname)
    assert dict1 == dict2


def test_dump_all_to_json():
    """Test that we can go straight from text files to the JSON."""
    fdir = 'mattpy/tests/data/'
    test = json_io.dump_all_to_json(fdir, search_str='spectra*.txt',
                                    json_file='spectra_dict.json',
                                    verify=True)
    assert test
