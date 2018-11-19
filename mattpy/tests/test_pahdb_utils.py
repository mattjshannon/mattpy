#!/usr/bin/env python3
"""
test_pahdb_utils.py

Test whether pahdb_utils.py behaves as expected."""


from mattpy import pahdb_utils


def test_convert_folder_cont_fit():
    """Test that we can convert the "cont_fit" folder contents
        into pickles properly."""
    fdir = 'mattpy/tests/data/'
    save_dir = fdir + 'test_folder/'

    assert pahdb_utils.convert_folder_cont_fit(fdir, sub_dir=None,
                                               save_dir=save_dir,
                                               method='pickle')

    assert pahdb_utils.convert_folder_cont_fit(fdir, sub_dir='run_lorentz',
                                               save_dir=save_dir,
                                               method='pickle')


def test_convert_folder_spectra():
    """Test that we can convert the "cont_fit" folder contents
        into pickles properly."""
    fdir = 'mattpy/tests/data/'
    save_dir = fdir + 'test_folder/'

    assert pahdb_utils.convert_folder_spectra(fdir, sub_dir=None,
                                              save_dir=save_dir,
                                              method='pickle')

    assert pahdb_utils.convert_folder_spectra(fdir, sub_dir='run_lorentz',
                                              save_dir=save_dir,
                                              method='pickle')


def test_parse_one_spectrum_file():
    """Test that a single PAHdb spectrum can be parsed."""
    mock_fname = 'mattpy/tests/data/spectra/spectra_beta_2.5_ionfrac_0.45.txt'
    dict1 = pahdb_utils.parse_one_spectrum_file(mock_fname)
    assert dict1['beta'] == 2.5
    assert dict1['ionfrac'] == 0.45
    assert len(dict1['flux']) == 400


def test_convert_txt_to_dict():
    """Test that spectra can be read in and returned as a dict
        (with tuple keys)."""
    fdir = 'mattpy/tests/data/spectra/'
    dict1 = pahdb_utils.convert_txt_to_dict(fdir, search_str='spectra*.txt')
    assert list(dict1.keys()) == [(2.5, 0.45), (3.0, 0.05)]

    dict1 = pahdb_utils.convert_txt_to_dict(fdir, search_str='random*.txt')
    assert dict1 == {}


def test_dump_all_to_disk():
    """Test that we can go straight from text files to the JSON."""
    fdir = 'mattpy/tests/data/'
    test = pahdb_utils.dump_all_to_disk(fdir, search_str='spectra*.txt',
                                        save_file='spectra_dict.json',
                                        save_dir=fdir + 'test_folder',
                                        method='json',
                                        verify=True)
    assert test
