#!/usr/bin/env python3
"""
test_pahdb_io.py

Test whether pahdb_io.py behaves as expected."""


from mattpy import pahdb_io


def test_convert_folder_cont_fit():
    """Test that we can convert the "cont_fit" folder contents
        into pickles properly."""
    fdir = 'mattpy/tests/data/'
    save_dir = fdir + 'test_folder/'

    assert pahdb_io.convert_folder_cont_fit(fdir, sub_dir=None,
                                            save_dir=save_dir,
                                            method='pickle')

    assert pahdb_io.convert_folder_cont_fit(fdir, sub_dir='run_lorentz',
                                            save_dir=save_dir,
                                            method='pickle')


def test_convert_folder_spectra():
    """Test that we can convert the "cont_fit" folder contents
        into pickles properly."""
    fdir = 'mattpy/tests/data/'
    save_dir = fdir + 'test_folder/'

    assert pahdb_io.convert_folder_spectra(fdir, sub_dir=None,
                                           save_dir=save_dir,
                                           method='pickle')

    assert pahdb_io.convert_folder_spectra(fdir, sub_dir='run_lorentz',
                                           save_dir=save_dir,
                                           method='pickle')
