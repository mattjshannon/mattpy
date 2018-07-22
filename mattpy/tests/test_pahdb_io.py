#!/usr/bin/env python3
"""
test_pahdb_io.py

Test whether pahdb_io.py behaves as expected."""


from mattpy import pahdb_io


def test_convert_folder_cont_fit():
    """Test that we can convert the "cont_fit" folder contents
        into JSON properly."""
    fdir = 'mattpy/tests/data/'
    json_dir = fdir + 'json_test_folder/'

    assert pahdb_io.convert_folder_cont_fit(fdir, sub_dir=None,
                                            json_dir=json_dir)

    assert pahdb_io.convert_folder_cont_fit(fdir, sub_dir='run_lorentz',
                                            json_dir=json_dir)


def test_convert_folder_spectra():
    """Test that we can convert the "cont_fit" folder contents
        into JSON properly."""
    fdir = 'mattpy/tests/data/'
    json_dir = fdir + 'json_test_folder/'

    assert pahdb_io.convert_folder_spectra(fdir, sub_dir=None,
                                           json_dir=json_dir)

    assert pahdb_io.convert_folder_spectra(fdir, sub_dir='run_lorentz',
                                           json_dir=json_dir)
