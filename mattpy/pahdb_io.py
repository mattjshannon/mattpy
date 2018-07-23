#!/usr/bin/env python3
"""
pahdb_io.py

Some handy input/output operations on PAHdb "run" folders.
"""

from . import data_io


def convert_folder_cont_fit(base_dir, sub_dir=None, save_dir=None,
                            method='pickle'):
    """Convert the continuum fit data into three files.

    Args:
        base_dir (str): Directory of the "run".
        sub_dir (str): 'run_lorentz' or 'run_gauss', run dependent.
        save_dir (str): Save file directory if not None.
        method (str): Whether to use pickles or JSON.

    Returns:
        True if successful.
    """

    working_dir = data_io.ensure_dir(base_dir) + 'cont_fit/'

    if method == 'pickle':
        sfx = '.pkl'
    elif method == 'json':
        sfx = '.json'

    # add subdirectory.
    if sub_dir:
        working_dir += sub_dir
        working_dir = data_io.ensure_dir(working_dir)
        # Identify output files with (e.g.,) '_run_lorentz'.
        save_file_contflux = 'pickles_' + sub_dir + '_contflux' + sfx
        save_file_contknots = 'pickles_' + sub_dir + '_contknots' + sfx
        save_file_cont77 = 'pickles_' + sub_dir + '_cont77' + sfx
    else:
        save_file_contflux = 'pickles_run_unknown_contflux' + sfx
        save_file_contknots = 'pickles_run_unknown_contknots' + sfx
        save_file_cont77 = 'pickles_run_unknown_cont77' + sfx

    # Collate the spline vectors.
    data_io.dump_all_to_disk(working_dir,
                             search_str='contflux_*.txt',
                             save_file=save_file_contflux,
                             save_dir=save_dir, method=method,
                             verify=True)

    # Collate the spline knots.
    data_io.dump_all_to_disk(working_dir,
                             search_str='contknots_*.txt',
                             save_file=save_file_contknots,
                             save_dir=save_dir, method=method,
                             verify=True)

    # Collate the 7.7-micron region cut-outs.
    data_io.dump_all_to_disk(working_dir,
                             search_str='cont77_*.txt',
                             save_file=save_file_cont77,
                             save_dir=save_dir, method=method,
                             verify=True)

    return True


def convert_folder_spectra(base_dir, sub_dir=None, save_dir=None,
                           method='pickle'):
    """Convert the PAHdb spectra to JSON/pickle.

    Args:
        base_dir (str): Directory of the "run".
        sub_dir (str): 'run_lorentz' or 'run_gauss', run dependent.
        save_dir (str): Save file directory if not None.
        method (str): Whether to use pickles or JSON.

    Returns:
        True if successful.
    """

    working_dir = data_io.ensure_dir(base_dir) + 'spectra/'

    if method == 'pickle':
        sfx = '.pkl'
    elif method == 'json':
        sfx = '.json'

    # add subdirectory.
    if sub_dir:
        working_dir += sub_dir
        working_dir = data_io.ensure_dir(working_dir)
        save_file_spectra = 'pickles_' + sub_dir + '_spectra' + sfx
    else:
        save_file_spectra = 'pickles_run_unknown_spectra' + sfx

    # Collate the spectra..
    data_io.dump_all_to_disk(working_dir,
                             search_str='spectra_*.txt',
                             save_file=save_file_spectra,
                             save_dir=save_dir, method=method,
                             verify=True)

    return True


def main():

    print('test')


if __name__ == '__main__':
    main()
