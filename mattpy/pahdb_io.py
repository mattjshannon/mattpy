#!/usr/bin/env python3
"""
pahdb_io.py

Some handy input/output operations on PAHdb "run" folders.
"""

import numpy as np

from . import json_io


def convert_folder_cont_fit(base_dir, sub_dir=None):
    """Convert the continuum fit data into three JSON files.

    Args:
        base_dir (str): Directory of the "run".
        sub_dir (str): 'run_lorentz' or 'run_gauss', depending on the run.

    Returns:
        True if succesful.
    """

    working_dir = base_dir + 'cont_fit/'

    # add subdirectory.
    if sub_dir:
        working_dir += sub_dir + '/'

    # Collate the spline vectors.
    json_io.dump_all_to_json(working_dir,
                             search_str='contflux_*.txt',
                             json_file='contflux_dict.json',
                             verify=True)

    # Collate the spline knots.
    json_io.dump_all_to_json(working_dir,
                             search_str='contknots_*.txt',
                             json_file='contknots_dict.json',
                             verify=True)

    # Collate the 7.7-micron region cut-outs.
    json_io.dump_all_to_json(working_dir,
                             search_str='cont77_*.txt',
                             json_file='cont77_dict.json',
                             verify=True)

    return True


def convert_folder_spectra(base_dir, sub_dir=None):
    """Convert the PAHdb spectra to JSON.

    Args:
        base_dir (str): Directory of the "run".
        sub_dir (str): 'run_lorentz' or 'run_gauss', depending on the run.

    Returns:
        True if succesful.
    """

    working_dir = base_dir + 'spectra/'

    # add subdirectory.
    if sub_dir:
        working_dir += sub_dir + '/'

    # Collate the spectra..
    json_io.dump_all_to_json(working_dir,
                             search_str='spectra_*.txt',
                             json_file='spectra_dict.json',
                             verify=True)

    return True


def main():

    print('test')


if __name__ == '__main__':
    main()
