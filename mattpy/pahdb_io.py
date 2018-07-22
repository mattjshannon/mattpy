#!/usr/bin/env python3
"""
pahdb_io.py

Some handy input/output operations on PAHdb "run" folders.
"""

from . import json_io


def convert_folder_cont_fit(base_dir, sub_dir=None, json_dir=None):
    """Convert the continuum fit data into three JSON files.

    Args:
        base_dir (str): Directory of the "run".
        sub_dir (str): 'run_lorentz' or 'run_gauss', run dependent.
        json_dir (str): Override for json file output directory if set.

    Returns:
        True if succesful.
    """

    working_dir = json_io.ensure_dir(base_dir) + 'cont_fit/'

    # add subdirectory.
    if sub_dir:
        working_dir += sub_dir
        working_dir = json_io.ensure_dir(working_dir)

        # Identify json files with (e.g.,) '_run_lorentz'.
        json_file_contflux = 'contflux_dict_' + sub_dir + '.json'
        json_file_contknots = 'contknots_dict_' + sub_dir + '.json'
        json_file_cont77 = 'cont77_dict_' + sub_dir + '.json'
    else:
        json_file_contflux = 'contflux_dict_run.json'
        json_file_contknots = 'contknots_dict_run.json'
        json_file_cont77 = 'cont77_dict_run.json'

    # Collate the spline vectors.
    json_io.dump_all_to_json(working_dir,
                             search_str='contflux_*.txt',
                             json_file=json_file_contflux,
                             json_dir=json_dir,
                             verify=True)

    # Collate the spline knots.
    json_io.dump_all_to_json(working_dir,
                             search_str='contknots_*.txt',
                             json_file=json_file_contknots,
                             json_dir=json_dir,
                             verify=True)

    # Collate the 7.7-micron region cut-outs.
    json_io.dump_all_to_json(working_dir,
                             search_str='cont77_*.txt',
                             json_file=json_file_cont77,
                             json_dir=json_dir,
                             verify=True)

    return True


def convert_folder_spectra(base_dir, sub_dir=None, json_dir=None):
    """Convert the PAHdb spectra to JSON.

    Args:
        base_dir (str): Directory of the "run".
        sub_dir (str): 'run_lorentz' or 'run_gauss', run dependent.
        json_dir (str): Override for json file output directory if set.

    Returns:
        True if succesful.
    """

    working_dir = json_io.ensure_dir(base_dir) + 'spectra/'

    # add subdirectory.
    if sub_dir:
        working_dir += sub_dir
        working_dir = json_io.ensure_dir(working_dir)

        json_file_spectra = 'spectra_dict_' + sub_dir + '.json'

    else:
        json_file_spectra = 'spectra_dict_run.json'

    # Collate the spectra..
    json_io.dump_all_to_json(working_dir,
                             search_str='spectra_*.txt',
                             json_file=json_file_spectra,
                             json_dir=json_dir,
                             verify=True)

    return True


def main():

    print('test')


if __name__ == '__main__':
    main()
