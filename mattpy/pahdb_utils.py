#!/usr/bin/env python3
"""
pahdb_utils.py

Convert full PAHdb runs into pickles to save resources/overheads.
Also include some utilities for converting specific subfolders.
"""

import glob
import numpy as np
import os
import pandas as pd
import shutil

from mattpy import io, utils


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

    working_dir = utils.ensure_dir(base_dir) + 'cont_fit/'

    if method == 'pickle':
        sfx = '.pkl'
    elif method == 'json':
        sfx = '.json'
    else:
        raise ValueError("Method must be one of ['pickle', 'json'].")

    # add subdirectory.
    if sub_dir:
        working_dir += sub_dir
        working_dir = utils.ensure_dir(working_dir)
        # Identify output files with (e.g.,) '_run_lorentz'.
        save_file_contflux = 'pickles_' + sub_dir + '_contflux' + sfx
        save_file_contknots = 'pickles_' + sub_dir + '_contknots' + sfx
        save_file_cont77 = 'pickles_' + sub_dir + '_cont77' + sfx
    else:
        save_file_contflux = 'pickles_run_unknown_contflux' + sfx
        save_file_contknots = 'pickles_run_unknown_contknots' + sfx
        save_file_cont77 = 'pickles_run_unknown_cont77' + sfx

    # Collate the spline vectors.
    dump_all_to_disk(working_dir,
                     search_str='contflux_*.txt',
                     save_file=save_file_contflux,
                     save_dir=save_dir, method=method,
                     verify=True)

    # Collate the spline knots.
    dump_all_to_disk(working_dir,
                     search_str='contknots_*.txt',
                     save_file=save_file_contknots,
                     save_dir=save_dir, method=method,
                     verify=True)

    # Collate the 7.7-micron region cut-outs.
    dump_all_to_disk(working_dir,
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

    working_dir = utils.ensure_dir(base_dir) + 'spectra/'

    if method == 'pickle':
        sfx = '.pkl'
    elif method == 'json':
        sfx = '.json'
    else:
        raise ValueError("Method must be one of ['pickle', 'json'].")

    # add subdirectory.
    if sub_dir:
        working_dir += sub_dir
        working_dir = utils.ensure_dir(working_dir)
        save_file_spectra = 'pickles_' + sub_dir + '_spectra' + sfx
    else:
        save_file_spectra = 'pickles_run_unknown_spectra' + sfx

    # Collate the spectra..
    dump_all_to_disk(working_dir,
                     search_str='spectra_*.txt',
                     save_file=save_file_spectra,
                     save_dir=save_dir, method=method,
                     verify=True)

    return True


def recursive_overwrite(src, dest, ignore=None):
    """Recursive copying file directories, c/o Stack Overflow."""
    if os.path.isdir(src):
        if not os.path.isdir(dest):
            os.makedirs(dest)
        files = os.listdir(src)
        if ignore is not None:
            ignored = ignore(src, files)
        else:
            ignored = set()
        for f in files:
            if f not in ignored:
                recursive_overwrite(os.path.join(src, f),
                                    os.path.join(dest, f),
                                    ignore)
    else:
        try:
            shutil.copyfile(src, dest)
        except Exception as e:
            print(e)


def copy_metadata(origin, destination):
    """Convenience function for copying run metadata."""
    recursive_overwrite(origin, destination)
    return


def copy_results(origin, destination, wave_type='wave77'):
    """Convenience function for copying run results."""
    folder_list = glob.glob(origin + '*')

    for folder in folder_list:

        if folder.split('/')[-1] == 'run_gauss':
            # copy folder/results_method2.txt to new one, rename.
            file_in = folder + '/results_method2.txt'
            file_out = destination + 'results_run_gauss_' + wave_type + '.txt'
            shutil.copyfile(file_in, file_out)
            print("Copied results to: ", file_out)

        elif folder.split('/')[-1] == 'run_lorentz':
            # copy folder/results_method2.txt to new one, rename.
            file_in = folder + '/results_method2.txt'
            file_out = \
                destination + 'results_run_lorentz_' + wave_type + '.txt'
            shutil.copyfile(file_in, file_out)
            print("Copied results to: ", file_out)

    return


def copy_plots(origin, destination):
    """Convenience function for copying run PDFs."""
    dir_list = ['run_lorentz', 'run_gauss']
    file_list = ['all_aroma.pdf', 'all_full.pdf', 'all_zoom.pdf']

    plot_list = []

    # Stuff to try copying.
    for the_dir in dir_list:
        for the_file in file_list:
            plot_list.append(the_dir + '/' + the_file)

    for each_plot in plot_list:
        full_origin = origin + each_plot
        full_dest = destination + 'plots_' + each_plot.replace('/all', '')
        recursive_overwrite(full_origin, full_dest)
        print('Copied plot to: ', full_dest)

    return


def parse_one_spectrum_file(fname):
    """Read a single PAHdb spectrum, return 'beta' and 'ionfrac',
    as well as the spectrum itself.

    Args:
        fname (str): File to parse.

    Returns:
        spec_dict (dict): Containing flux array, beta, ionfrac.
    """

    def parse_filename(filename):
        """Extract beta and ionfrac values from filename."""

        beta_value = filename.split('_beta_')[1].split('_')[0]
        ionfrac_value = filename.split('_ionfrac_')[-1].split('.txt')[0]

        return float(beta_value), float(ionfrac_value)

    beta, ionfrac = parse_filename(fname)

    # Read the file as a pandas dataframe.
    try:
        dataframe = pd.read_csv(fname, sep=',', names=['flux', 'na'])
    except Exception as error:
        raise error

    # Remove trivial column that arises from the csv format.
    del dataframe['na']

    spec_dict = dataframe.to_dict()
    spec_dict['beta'] = beta
    spec_dict['ionfrac'] = ionfrac

    return spec_dict


def convert_txt_to_dict(file_dir, search_str='spectra*.txt'):
    """Import all spectra*.txt files, combine them into a dictionary.

    Args:
        file_dir (str): location of the PAHdb spectra.
        search_str (str): Regex search string for identifying all spectra.

    Note:
        Utilizes pandas DataFrame for reading.
        Dataframe should contain columns for 'beta' and 'ionfrac'.
        FULL_DICT has keys in form (beta, ionfrac) as floats.
    """

    # Data to be combined.
    try:
        glob_files = glob.glob(utils.ensure_dir(file_dir) + search_str)
    except IOError as error:
        raise error

    spectra_files = np.sort(glob_files)

    # Dictionary for holding results.
    full_dict = {}

    # Iterate over all the spectra.
    for index, value in enumerate(spectra_files):

        # Parse the single file into a temporary dataframe.
        tmp_dataframe = parse_one_spectrum_file(value)

        # Create a new key, test that it's unique.
        new_key = (tmp_dataframe['beta'], tmp_dataframe['ionfrac'])
        if new_key in full_dict.keys():
            raise ValueError("key already defined!")
        else:
            full_dict[new_key] = tmp_dataframe['flux']

    return full_dict


def regular_run_structure(run_dir, run_name):
    """Perform all copying operations for a single run."""

    def is_a_run_directory(path):
        """Confirm this directory contains a PAHdb run."""

        path_contents = [x.split('/')[-1] for x in glob.glob(path + '*')]
        if 'METADATA' not in path_contents:
            return False
        else:
            return True

    working_dir = utils.ensure_dir(run_dir) + run_name
    working_dir = utils.ensure_dir(working_dir)

    if not is_a_run_directory(working_dir):
        print("Not a run directory: ", working_dir)
        return

    save_dir = run_dir + '_pickled/' + run_name
    save_dir = utils.ensure_dir(save_dir)
    utils.ensure_exists(save_dir)

    # First, copy the spectra themselves as a pickle.
    convert_folder_spectra(working_dir,
                           sub_dir='run_lorentz',
                           save_dir=save_dir,
                           method='pickle')

    # Next, copy the continuum fit results as a pickle.
    convert_folder_cont_fit(working_dir,
                            sub_dir='run_lorentz',
                            save_dir=save_dir,
                            method='pickle')

    # Now, copy the metadata from the old folder to the new one.
    meta_origin = working_dir + 'METADATA/'
    meta_destination = save_dir + 'metadata/'
    copy_metadata(meta_origin, meta_destination)

    # And copy the results files (beta, ionfrac, Wpeak).
    results_destination = save_dir
    results_list = ['results_wave62/', 'results_wave77/', 'results_wave112/']
    results_type = ['wave62', 'wave77', 'wave112']

    for index, result in enumerate(results_list):
        results_origin = working_dir + result
        if os.path.isdir(results_origin):
            copy_results(results_origin, results_destination,
                         wave_type=results_type[index])

    # And copy the combined (pdftk) plots.
    plots_origin = working_dir + 'plots/'
    plots_destionation = save_dir
    copy_plots(plots_origin, plots_destionation)

    return


def full_run_conversion(run_dir, search_for_sub_dirs=False):
    """Perform a full conversion for all runs within this folder."""

    run_names = glob.glob(run_dir + '/*')
    run_names = [x.split('/')[-1] for x in run_names]

    for run_name in run_names:
        if search_for_sub_dirs:
            sub_dirs = glob.glob(run_dir + '/' + run_name + '/*')
            subby_dirs = [x.split(run_dir + '/')[-1] for x in sub_dirs]
            for sub_name in subby_dirs:
                regular_run_structure(run_dir, sub_name)
        else:
            regular_run_structure(run_dir, run_name)

    return


def dump_all_to_disk(file_dir, search_str='spectra*.txt',
                     save_file='spectra_dict.json',
                     save_dir=None, method='pickle',
                     verify=True):
    """Shorthand for converting all .txt PAHdb files to JSON or df pickle.

    Args:
        file_dir (str): Directory containg 'spectra*.txt'.
        search_str (str): Glob search string for files to include.
        save_file (str): Output filename.
        save_dir (str): Desired JSON output directory. If None,
            will default to filedir.
        method (str): Whether to create a pickle or JSON.
        verify (bool): Whether to verify the output matches the input after
            being written/read.

    Note:
        Places resulting JSON files in the same directory as the .txt
            PAHdb files.
        Method can be either 'pickle' or 'json'.
    """

    file_dir = utils.ensure_dir(file_dir)
    mydict = convert_txt_to_dict(file_dir, search_str)

    # Determine location for output.
    if save_dir:
        save_path = utils.ensure_dir(save_dir) + save_file
    else:
        save_path = utils.ensure_dir(file_dir) + save_file

    # Write to disk. If pickling, convert to dataframe first.
    if method == 'pickle':
        df = pd.DataFrame.from_dict(mydict)
        io.write_dataframe_to_pickle(save_path, df)
        if verify:
            load_df = pd.read_pickle(save_path)
            assert utils.verify_dataframe_equality(df, load_df)
    elif method == 'json':
        io.write_dict_to_json(save_path, mydict)
        if verify:
            load_dict = io.read_dict_from_json(save_path)
            assert utils.verify_dict_equality(mydict, load_dict)
    else:
        raise ValueError("Unknown method for saving to disk.")

    return True


def main():
    print('Uncomment below...')
    # full_run_conversion('runs_baypahs')
    # full_run_conversion('runs_cutsmall')
    # full_run_conversion('runs_dehydro')
    # full_run_conversion('runs_fit62112')
    # full_run_conversion('runs_morepts')
    # full_run_conversion('runs_new')
    # full_run_conversion('runs_vary_teff', search_for_sub_dirs=True)


if __name__ == '__main__':
    main()
