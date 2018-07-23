#!/usr/bin/env python3
"""
convert_pahdb_runs.py

Convert old PAHdb runs into pickles to save resources/overheads.
"""

import glob
import os
import shutil

from mattpy import data_io, pahdb_io
# from ipdb import set_trace as st


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


def ensure_exists(path):
    """Make sure the path exists."""
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError as error:
            raise(error)


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


def regular_run_structure(run_dir, run_name):
    """Perform all copying operations for a single run."""

    def is_a_run_directory(path):
        """Confirm this directory contains a PAHdb run."""

        path_contents = [x.split('/')[-1] for x in glob.glob(path + '*')]
        if 'METADATA' not in path_contents:
            return False
        else:
            return True

    working_dir = data_io.ensure_dir(run_dir) + run_name
    working_dir = data_io.ensure_dir(working_dir)

    if not is_a_run_directory(working_dir):
        print("Not a run directory: ", working_dir)
        return

    save_dir = run_dir + '_pickled/' + run_name
    save_dir = data_io.ensure_dir(save_dir)
    ensure_exists(save_dir)

    # First, copy the spectra themselves as a pickle.
    pahdb_io.convert_folder_spectra(working_dir,
                                    sub_dir='run_lorentz',
                                    save_dir=save_dir,
                                    method='pickle')

    # Next, copy the continuum fit results as a pickle.
    pahdb_io.convert_folder_cont_fit(working_dir,
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
