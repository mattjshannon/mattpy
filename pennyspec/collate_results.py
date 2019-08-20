#!/usr/bin/env python
"""
collate_results.py

Collate the pickly results from analyze_pahs.py.
"""

import glob
import numpy as np
import pickle

# Load data:
PICKLE_DIR = 'output_table2/numeric/'
OUTPUT_DIR = 'output_table2/numeric/'
FILE_LIST = np.sort(glob.glob(PICKLE_DIR + '*.pkl'))

# Store results somehow...
results = {
    'index': [],
    'basename': [],
    'pah77': [],
    'line69': [],
    'line72': [],
    'g76': [],
    'g78': [],
    'g82': [],
    'g86': [],
}

# Iterate over each spectrum and produce plots/fit parameters:
for index, filename in enumerate(FILE_LIST):
    print(index, filename)

    basename = filename.split('.pkl')[0].split('/')[-1]
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    results['index'].append(index)
    results['basename'].append(basename)
    for key in data:
        line = data[key]

        if key == 'pah77':
            results[key].append((line['centroid'], line['flux'],
                                 line['fluxerr']))
        else:
            results[key].append((line['position'],
                                 line['integrated_flux'],
                                 line['integrated_fluxerr'],
                                 line['sigma']))

# Combine results for all sources.
rr = np.column_stack((results['index'], results['basename'],
                      results['pah77'], results['line69'],
                      results['line72'], results['g76'], results['g78'],
                      results['g82'], results['g86']))

header = 'index, basename, pah77 (position, flux, fluxerr), ' \
         'aliphatic 6.9 (*4), aliphatic 7.2 (*4), ' \
         'g76 (*4), g78 (*4), g82 (*3), g86 (*4)\n' \
         'NOTE: (*4) means 4 columns: ' \
         'position, integrated flux (in W/m^2), ' \
         'integrated flux error, sigma.\n' \
         'NOTE: pah77 flux is sum of g76, g78 and g82.'

# Save to disk.
np.savetxt('results_6gauss.txt', rr, fmt='%s', delimiter=',',
           header=header)
print('Saved all results to: ', 'results_6gauss.txt')
