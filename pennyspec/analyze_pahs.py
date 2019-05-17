#!/usr/bin/env python
"""
analyze_pahs.py

Measure the aliphatic and aromatic features in Spitzer IRS spectra.

Notes:
    Peeters et al. 2017:
    "λ (FWHM) of 7.59 (0.450), 7.93 (0.300), 8.25 (0.270), and 8.58 (0.344) μm"
    Allow to vary... doesn't say how much .

    Stock et al. 2017:
    Allow peak to vary within 0.2 µm window, FWHM to vary within 0.25µm window.
    Fixed ones, averages:
    7.55 (0.44 fwhm)
    7.87 (0.40 fwhm)
    8.25 (0.29 fwhm)
    8.59 (0.36 fwhm)

"""

import glob
import numpy as np

from scripts.spectrum import Spectrum
from scripts.spectrum_full import FullSpectrum

# Load data:
# DATA_DIR = 'input/'
DATA_DIR = 'input_table2/'
# OUTPUT_DIR = 'output/'
OUTPUT_DIR = 'output_table2/'
FILE_LIST = np.sort(glob.glob(DATA_DIR + '*CWsub.txt'))

# Iterate over each spectrum and produce plots/fit parameters:
for index, filename in enumerate(FILE_LIST):
    print(index, filename)

    if index not in [17]:
        continue

    # spectrum = Spectrum(filename=filename, is_Windows=False)
    # spectrum.plot_spectrum(output_dir=OUTPUT_DIR + 'spectra/', units='si')
    # spectrum.fit_aliphatics(output_dir=OUTPUT_DIR + 'diagnostics/')
    # spectrum.fit_aromatics(output_dir=OUTPUT_DIR + 'diagnostics/')
    # spectrum.save_results(output_dir=OUTPUT_DIR + 'parameters/')

    fullspec = FullSpectrum(filename=filename)
    fullspec.measure_all_bands(output_dir=OUTPUT_DIR)
    # TODO: save model results if we're going that way.
