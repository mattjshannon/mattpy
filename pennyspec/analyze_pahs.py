#!/usr/bin/env python
"""
analyze_pahs.py

Measure the aliphatic and aromatic features in Spitzer IRS spectra.
"""

import glob
import numpy as np

from scripts.spectrum import Spectrum
from scripts.spectrum_full import FullSpectrum

# Load data:
data_dir = 'input/'
output_dir = 'output/'
file_list = np.sort(glob.glob(data_dir + '*_CWsub.txt'))

# Iterate over each spectrum and produce plots/fit parameters:
for index, filename in enumerate(file_list):
    # if index != 9:
    #     continue

    # spectrum = Spectrum(filename=filename, is_Windows=False)
    # spectrum.plot_spectrum(output_dir=output_dir + 'spectra/', units='si')
    # spectrum.fit_aliphatics(output_dir=output_dir + 'diagnostics/')
    # spectrum.fit_aromatics(output_dir=output_dir + 'diagnostics/')
    # spectrum.save_results(output_dir=output_dir + 'parameters/')

    fullspec = FullSpectrum(filename=filename)
    fullspec.measure_all_bands(output_dir=output_dir + 'fullspec/')

    # raise SystemExit("Early exit for testing.")
