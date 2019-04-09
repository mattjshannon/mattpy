#!/usr/bin/env python3
# -*- coding: <encoding name> -*-
"""
spectrum.py

Spectrum class for wrapping up Spitzer observations.

Matt J. Shannon
Nov. 2018
"""

import numpy as np

import scripts.helpers as helpers


class FullSpectrum:
    """Wrap up a spectrum for analysis."""

    def __init__(self, filename, is_windows=False):
        self.filename = filename
        self.is_windows = is_windows
        self.__set_basename()
        self.__load_data()

    def __set_basename(self):
        """Extract the raw filename, minus the extension."""
        if self.is_windows:
            split_char = '\\'
        else:
            split_char = '/'

        self.basename = self.filename.split(split_char)[-1].split('.txt')[0]

    def __load_data(self):
        # Load spectrum (in units of Jy).
        wave, flux, fluxerr = np.loadtxt(self.filename, delimiter=',').T
        self.wave = wave
        self.flux_jy = flux
        self.fluxerr_jy = fluxerr
        self.flux_si = helpers.jy_to_si(self.flux_jy, wave)
        self.fluxerr_si = helpers.jy_to_si(self.fluxerr_jy, wave)
        self.rms = helpers.measure_112_RMS(wave, self.flux_si)

    def measure_all_bands(self, output_dir='output/'):
        """Measure aliphatics and aromatics simultaneously."""
        wrap = (self.basename, self.wave, self.flux_si,
                self.fluxerr_si, self.rms, output_dir)
        results = helpers.fit_all(*wrap)
