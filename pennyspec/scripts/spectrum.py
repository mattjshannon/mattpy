#!/usr/bin/env python3
# -*- coding: <encoding name> -*-
"""
spectrum.py

Spectrum class for wrapping up Spitzer observations.

Matt J. Shannon
Nov. 2018
"""

import matplotlib.pyplot as plt
import numpy as np

import scripts.helpers as helpers


class Spectrum:
    """Wrap up a spectrum for analysis."""

    def __init__(self, filename, is_Windows=False):
        self.filename = filename
        self.is_Windows = is_Windows
        self.__set_basename()
        self.__load_data()

    def __set_basename(self):
        """Extract the raw filename, minus the extension."""
        if self.is_Windows:
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

    def plot_spectrum(self, output_dir='output/', units='si', verbose=True):
        """Save a PDF of the spectrum, raw and smoothed."""
        if units == 'si':
            flux = self.flux_si
            fluxerr = self.fluxerr_si
        elif units == 'jy':
            flux = self.flux_jy
            fluxerr = self.fluxerr_jy
        else:
            raise ValueError('Units must be one of ("si", "jy")')

        """
        Smooth spectrum? Well, a window length of 5 is basically no smoothing
        for this resolution of spectrum. Might as well skip it.

        # ax.plot(self.wave, flux, '-', lw=2, label=self.basename)
        # smoothed_flux = helpers.smooth(flux, window_len=5)
        # ax.plot(self.wave, smoothed_flux, '--', lw=1, label='smoothed')
        """

        # Plot full cont subtracted spectrum
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.errorbar(self.wave, flux, fluxerr, color='r', ecolor='0.45',
                    lw=2, elinewidth=1)

        # Set plot parameters.
        ax.axhline(y=0, color='k', ls='-', zorder=-10, lw=1)
        ax.set_xlabel('Wavelength (micron)', fontsize=12)
        ax.set_ylabel(r'Flux density (${\rm W}/{\rm m}^2$)', fontsize=12)
        ax.set_title(self.basename + ' -- Full continuum-subtracted spectrum')
        ax.grid(ls=':')
        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', right=True, top=True)

        # Save and close.
        pdf_filename = output_dir + self.basename + '.pdf'
        fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        plt.close()
        fig.clear()

        if verbose:
            print('Saved: ', pdf_filename)

    def fit_aliphatics(self, output_dir='output/', skip_72=False):
        """Measure the aliphatic bands."""
        wrap = (self.basename, self.wave, self.flux_si,
                self.fluxerr_si, self.rms, output_dir)

        fitAli, waveLim, area69, SNR69, area72, SNR72 = \
            helpers.fit_aliphatics(*wrap)

        self.fitAli = fitAli
        self.waveLim = waveLim
        self.area69 = area69
        self.SNR69 = SNR69
        self.area72 = area72
        self.SNR72 = SNR72

    def fit_aromatics(self, output_dir='output/', skip_72=False):
        """Measure the aromatic bands."""
        wrap = (self.basename, self.wave, self.flux_si,
                self.fluxerr_si, self.rms, output_dir)

        waveLim77, fit77, area77, feature = helpers.fit_aromatics(*wrap)
        area11 = helpers.measure_112(*wrap)

        self.waveLim77 = waveLim77
        self.fit77 = fit77
        self.area77 = area77
        self.feature = feature
        self.area11 = area11

    def save_results(self, output_dir='output/'):
        """Write the results to disk."""
        wrap = (self.fitAli, self.fit77, self.basename, self.waveLim,
                self.waveLim77, self.area69, self.area72, self.area77,
                self.area11, self.SNR69, self.SNR72, self.feature)

        helpers.save_fit_parameters(output_dir, wrap)

        return
