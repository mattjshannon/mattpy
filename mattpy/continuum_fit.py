#!/usr/bin/env python3
"""
continuum_fit.py

Fit and subtract a spline continuum from a spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

from mattpy import continuum_fluxes as cf
from mattpy import measure_features as mf


class Spectrum(object):
    """Create an object to hold fit parameters, anchors, etc.

    Attributes:
        anchor_abscissa (np.array): Abscissa of anchor knots.
        anchor_ordinate (np.array): Ordinate of anchor knots.

    """

    def __init__(self, wave, flux, spline_flux=None, csub=None,
                 anchor_wave_input=None,
                 anchor_wave=None, anchor_flux=None,
                 window_size=0.12, smooth_size=8,
                 flag_smooth=None, flag_findmin=None,
                 anchor_smooth_array=None, anchor_findmin_array=None,
                 anchor_width_array=None):
        """Return a Spectrum object.

        Note:
            Hm, should require anchors? Or more general?

        Args:
            file_path (str):

        """
        self.wave = wave
        self.flux = flux
        self.csub = csub

        # Set anchor parameters.
        self.window_size = window_size
        self.smooth_size = smooth_size
        self.flag_smooth = flag_smooth
        self.flag_findmin = flag_findmin
        self.anchor_wave = anchor_wave
        self.anchor_wave_input = anchor_wave_input
        self.anchor_smooth_array = anchor_smooth_array
        self.anchor_findmin_array = anchor_findmin_array
        self.anchor_width_array = anchor_width_array

        if anchor_flux:
            self.anchor_wave = anchor_wave_input
            self.anchor_flux = anchor_flux
        else:
            # self._set_anchors()
            if self.anchor_wave_input:
                self._set_anchors()
            else:
                self.anchor_wave, self.anchor_findmin_array, \
                    self.anchor_width_array, self.anchor_smooth_array = \
                    mf.ctPoints_special()

                self.anchor_wave, self.anchor_flux = \
                    cf.measure(self.anchor_wave, self.wave, self.flux,
                               self.anchor_findmin_array,
                               self.anchor_width_array,
                               self.anchor_smooth_array,
                               smoothsize=self.smooth_size,
                               windowsize=self.window_size)

        # Fit spline.
        self._fit_spline()

    def _set_anchors(self):
        """Find the anchor positions (wave and flux).

        Note:
            Defines arrays for smooth, width, and findmin if not passed.
        """

        # # Take some default values.
        # self.anchor_wave, self.anchor_findmin_array, \
        #     self.anchor_width_array, self.anchor_smooth_array = \
        #     mf.ctPoints_special()

        # Override if anchors defined directly.
        if self.anchor_wave_input:
            self.anchor_wave = self.anchor_wave_input
            n_elements = len(self.anchor_wave)

            # Determine which anchor points to smooth.
            if not self.anchor_smooth_array:
                if self.flag_smooth:
                    self.anchor_smooth_array = [1] * n_elements
                else:
                    self.anchor_smooth_array = [0] * n_elements

            # Determine which anchor points to find the minimum for.
            if not self.anchor_findmin_array:
                if self.flag_findmin:
                    self.anchor_findmin_array = [1] * n_elements
                else:
                    self.anchor_findmin_array = [0] * n_elements

            # Determine width in abscissa for local minimum searching.
            if not self.anchor_width_array:
                self.anchor_width_array = [self.window_size] * n_elements

        # Override smoothing if defined.
        if self.flag_smooth:
            n_elements = len(self.anchor_wave)
            if self.flag_smooth == 1:
                self.anchor_smooth_array = [1] * n_elements
            elif self.flag_smooth == 0:
                self.anchor_smooth_array = [0] * n_elements

        # Override findmin if defined.
        if self.flag_findmin:
            n_elements = len(self.anchor_wave)
            if self.flag_findmin == 1:
                self.anchor_findmin_array = [1] * n_elements
            elif self.flag_findmin == 0:
                self.anchor_findmin_array = [0] * n_elements

        # st()

        # Interpolate the anchor positions.
        self.anchor_wave, self.anchor_flux = \
            cf.measure(self.anchor_wave, self.wave, self.flux,
                       self.anchor_findmin_array, self.anchor_width_array,
                       self.anchor_smooth_array,
                       smoothsize=self.smooth_size,
                       windowsize=self.window_size)
        return

    def _fit_spline(self):
        """Fit a standard cubic spline a la Peeters et al. 2002.

        Note:

        """
        # st()
        spl = interp.splrep(self.anchor_wave, self.anchor_flux)
        self.spline_flux = interp.splev(self.wave, spl)

        self.csub = self.flux - self.spline_flux
        return

    def plot(self):
        """Plot stuff to make sure the fit is good."""
        plt.plot(self.wave, self.flux, label='Data')
        plt.plot(self.wave, self.spline_flux, label='Spline')
        plt.plot(self.anchor_wave, self.anchor_flux, 'ko')
        plt.plot(self.wave, self.csub, label='Csub')
        plt.legend(loc=0)
        plt.show()
        plt.close()
        return


if __name__ == '__main__':

    wave, flux, fluxerr = \
        np.loadtxt('sample/datamrk33_PAHFit_full.tbl', skiprows=2).T

    test = Spectrum(wave=wave, flux=flux, flag_smooth=True)
    test.plot()
