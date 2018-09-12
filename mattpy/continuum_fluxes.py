#!/usr/bin/env python3
"""
continuum_fluxes.py

Measure the continuum flux levels of a spectrum.
"""

import numpy as np

from mattpy.utils import find_nearest, smooth


def measure(wave_knots, wave, flux, find_min, width, smooth_flag,
            smoothsize, windowsize):
    """Measure the continuum fluxes from a spectrum."""

    holdWave = []
    holdFlux = []

    smoowave = wave
    smooflux = smooth.smooth(flux, window_len=smoothsize)

    for i in range(len(wave_knots)):

        # Determine window size and nominal wavelength position.
        lamb = wave_knots[i]

        if wave[0] <= lamb <= wave[-1]:

            if width[i] == 0:
                width[i] = windowsize

            # Use smoothed flux if desired.
            if smooth_flag[i] == 1:
                inwave = smoowave
                influx = smooflux
            else:
                inwave = wave
                influx = flux

            # Measure, depending on if you want minima finding or not.
            if find_min[i] == 1:
                dL = width[i]
                winDX = np.where((inwave >= lamb - dL) &
                                 (inwave <= lamb + dL))
                winWave = inwave[winDX]
                winFlux = influx[winDX]

                if np.all(np.isnan(winFlux)):
                    holdWave.append(np.nan)
                    holdFlux.append(np.nan)
                    continue

                minWave = winWave[np.nanargmin(winFlux)]
                minFlux = np.nanmin(winFlux)
                holdWave.append(minWave)
                holdFlux.append(minFlux)
            else:
                tmpwave = lamb
                tmpflux = influx[find_nearest.find_nearest(inwave, tmpwave)]
                holdWave.append(tmpwave)
                holdFlux.append(tmpflux)

    return np.array(holdWave), np.array(holdFlux)
