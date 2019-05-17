#!/usr/bin/env python3
"""
helpers.py

Helpers functions for analyze_pahs.py
"""

import errno
import os
import pickle

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from gaussfitter import onedgaussian, onedgaussfit, n_gaussian, \
    multigaussfit
from ipdb import set_trace as st
from scipy.integrate import simps

from mattpy.utils import to_sigma, to_fwhm, quant_str
from scripts.mpfit import mpfit


def ensure_exists(path):
    """Ensure the path exists; if not, make the directory."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def jy_to_si(flux_jy, wave):
    """Returns a flux array (converted from Jy to W/m^2/micron)."""
    flux_si = flux_jy * 3e-12 / wave**2
    return flux_si


def smooth(x, window_len=50, window='hanning'):
    """Returns a smoothed version of an array, from Stack Overflow."""
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector must be bigger than window size.")
    if window_len < 3:
        return x

    acceptable_windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    if window not in acceptable_windows:
        raise ValueError("Window must be in: ", str(acceptable_windows))

    s = np.r_[2 * x[0] - x[window_len - 1::-1], x,
              2 * x[-1] - x[-1:-window_len:-1]]

    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='same')

    return y[window_len:-window_len + 1]


def compute_feature_uncertainty(gposition, gsigma, wave_feat, rms):

    myrange = [gposition - (3. * gsigma), gposition + (3. * gsigma)]

    dl = wave_feat[1] - wave_feat[0]
    N = (myrange[1] - myrange[0]) / dl
    feature_uncertainty = (rms * np.sqrt(N) * dl * 2)

    return feature_uncertainty


def params_6gauss(basename, guess):

    p0 = {
        'params':
            [
            guess / 2.,  6.89, to_sigma(0.15),
            guess / 4.,  7.25, to_sigma(0.12),
            guess / 2.,  7.55, to_sigma(0.44),
            guess / 1.,  7.87, to_sigma(0.40),
            guess / 2.,  8.25, to_sigma(0.29),
            guess / 2.,  8.59, to_sigma(0.36),
        ],
        'limitedmin': [True] * 18,
        'limitedmax': [True] * 18,
        'fixed':
            [
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        ],
        'minpars':
            [
            guess / 30., 6.82,  to_sigma(0.06),
            0,           7.15,  to_sigma(0.05),
            0,           7.45,  to_sigma(0.315),
            0,           7.77,  to_sigma(0.275),
            0,           8.15,  to_sigma(0.165),
            0,           8.49,  to_sigma(0.235),
        ],
        'maxpars':
            [
            guess, 6.96, to_sigma(0.21),
            guess, 7.35, to_sigma(0.15),
            guess, 7.65, to_sigma(0.565),
            guess, 7.97, to_sigma(0.525),
            guess, 8.35, to_sigma(0.415),
            guess, 8.69, to_sigma(0.485),
        ]
    }

    p1 = {
        'params':
            [
            guess / 2.,  6.89, to_sigma(0.15),
            guess / 4.,  7.25, to_sigma(0.12),
            guess / 2.,  7.55, to_sigma(0.44),
            guess / 1.,  7.87, to_sigma(0.40),
            guess / 2.,  8.25, to_sigma(0.29),
            guess / 2.,  8.59, to_sigma(0.36),
        ],
        'limitedmin': [True] * 18,
        'limitedmax': [True] * 18,
        'fixed':
            [
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        ],
        'minpars':
            [
            guess / 30., 6.82,  to_sigma(0.06),
            guess / 40., 7.15,  to_sigma(0.05),
            guess / 30., 7.45,  to_sigma(0.315),
            guess / 30., 7.77,  to_sigma(0.275),
            0,           8.15,  to_sigma(0.165),
            guess / 30., 8.49,  to_sigma(0.235),
        ],
        'maxpars':
            [
            guess, 6.96, to_sigma(0.21),
            guess, 7.35, to_sigma(0.15),
            guess, 7.65, to_sigma(0.565),
            guess, 7.97, to_sigma(0.525),
            guess, 8.35, to_sigma(0.415),
            guess, 8.69, to_sigma(0.485),
        ]
    }

    p2 = {
        'params':
            [
            guess / 2.,  6.89, to_sigma(0.15),
            0., 7.25, to_sigma(0.12),
            guess / 2.,  7.55, to_sigma(0.44),
            guess / 1.,  7.87, to_sigma(0.40),
            guess / 2.,  8.25, to_sigma(0.29),
            guess / 2.,  8.59, to_sigma(0.36),
        ],
        'limitedmin': [True] * 18,
        'limitedmax': [True] * 18,
        'fixed':
            [
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        ],
        'minpars':
            [
            guess / 30., 6.82,  to_sigma(0.06),
            0.,          7.15,  to_sigma(0.05),
            guess / 30., 7.45,  to_sigma(0.315),
            guess / 30., 7.77,  to_sigma(0.275),
            0.,          8.15,  to_sigma(0.165),
            guess / 30., 8.49,  to_sigma(0.235),
        ],
        'maxpars':
            [
            guess, 6.96, to_sigma(0.21),
            guess, 7.35, to_sigma(0.15),
            guess, 7.65, to_sigma(0.565),
            guess, 7.97, to_sigma(0.525),
            guess, 8.35, to_sigma(0.415),
            guess, 8.69, to_sigma(0.485),
        ]
    }

    p3 = {
        'params':
            [
            guess / 2.,  6.89, to_sigma(0.15),
            guess / 4., 7.25, to_sigma(0.12),
            guess / 2.,  7.55, to_sigma(0.44),
            guess / 1.,  7.87, to_sigma(0.40),
            guess / 2.,  8.25, to_sigma(0.29),
            guess / 2.,  8.59, to_sigma(0.36),
        ],
        'limitedmin': [True] * 18,
        'limitedmax': [True] * 18,
        'fixed':
            [
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        ],
        'minpars':
            [
            guess / 30., 6.82,  to_sigma(0.06),
            guess / 30., 7.15,  to_sigma(0.05),
            guess / 30., 7.45,  to_sigma(0.315),
            guess / 30., 7.77,  to_sigma(0.275),
            guess / 30., 8.15,  to_sigma(0.165),
            guess / 30., 8.49,  to_sigma(0.235),
        ],
        'maxpars':
            [
            guess, 6.96, to_sigma(0.21),
            guess, 7.35, to_sigma(0.15),
            guess, 7.65, to_sigma(0.565),
            guess, 7.97, to_sigma(0.525),
            guess, 8.35, to_sigma(0.415),
            guess, 8.69, to_sigma(0.485),
        ]
    }

    p4 = {
        'params':
            [
            guess / 2.,  6.89, to_sigma(0.15),
            guess / 6.,  7.25, to_sigma(0.12),
            guess / 2.,  7.55, to_sigma(0.44),
            guess / 1.,  7.87, to_sigma(0.40),
            guess / 2.,  8.25, to_sigma(0.29),
            guess / 2.,  8.59, to_sigma(0.36),
        ],
        'limitedmin': [True] * 18,
        'limitedmax': [True] * 18,
        'fixed':
            [
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        ],
        'minpars':
            [
            guess / 30., 6.82,  to_sigma(0.06),
            guess / 30., 7.15,  to_sigma(0.05),
            guess / 30., 7.45,  to_sigma(0.315),
            guess / 30., 7.77,  to_sigma(0.275),
            0.,          8.15,  to_sigma(0.165),
            guess / 30., 8.49,  to_sigma(0.235),
        ],
        'maxpars':
            [
            guess, 6.96, to_sigma(0.21),
            guess, 7.35, to_sigma(0.15),
            guess, 7.65, to_sigma(0.565),
            guess, 7.97, to_sigma(0.525),
            guess, 8.35, to_sigma(0.415),
            guess, 8.69, to_sigma(0.485),
        ]
    }

    p5 = {
        'params':
            [
            guess / 2.,  6.89, to_sigma(0.15),
            1.21852599e-15 * 0.1,  7.25, to_sigma(0.1),
            guess / 2.,  7.55, to_sigma(0.44),
            guess / 1.,  7.87, to_sigma(0.40),
            guess / 2.,  8.25, to_sigma(0.29),
            guess / 2.,  8.59, to_sigma(0.36),
        ],
        'limitedmin': [True] * 18,
        'limitedmax': [True] * 18,
        'fixed':
            [
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        ],
        'minpars':
            [
            guess / 30., 6.82,  to_sigma(0.06),
            0., 7.15,  to_sigma(0.05),
            guess / 30., 7.45,  to_sigma(0.315),
            guess / 30., 7.77,  to_sigma(0.275),
            0.,          8.15,  to_sigma(0.165),
            guess / 30., 8.49,  to_sigma(0.235),
        ],
        'maxpars':
            [
            guess, 6.96, to_sigma(0.21),
            1.21852599e-15 * 0.25, 7.35, to_sigma(0.13),
            guess, 7.65, to_sigma(0.565),
            guess, 7.97, to_sigma(0.525),
            guess, 8.35, to_sigma(0.415),
            guess, 8.69, to_sigma(0.485),
        ]
    }

    p6 = {
        'params':
            [
            guess / 2.,  6.93, to_sigma(0.15),
            guess / 6.,  7.25, to_sigma(0.12),
            guess / 2.,  7.55, to_sigma(0.40),
            guess / 1.,  7.87, to_sigma(0.40),
            guess / 2.,  8.25, to_sigma(0.29),
            guess / 2.,  8.59, to_sigma(0.36),
        ],
        'limitedmin': [True] * 18,
        'limitedmax': [True] * 18,
        'fixed':
            [
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        ],
        'minpars':
            [
            guess / 30., 6.91,  to_sigma(0.06),
            0., 7.15,  to_sigma(0.05),
            guess / 30., 7.53,  to_sigma(0.315),
            guess / 30., 7.77,  to_sigma(0.275),
            0.,          8.15,  to_sigma(0.165),
            guess / 30., 8.49,  to_sigma(0.235),
        ],
        'maxpars':
            [
            guess, 6.96, to_sigma(0.16),
            guess, 7.35, to_sigma(0.15),
            guess, 7.65, to_sigma(0.41),
            guess, 7.97, to_sigma(0.525),
            guess, 8.35, to_sigma(0.415),
            guess, 8.69, to_sigma(0.485),
        ]
    }

    p7 = {
        'params':
            [
            guess / 2.,  6.89, to_sigma(0.15),
            guess / 6.,  7.25, to_sigma(0.12),
            guess / 2.,  7.55, to_sigma(0.44),
            guess / 1.,  7.87, to_sigma(0.40),
            guess / 2.,  8.25, to_sigma(0.29),
            guess / 2.,  8.59, to_sigma(0.36),
        ],
        'limitedmin': [True] * 18,
        'limitedmax': [True] * 18,
        'fixed':
            [
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        ],
        'minpars':
            [
            guess / 30., 6.82,  to_sigma(0.06),
            guess / 30., 7.15,  to_sigma(0.05),
            guess / 30., 7.45,  to_sigma(0.315),
            guess / 30., 7.77,  to_sigma(0.275),
            0.,          8.15,  to_sigma(0.165),
            guess / 30., 8.49,  to_sigma(0.235),
        ],
        'maxpars':
            [
            guess, 6.96, to_sigma(0.21),
            guess, 7.35, to_sigma(0.15),
            guess, 7.65, to_sigma(0.565),
            guess, 7.97, to_sigma(0.525),
            guess, 8.35, to_sigma(0.415),
            guess, 8.69, to_sigma(0.485),
        ]
    }

    p8 = {
        'params':
            [
            guess / 2.,  6.89, to_sigma(0.15),
            guess / 6.,  7.25, to_sigma(0.12),
            guess / 2.,  7.60, to_sigma(0.44),
            guess / 1.,  7.87, to_sigma(0.40),
            guess / 2.,  8.25, to_sigma(0.29),
            guess / 2.,  8.59, to_sigma(0.36),
        ],
        'limitedmin': [True] * 18,
        'limitedmax': [True] * 18,
        'fixed':
            [
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        ],
        'minpars':
            [
            guess / 30., 6.82,  to_sigma(0.06),
            guess / 30., 7.15,  to_sigma(0.05),
            guess / 30., 7.45,  to_sigma(0.315),
            guess / 30., 7.77,  to_sigma(0.275),
            guess / 30., 8.15,  to_sigma(0.165),
            guess / 30., 8.49,  to_sigma(0.235),
        ],
        'maxpars':
            [
            guess, 6.96, to_sigma(0.21),
            guess, 7.35, to_sigma(0.15),
            guess, 7.65, to_sigma(0.565),
            guess, 7.97, to_sigma(0.525),
            guess, 8.35, to_sigma(0.415),
            guess, 8.69, to_sigma(0.485),
        ]
    }

    p9 = {
        'minpars':
            [
            guess / 30., 6.82, to_sigma(0.06),
            0.,          7.15, to_sigma(0.05),
            guess / 30., 7.45, to_sigma(0.315),
            guess / 30., 7.77, to_sigma(0.275),
            0.,          8.15, to_sigma(0.165),
            guess / 30., 8.49, to_sigma(0.235),
        ],
        'params':
            [
            guess / 2.,  6.89, to_sigma(0.15),
            guess / 6.,  7.25, to_sigma(0.12),
            guess / 2.,  7.55, to_sigma(0.44),
            guess / 1.,  7.87, to_sigma(0.40),
            guess / 2.,  8.25, to_sigma(0.29),
            guess / 2.,  8.59, to_sigma(0.36),
        ],
        'maxpars':
            [
            guess,       6.96, to_sigma(0.21),
            guess,       7.35, to_sigma(0.15),
            guess,       7.65, to_sigma(0.565),
            guess,       7.97, to_sigma(0.525),
            guess,       8.35, to_sigma(0.415),
            guess,       8.69, to_sigma(0.485),
        ],
        'limitedmin': [True] * 18,
        'limitedmax': [True] * 18,
        'fixed':
            [
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        ]
    }

    p10 = {
        'params':
            [
            guess / 2.,  6.89, to_sigma(0.15),
            guess / 4.,  7.25, to_sigma(0.12),
            guess / 2.,  7.55, to_sigma(0.44),
            guess / 1.,  7.87, to_sigma(0.40),
            guess / 2.,  8.25, to_sigma(0.29),
            guess / 2.,  8.59, to_sigma(0.36),
        ],
        'limitedmin': [True] * 18,
        'limitedmax': [False] * 18,
        'fixed':
            [
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        False, False, False,
        ],
        'minpars':
            [
            guess / 30., 6.82,  to_sigma(0.06),
            guess / 40., 7.22,  to_sigma(0.07),
            guess / 30., 7.45,  to_sigma(0.315),
            guess / 30., 7.77,  to_sigma(0.275),
            0,           8.15,  to_sigma(0.165),
            guess / 30., 8.49,  to_sigma(0.235),
        ],
        'maxpars':
            [
            guess, 6.96, to_sigma(0.16),
            guess, 7.32, to_sigma(0.15),
            guess, 7.65, to_sigma(0.6),
            guess, 7.97, to_sigma(0.525),
            guess, 8.35, to_sigma(0.415),
            guess, 8.69, to_sigma(0.485),
        ]
    }

    param_dict = {
        'hd97048_convCWsub': p0,            # GOOD, wouldn't trust 72 tho
        'hd135344_convCWsub': p0,           # * NO ALIPHATICS TRUSTED!!! *
        'IRAS05063_CWsub': p3,              # GOOD
        'IRAS05092_CWsub': p0,              # GOOD
        'IRAS05186_CWsub': p0,              # GOOD
        'IRAS05361_CWsub': p0,              # GOOD -- TRY RESTRICTING 7.6 GASUSS FROM LEFT?
        'IRAS05370_CWsub': p4,              # GOOD, don't trust 7.2
        'IRAS05413_CWsub': p2,              # GOOD ENOUGH I GUESS? ONLY TRUST 6.9, maybe 77 flux
        'IRAS05588_CWsub': p0,              # GOOD
        'IRAS06111_CWsub': p0,              # GOOD
        'IRAS14429_CWsub': p0,              # GOOD
        'IRAS15482_CWsub': p5,              # GOOD, don't trust 7.2 maybe (manual)
        'iras17047_SWS_CWsub': p10,         # GOOD, had to do ct myself
        'IRASF05110-6616_LR_CWsub': p0,     # GOOD
        'IRASf05192_CWsub': p1,             # GOOD, quesitonable 69/72. tho
        'J004441_CWsub': p0,                # GOOD
        'J010546_CWsub': p6,                # GOOD, not perfect but good enough?
        'j050713_CWsub': p7,                # GOOD
        'J052043_CWsub': p8,                # GOOD, had to drop errors (not fitting?)
        'J052520_CWsub': p1,                # GOOD
        'NGC1978WBT2665_CWsub': p1,         # GOOD
        'SMPLMC076_CWsub': p1,              # new
        'SMPSMC006_CWsub': p9,              # GOOD, dropping fluxerr in fit (!!)
        'SMPSMC011_CWsub': p1,              # GOOD
    }

    # TO DO: UNCERTAINTIES!!!

    # pos, flux, sigma = line69_params[basename]
    # amp = flux / (np.sqrt(2) * np.abs(sigma) * np.sqrt(np.pi))

    # max_flux72 = 0.35 * flux
    # amp_72_approx = max_flux72 / (np.sqrt(2) * np.abs(sigma) * np.sqrt(np.pi))


    # p0['params'][0] = amp
    # p0['params'][1] = pos
    # p0['params'][2] = sigma

    # p0['fixed'][0] = True
    # p0['fixed'][1] = True
    # p0['fixed'][2] = True

    # p0['maxpars'][3] = amp_72_approx
    # p0['params'][3] = amp_72_approx * 0.5

    return param_dict[basename]


def measure_112_RMS(wave, csub):
    xmin = 11.9
    xmax = 12.1

    myrange = np.where((wave >= xmin) & (wave <= xmax))
    csub_mr = csub[myrange]
    rms = np.sqrt(np.mean(csub_mr**2))

    return rms  # , xmin, xmax


def fit_aliphatics(basename, wave, flux, fluxerr, rms, output_dir):

    def fit_straight_line(wave, flux, fluxerr):

        # Section 1: Fit Aliphatics
        # Define wavelength range of relevance
        lim = np.where((wave > 6.6) & (wave < 7.65))
        waveLim = wave[lim]
        fluxLim = flux[lim]
        errLim = fluxerr[lim]

        # Region where 7.2 is found
        lim2 = np.where((wave > 6.7) & (wave < 7.6))
        waveLim2 = wave[lim2]
        fluxLim2 = flux[lim2]
        # errLim2 = fluxerr[lim2]

        # Draw a straight line under 7.2 feature
        # Comment section out if no 7.2

        winDX = np.where((waveLim2 >= 7.) & (waveLim2 <= 7.2)
                         )  # Find the first ancor point
        winWave = waveLim2[winDX]
        winFlux = fluxLim2[winDX]
        anchor1Wave = winWave[np.nanargmin(winFlux)]
        anchor1Flux = np.nanmin(winFlux)

        winDX = np.where((waveLim2 >= 7.5))  # Find the second anchor point
        winWave = waveLim2[winDX]
        winFlux = fluxLim2[winDX]
        anchor2Wave = winWave[np.nanargmin(winFlux)]
        anchor2Flux = np.nanmin(winFlux)

        # Define the straight line from the anchor points
        x = np.array([anchor1Wave, anchor2Wave])
        y = np.array([anchor1Flux, anchor2Flux])
        StrLine = np.polyfit(x, y, deg=1)  # Fit straight line
        StrLineFit = StrLine[0] * waveLim2 + StrLine[1]

        # Plot straight line to check
        fig1, ax = plt.subplots()
        ax.plot(waveLim, fluxLim, '-r', lw=2)  # Make figure
        ax.errorbar(
            waveLim,
            fluxLim,
            errLim,
            color='r',
            ecolor='0.45',
            lw=2,
            elinewidth=1)
        ax.plot(waveLim2, StrLineFit, 'g-', label='7.2 Cont', lw=2)
        ax.plot(x, y, 'bo')

        ax.legend(loc=0, fontsize='small')
        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('Flux (W/m^2)')
        # ax.set_title(fnameStr + ' -- Line fit')
        ax.grid()
        ax.minorticks_on()

        ensure_exists(output_dir)
        pdf_filename = output_dir + basename + '_aliphatic_fit_1.pdf'
        print('Saved: ', pdf_filename)
        fig1.savefig(pdf_filename, format='pdf', bbox_inches='tight')

        plt.close()
        fig1.clear()

        if StrLine[0] > 0:  # Subtract straight line from data

            lim69 = np.where(waveLim < x[0])  # Create limits of subtraction
            lim72 = np.where((waveLim >= x[0]) & (waveLim <= x[1]))
            limOver = np.where(waveLim > x[1])

            flux69 = fluxLim[lim69]
            # Subtraction
            flux72 = fluxLim[lim72] - \
                (StrLine[0] * waveLim[lim72] + StrLine[1])
            fluxOver = fluxLim[limOver]

            fluxFull = []
            fluxFull = np.append(flux69, flux72)
            fluxFull = np.append(fluxFull, fluxOver)  # Create array

        else:
            fluxFull = fluxLim

        return waveLim, fluxFull, errLim

    def fit_gaussians(waveLim, fluxFull, errLim, fnameStr='temp_label'):

        # End of commented section if no 7.2
        # fluxFull = fluxLim # Comment if spectrum has 7.2 feature

        # Fit Gaussian functions to peaks
        # Change ngauss and parameters as needed

        # fitAli = multigaussfit(
        #     waveLim, fluxFull, ngauss=2, err=errLim,
        #     params=[0.12e-15,6.85,0.1,0.5e-16,7.23,0.05],
        #     limitedmin=[True,True,True,True,True,True],
        #     limitedmax=[True,True,True,True,True,True],
        #     minpars=[0.01e-18,6.8,0.03,0,7,0],
        #     maxpars=[1.5e-14,6.86,0.2,0.5e-15,7.3,0.06]
        # )

        fitAli = multigaussfit(
            waveLim, fluxFull, ngauss=2, err=errLim, params=[
                0.2e-14, 6.9, 0.1, 0.17e-14, 7.23, 0.05], limitedmin=[
                True, True, True, True, True, True], minpars=[
                0, 6.8, 0.04, 0, 7, 0], limitedmax=[
                    True, True, True, True, True, True], maxpars=[
                        0.5e-13, 7, 0.2, 0.1e-13, 7.25, 0.2])

        # Plot fit
        fig2, ax = plt.subplots()

        ax.plot(waveLim, fluxFull, '-r', label=fnameStr, lw=2)
        ax.errorbar(
            waveLim,
            fluxFull,
            errLim,
            color='r',
            ecolor='0.45',
            lw=2,
            elinewidth=1)
        ax.plot(waveLim, fitAli[1], '-g', label='Spectral Fit', lw=1.5)
        ax.fill_between(waveLim, fitAli[1], facecolor='green', alpha=0.15)
        ax.axhline(y=0, color='k', ls='-', zorder=-10, lw=2)

        ax.legend(loc=0, fontsize='small')
        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('Flux (W/m^2)')
        ax.set_title(fnameStr + ' -- Gaussian Fit')
        ax.grid()
        ax.minorticks_on()

        ensure_exists(output_dir)
        pdf_filename = output_dir + basename + '_aliphatic_fit_2.pdf'
        print('Saved: ', pdf_filename)
        fig2.savefig(pdf_filename, format='pdf', bbox_inches='tight')

        plt.close()
        fig2.clear()

        # print('Fit parameters for aliphatic features:')
        # print(fitAli[0])

        return fitAli

    def compute_aliphatic_fluxes(fitAli, waveLim, rms):

        # Calculate integrated flux of aliphatic features from fit
        Gauss69 = fitAli[0][0] * \
            np.exp(-(waveLim - fitAli[0][1])**2 / (2 * fitAli[0][2]**2))
        area69 = simps(Gauss69, waveLim)

        Gauss72 = fitAli[0][3] * \
            np.exp(-(waveLim - fitAli[0][4])**2 / (2 * fitAli[0][5]**2))
        area72 = simps(Gauss72, waveLim)

        err69 = compute_feature_uncertainty(
            fitAli[0][1], fitAli[0][2], waveLim, rms)
        err72 = compute_feature_uncertainty(
            fitAli[0][4], fitAli[0][5], waveLim, rms)

        # fitErr69 = np.sqrt((fitAli[2][0]/fitAli[0][0])**2 + \
        #     (fitAli[2][2]/fitAli[0][2])**2) * area69
        # fitErr72 = np.sqrt((fitAli[2][3]/fitAli[0][3])**2 + \
        #     (fitAli[2][5]/fitAli[0][5])**2) * area72

        errTot69 = err69  # + fitErr69
        errTot72 = err72  # + fitErr72

        # print('Integrated fluxes of aliphatics:')
        # print(area69, area72)

        SNR69 = area69 / errTot69
        SNR72 = area72 / errTot72

        # print('S/N of aliphatics: ', SNR69, SNR72)

        return area69, err69, SNR69, area72, err72, SNR72

    waveLim, fluxFull, errLim = fit_straight_line(wave, flux, fluxerr)

    fitAli = fit_gaussians(waveLim, fluxFull, errLim, fnameStr='temp_label')

    area69, err69, SNR69, area72, err72, SNR72 = \
        compute_aliphatic_fluxes(fitAli, waveLim, rms)

    return fitAli, waveLim, area69, SNR69, area72, SNR72


def fit_aromatics(basename, wave, flux, fluxerr, rms, output_dir):

    def fit_straight_line(wave, flux, fluxerr, fnameStr='temp_label'):

        # Section 2: Fit 7.7
        # Limits of feature - change as needed
        lim77 = np.where((wave >= 6.9) & (wave <= 9))
        waveLim77 = wave[lim77]
        fluxLim77 = flux[lim77]
        errLim77 = fluxerr[lim77]

        # Limit for 7.2 feature - change as needed
        lim2_a = np.where((wave > 6.9) & (wave < 7.45))
        waveLim2_a = wave[lim2_a]
        fluxLim2_a = flux[lim2_a]
        # errLim2_a = fluxerr[lim2_a]

        # Draw a straight line under 7.2 feature
        # Comment section out if no 7.2

        winDX77 = np.where((waveLim2_a >= 7.) & (waveLim2_a <= 7.2))
        winWave77 = waveLim2_a[winDX77]
        winFlux77 = fluxLim2_a[winDX77]
        anchor1Wave77 = winWave77[np.nanargmin(winFlux77)]
        anchor1Flux77 = np.nanmin(winFlux77)

        winDX77 = np.where((waveLim2_a >= 7.2))
        winWave77 = waveLim2_a[winDX77]
        winFlux77 = fluxLim2_a[winDX77]
        anchor2Wave77 = winWave77[np.nanargmin(winFlux77)]
        anchor2Flux77 = np.nanmin(winFlux77)

        # Define the straight line from the anchor points
        x77 = np.array([anchor1Wave77, anchor2Wave77])
        y77 = np.array([anchor1Flux77, anchor2Flux77])
        StrLine77 = np.polyfit(x77, y77, deg=1)  # Fit straight line
        StrLineFit77 = StrLine77[0] * waveLim2_a + StrLine77[1]

        # Comment out section if no 7.2
        fig3, ax = plt.subplots()  # Define figure
        ax.plot(waveLim77, fluxLim77, '-r', label=fnameStr, lw=2)
        ax.errorbar(
            waveLim77,
            fluxLim77,
            errLim77,
            color='r',
            ecolor='0.45',
            lw=2,
            elinewidth=1)
        # Plot straight line - comment out if no 7.2 feature
        ax.plot(waveLim2_a, StrLineFit77, 'g-', lw=2)

        ax.axhline(y=0, color='k', ls='-', zorder=-10, lw=2)
        ax.legend(loc=0, fontsize='small')
        ax.grid()
        ax.minorticks_on()

        ax.set_title(fnameStr + ' 7.7 complex')
        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('Flux (W/m^2)')

        ensure_exists(output_dir)
        pdf_filename = output_dir + basename + '_aromatic_fit_1.pdf'
        print('Saved: ', pdf_filename)
        fig3.savefig(pdf_filename, format='pdf', bbox_inches='tight')

        plt.close()
        fig3.clear()
        # End comments if no 7.2

        if StrLine77[0] > 0:  # Use if spectrum has 7.2
            # Create limits of subtraction
            lim69_a = np.where(waveLim77 < x77[0])
            lim72_a = np.where((waveLim77 >= x77[0]) & (waveLim77 <= x77[1]))
            limOver_a = np.where(waveLim77 > x77[1])

            flux69_a = fluxLim77[lim69_a]
            flux72_a = (
                StrLine77[0] *
                waveLim77[lim72_a] +
                StrLine77[1])  # Straight Line
            fluxOver_a = fluxLim77[limOver_a]

            fluxFull77 = []
            fluxFull77 = np.append(flux69_a, flux72_a)
            # Create array from wavelength subtraction
            fluxFull77 = np.append(fluxFull77, fluxOver_a)

        else:
            fluxFull77 = fluxLim77  # Use if spectrum has no 7.2

        return waveLim77, fluxLim77, errLim77, fluxFull77

    def fit_gaussians(waveLim77, fluxLim77, errLim77, fluxFull77,
                      fnameStr='temp_label'):

        # End commented section
        # fluxFull77 = fluxLim77 # Comment if spectrum has a 7.2 feature

        # Define feature:
        feature = np.where(
            (waveLim77 > 7.1) & (
                waveLim77 < 8.9))  # Change as needed

        # # Fit Gaussian
        # fit77 = multigaussfit(
        #     waveLim77[feature], fluxFull77[feature], ngauss=4,
        #     err=errLim77[feature],
        #     params=[5e-15,7.5,0.07,5.8e-15,7.75,0.06,
        #             1.2e-15,8.1,0.07,1.8e-15,8.65,0.07],
        #     limitedmin=[True,True,True,True,True,True,
        #                 True,True,True,True,True,True],
        #     limitedmax=[False,True,False,False,True,False,
        #                 False,True,False,True,True,False],
        #     minpars=[0,7.4,0.05,0,7.7,0.2e-16,8.1,
        #              0.01,0.7e-16,8.4,0.01],
        #     maxpars=[3e-14,7.7,0.1,3e-14,8.1,0.2,
        #              3e-14,8.4,0.2,3e-14,8.7,0.2]
        # )

        fit77 = multigaussfit(
            waveLim77[feature],
            fluxFull77[feature],
            ngauss=4,
            err=errLim77[feature],
            params=[
                1.25e-14,
                7.68,
                0.1,
                0.2e-14,
                7.95,
                0.06,
                3e-15,
                8.227557,
                0.15,
                0.3e-14,
                8.609484,
                0.08],
            limitedmin=[
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True],
            minpars=[
                0.2e-18,
                7.5,
                0.05,
                0.2e-18,
                7.75,
                0.01,
                0.2e-18,
                8.1,
                0.03,
                0.3e-18,
                8.5,
                0],
            limitedmax=[
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                False],
            maxpars=[
                2.3e-12,
                7.9,
                0.25,
                1.5e-12,
                8.2,
                0.25,
                8e-12,
                8.5,
                0.2,
                1e-11,
                8.7,
                0.12])

        # print('Fit parameters of the 7.7 micron complex:')
        # print(fit77[0])
        waveArr = np.arange(waveLim77[feature][0],
                            waveLim77[feature][-1], 0.0001)
        # Seperate Gaussian functions
        Gauss76 = fit77[0][0] * \
            np.exp(-(waveArr - fit77[0][1])**2 / (2 * fit77[0][2]**2))
        Gauss79 = fit77[0][3] * \
            np.exp(-(waveArr - fit77[0][4])**2 / (2 * fit77[0][5]**2))
        Gauss82 = fit77[0][6] * \
            np.exp(-(waveArr - fit77[0][7])**2 / (2 * fit77[0][8]**2))
        Gauss86 = fit77[0][9] * \
            np.exp(-(waveArr - fit77[0][10])**2 / (2 * fit77[0][11]**2))

        # err76 = compute_feature_uncertainty(
        #     fit77[0][1], fit77[0][2], waveLim77[feature], rms
        # )
        # err79 = compute_feature_uncertainty(
        #     fit77[0][4], fit77[0][5], waveLim77[feature], rms
        # )
        # err82 = compute_feature_uncertainty(
        #     fit77[0][7], fit77[0][8], waveLim77[feature], rms
        # )

        fluxFeat = Gauss76 + Gauss79 + Gauss82
        waveFeat = waveArr
        area77 = simps(fluxFeat, waveFeat)

        # fitErr77 = area77 * np.sqrt(
        #     (fit77[2][0]/fit77[0][0])**2 + (fit77[2][2]/fit77[0][2])**2 +
        #     (fit77[2][3]/fit77[0][3])**2 + (fit77[2][5]/fit77[0][5])**2 +
        #     (fit77[2][6]/fit77[0][6])**2 + (fit77[2][8]/fit77[0][8])**2
        # )
        # errTot77 = fitErr77

        # SNR77 = area77/errTot77

        # wave0 = 7.9  # Initial guess for central wavelength
        # errPercent = 1
        # count = 0
        '''
        # Define function to calculate difference in blue and red flux
        def areaEq(wave0):
            blue = np.where(waveFeat<=wave0)
            # Integration limits to find central wavelengths
            red = np.where(waveFeat>wave0)
            print '**************'
            area77b = simps(fluxFeat[blue], waveFeat[blue])
            area77r = simps(fluxFeat[red], waveFeat[red])
            return area77b - area77r

        lambdaC = fsolve(areaEq, wave0, xtol=1.5E-1)
        # Optimise difference in blue and red flux to find central wavelength
        print 'Central wavelength of 7.7 complex'
        print lambdaC

        while errPercent >= 0.01:
            count = count + 1
            #print count
            blue = np.where(waveFeat<=wave0)
            # Integration limits to find central wavelengths

            red = np.where(waveFeat>wave0)
            area77b = simps(fluxFeat[blue], waveFeat[blue])
            area77r = simps(fluxFeat[red], waveFeat[red])
            errPercent = np.absolute((area77b-area77r)/((area77b+area77r)/2))
            if area77b > area77r:
                wave0 = wave0 - 0.001
            elif area77r > area77b:
                wave0 = wave0 + 0.001
            else:
                continue
            #print area77b, area77r, wave0
            if count > 1000:
                break

        print 'error: ', errPercent
        print 'count: ', count
        lambdaC  = wave0
        print 'Central wavelength of 7.7 complex: ', lambdaC

        blue1 = np.where(waveFeat<=lambdaC)
        red1 = np.where(waveFeat>lambdaC)
        area77blue = simps(fluxFeat[blue1], waveFeat[blue1])
        area77red  = simps(fluxFeat[red1], waveFeat[red1])

        print 'Total integrated flux of 7.7 complex: ', area77
        #print 'SNR77: ', SNR77
        print 'Blue flux: ', area77blue
        print 'Red flux: ', area77red
        '''
        # Plot Gaussian fit
        fig4, ax = plt.subplots()  # Define figure

        ax.plot(waveLim77, fluxLim77, '-r', label=fnameStr, lw=2)
        ax.errorbar(
            waveLim77,
            fluxLim77,
            errLim77,
            color='r',
            ecolor='0.45',
            lw=2,
            elinewidth=1)
        ax.plot(waveLim77[feature], fit77[1], '-g', label='Spectral fit', lw=2)
        # ax.plot(waveLim2_a, StrLineFit77, 'b-', lw=2)
        # Plot straight line - comment out if no 7.2 feature

        # ax.plot(x77, y77, 'bo')
        # Straight line anchor points - comments out
        # if no 7.2 feature

        # Overplot individual Gaussian functions
        ax.plot(waveArr, Gauss76, '-g', lw=2)
        ax.fill_between(waveArr, Gauss76, facecolor='green', alpha=0.15)
        ax.plot(waveArr, Gauss79, '-g', lw=2)
        ax.fill_between(waveArr, Gauss79, facecolor='green', alpha=0.15)
        ax.plot(waveArr, Gauss82, '-g', lw=2)
        ax.fill_between(waveArr, Gauss82, facecolor='green', alpha=0.15)
        ax.plot(waveArr, Gauss86, '-g', lw=2)
        ax.fill_between(waveArr, Gauss86, facecolor='green', alpha=0.15)

        # ax.axvline(x=lambdaC, color='black', ls='-', lw=2)
        ax.axhline(y=0, color='b', ls='-', zorder=-10, lw=2)
        ax.legend(loc=0, fontsize='small')
        ax.grid()
        ax.minorticks_on()

        ax.set_title(fnameStr + ' 7.7 complex - fit')
        ax.set_xlabel(r'Wavelength ($\mu m$)')
        ax.set_ylabel('Flux ($W$/$m^2$)')
        axes = plt.gca()
        axes.set_xlim([7.1, 9])
        # axes.set_ylim([0,2e-15])

        ensure_exists(output_dir)
        pdf_filename = output_dir + basename + '_aromatic_fit_2.pdf'
        print('Saved: ', pdf_filename)
        fig4.savefig(pdf_filename, format='pdf', bbox_inches='tight')

        plt.close()
        fig4.clear()

        return fit77, area77, feature

    waveLim77, fluxLim77, errLim77, fluxFull77 = \
        fit_straight_line(wave, flux, fluxerr)

    fit77, area77, feature = fit_gaussians(
        waveLim77, fluxLim77, errLim77, fluxFull77)

    return waveLim77, fit77, area77, feature


def fit_all(basename, wave, flux, fluxerr, rms, output_dir):
    """Fit Gaussians and straight line at the same time or something. Or maybe
    no straight line."""
    def param_constraints_OK(p0, line, index):
        # Test if any parameter hitting min/max of constrained range.

        def nums_equal(num1, num2, acc=0.01):
            """Returns True if numbers are equal within some accuracy."""
            if np.abs(num1 - num2)  < acc:
                return False
            else:
                return True

        # Line position.
        pindex = index * 3 + 1
        fixed_position = p0['fixed'][pindex]

        if not fixed_position:
            limited_min = p0['limitedmin'][pindex]
            limited_max = p0['limitedmax'][pindex]
            if limited_min:
                if not nums_equal(p0['minpars'][pindex], line['position']):
                    print('Hitting minimum line position.')
                    return False
            if limited_max:
                if not nums_equal(p0['maxpars'][pindex], line['position']):
                    print('Hitting maximum line position.')
                    return False

        # Line sigma.
        pindex = index * 3 + 2
        fixed_sigma = p0['fixed'][pindex]

        if not fixed_sigma:
            limited_min = p0['limitedmin'][pindex]
            limited_max = p0['limitedmax'][pindex]
            if limited_min:
                if not nums_equal(p0['minpars'][pindex], line['sigma']):
                    print('Hitting minimum line sigma.')
                    return False
            if limited_max:
                if not nums_equal(p0['maxpars'][pindex], line['sigma']):
                    print('Hitting maximum line sigma.')
                    return False

        return True

    def fit_4gauss_2lines(wave, flux, fluxerr, trim, trim_wide):

        # Multigauss fit. Intensity, center, sigma (or FWHM?).
        yscale = flux[trim]
        guess = np.nanmax(yscale)
        yfit = multigaussfit(
            wave[trim], flux[trim], ngauss=4, err=fluxerr[trim],
            params=[
                # 0.2e-14,  6.90, 0.10,
                # 0.2e-14,  7.23, 0.05,
                guess / 2.,  7.68, 0.10,
                guess,  7.95, 0.06,
                guess / 2.,  8.23, 0.15,
                guess / 2.,  8.61, 0.08
            ],
            limitedmin=[True] * 12,
            limitedmax=[True] * 12,
            minpars=[
                # 0, 6.8, 0.04,
                # 0, 7, 0,
                0, 7.5, 0.05,
                0, 7.75, 0.01,
                0, 8.1, 0.03,
                0, 8.5, 0
            ],
            maxpars=[
                # 0.5e-13, 7, 0.2,
                # 0.1e-13, 7.25, 0.2,
                guess, 7.9, 0.25,
                guess, 8.2, 0.25,
                guess, 8.5, 0.2,
                guess, 8.7, 0.12
            ])

        g76 = onedgaussian(wave, 0, yfit[0][0], yfit[0][1], yfit[0][2])
        g78 = onedgaussian(wave, 0, yfit[0][3], yfit[0][4], yfit[0][5])
        g82 = onedgaussian(wave, 0, yfit[0][6], yfit[0][7], yfit[0][8])
        g86 = onedgaussian(wave, 0, yfit[0][9], yfit[0][10], yfit[0][11])
        model = g76 + g78 + g82 + g86

        # Multigauss fit. Intensity, center, sigma (or FWHM?).
        resid = flux - model
        yfit2 = multigaussfit(
            wave, resid, ngauss=2,
            params=[
                np.nanmax(resid) / 2.,  6.88, 0.10,
                np.nanmax(resid) / 2.,  7.23, 0.05,
            ],
            limitedmin=[True] * 6,
            limitedmax=[True] * 6,
            minpars=[
                0, 6.8, 0.04,
                0, 7, 0,
            ],
            maxpars=[
                np.nanmax(resid), 7, 0.2,
                np.nanmax(resid), 7.30, 0.2,
            ])

        line69 = onedgaussian(wave, 0, yfit2[0][0], yfit2[0][1], yfit2[0][2])
        line72 = onedgaussian(wave, 0, yfit2[0][3], yfit2[0][4], yfit2[0][5])

        wpeak = {
            '69': yfit2[0][1],
            '72': yfit2[0][4],
        }

        return g76, g78, g82, g86, line69, line72, model, yfit, yfit2, wpeak

    def fit_6gauss(wave, flux, fluxerr, trim, basename):

        # Initial parameters and constraints.
        yscale = flux[trim]
        guess = np.nanmax(yscale)
        p_init = params_6gauss(basename, guess)

        # If fluxerr[trim] has zeroes, don't use errors for now?
        if 0 in fluxerr[trim]:
            errpass = None
        else:
            errpass = fluxerr[trim]

        if basename in ['J052043_CWsub', 'SMPSMC006_CWsub']:
            errpass = None

        # Multigauss fit. Intensity, center, sigma (or FWHM?).
        yfit = multigaussfit(
            wave[trim], flux[trim], ngauss=6, err=errpass,
            params=p_init['params'],
            limitedmin=p_init['limitedmin'],
            limitedmax=p_init['limitedmax'],
            fixed=p_init['fixed'],
            minpars=p_init['minpars'],
            maxpars=p_init['maxpars']
            )

        # Save results.
        features = ('line69', 'line72', 'g76', 'g78', 'g82', 'g86')
        keys = ('scale_factor', 'position', 'sigma')
        results = {}

        for i in range(6):
            fit_params = (yfit[0][3 * i:3 * i + 3])
            results[features[i]] = dict(zip(keys, fit_params))
            results[features[i]]['wave'] = wave
            results[features[i]]['spectrum'] = onedgaussian(
                wave, 0, *fit_params)
            results[features[i]]['integrated_flux'] = simps(
                results[features[i]]['spectrum'], results[features[i]]['wave'])

        # if basename == 'IRAS15482_CWsub':
        #     st()

        return yfit, results, p_init

    def fit_6gauss_lmfit(wave, flux, fluxerr, trim):

        # define objective function: returns the array to be minimized
        def fcn2min(params, x, data):
            """Model a decaying sine wave and subtract data."""
            amp1 = params['amp1']
            amp2 = params['amp2']
            amp3 = params['amp3']
            amp4 = params['amp4']
            amp5 = params['amp5']
            amp6 = params['amp6']

            pos1 = params['pos1']
            pos2 = params['pos2']
            pos3 = params['pos3']
            pos4 = params['pos4']
            pos5 = params['pos5']
            pos6 = params['pos6']

            fwhm1 = params['fwhm1']
            fwhm2 = params['fwhm2']
            fwhm3 = params['fwhm3']
            fwhm4 = params['fwhm4']
            fwhm5 = params['fwhm5']
            fwhm6 = params['fwhm6']

            model1 = amp1 * np.exp(-(x - pos1)**2 / (2.0 * to_sigma(fwhm1)**2))
            model2 = amp2 * np.exp(-(x - pos2)**2 / (2.0 * to_sigma(fwhm2)**2))
            model3 = amp3 * np.exp(-(x - pos3)**2 / (2.0 * to_sigma(fwhm3)**2))
            model4 = amp4 * np.exp(-(x - pos4)**2 / (2.0 * to_sigma(fwhm4)**2))
            model5 = amp5 * np.exp(-(x - pos5)**2 / (2.0 * to_sigma(fwhm5)**2))
            model6 = amp6 * np.exp(-(x - pos6)**2 / (2.0 * to_sigma(fwhm6)**2))

            model = model1 + model2 + model3 + model4 + model5 + model6

            return model - data

        # Initial parameters and constraints.
        yscale = flux[trim]
        scale_fac = np.nanmedian(yscale)

        # Scale to be near unity for computational reasons?
        x = wave[trim]
        data = flux[trim] / scale_fac

        # Guess for parameters.
        gg = np.nanmax(data)

        # create a set of Parameters
        params = Parameters()
        params.add('amp1', value=gg/2., max=gg, min=gg/30.)
        params.add('amp2', value=gg/2., max=gg, min=0)
        params.add('amp3', value=gg/2., max=gg, min=gg/30.)
        params.add('amp4', value=gg/1., max=gg, min=gg/30.)
        params.add('amp5', value=gg/2., max=gg, min=0)
        params.add('amp6', value=gg/2., max=gg, min=gg/30.)

        params.add('pos1', value=6.89, min=6.82, max=6.96)
        params.add('pos2', value=7.25, min=7.15, max=7.30)
        params.add('pos3', value=7.55, min=7.45, max=7.65)
        params.add('pos4', value=7.87, min=7.77, max=7.97)
        params.add('pos5', value=8.25, min=8.15, max=8.35)
        params.add('pos6', value=8.59, min=8.49, max=8.69)

        params.add('fwhm1', value=0.15, min=0.060, max=0.21)
        params.add('fwhm2', value=0.12, min=0.050, max=0.15)
        params.add('fwhm3', value=0.44, min=0.315, max=0.565)
        params.add('fwhm4', value=0.40, min=0.275, max=0.525)
        params.add('fwhm5', value=0.29, min=0.165, max=0.415)
        params.add('fwhm6', value=0.36, min=0.235, max=0.485)

        # do fit, here with leastsq model
        minner = Minimizer(fcn2min, params, fcn_args=(x, data))
        result = minner.minimize()

        # calculate final result
        final = data + result.residual

        # write error report
        report_fit(result)

        # plt.plot(x, data, 'k+')
        # plt.plot(x, final, 'r')
        # plt.show()

        # # If fluxerr[trim] has zeroes, don't use errors for now?
        # if 0 in fluxerr[trim]:
        #     errpass = None
        # else:
        #     errpass = fluxerr[trim]

        # # Multigauss fit. Intensity, center, sigma (or FWHM?).
        # yfit = multigaussfit(
        #     wave[trim], flux[trim], ngauss=6, err=errpass,
        #     params=p0['params'],
        #     limitedmin=p0['limitedmin'],
        #     limitedmax=p0['limitedmax'],
        #     fixed=p0['fixed'],
        #     minpars=p0['minpars'],
        #     maxpars=p0['maxpars']
        #     )

        # # Save results.
        # features = ('line69', 'line72', 'g76', 'g78', 'g82', 'g86')
        # keys = ('scale_factor', 'position', 'sigma')
        # results = {}

        # for i in range(6):
        #     fit_params = (yfit[0][3 * i:3 * i + 3])
        #     results[features[i]] = dict(zip(keys, fit_params))
        #     results[features[i]]['wave'] = wave
        #     results[features[i]]['spectrum'] = onedgaussian(
        #         wave, 0, *fit_params)
        #     results[features[i]]['integrated_flux'] = simps(
        #         results[features[i]]['spectrum'], results[features[i]]['wave'])

        return None, None, None

    print(basename)

    fit4 = False

    if fit4:

        # Only fit 7-9 micron zone.
        trim = np.where((wave > 7.3) & (wave < 9.2))
        trim_wide = np.where((wave >= 6.0) & (wave <= 10))

        # Return fit.
        g76, g78, g82, g86, line69, line72, model, yfit, yfit2, wpeak = \
            fit_4gauss_2lines(wave, flux, fluxerr, trim, trim_wide)

        flux69 = simps(line69, wave)
        flux72 = simps(line72, wave)

        # Plot results.
        fig = plt.figure()
        gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace=0.3)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        # Upper panel.
        gsum = g76 + g78 + g82
        ax1.plot(wave[trim_wide], flux[trim_wide], label='Data')
        ax1.plot(wave[trim_wide], model[trim_wide],
                 label='Model (Flux: {:.2e} W/m^2)'.format(simps(gsum, wave)))
        for index, item in enumerate((g76, g78, g82, g86)):
            ax1.fill_between(
                wave[trim_wide],
                wave[trim_wide] * 0,
                item[trim_wide],
                lw=0.5,
                alpha=0.3)
        ax1.axhline(y=0, ls='--', lw=0.5, color='k')
        ax1.legend(loc=0)
        xmin, xmax = ax1.get_xlim()

        # Lower panel.
        trim_wide2 = np.where((wave >= 6.5) & (wave <= 10))
        ax2.plot(
            wave[trim_wide2],
            flux[trim_wide2] -
            model[trim_wide2],
            label='Residual from 4gauss')
        label69 = \
            '6.9 ({:.2f} µm, Flux: {:.2e} W/m^2)'.format(wpeak['69'], flux69)
        label72 = \
            '7.2 ({:.2f} µm, Flux: {:.2e} W/m^2)'.format(wpeak['72'], flux72)
        ax2.fill_between(wave, wave * 0, line69, alpha=0.3, label=label69)
        ax2.fill_between(wave, wave * 0, line72, alpha=0.3, label=label72)
        ax2.axhline(y=0, ls='--', lw=0.5, color='k')
        ax2.legend(loc=0)
        ax2.set_xlim(xmin, xmax)

        # Save.
        savename = output_dir + 'fullspec/' + basename + '_test.pdf'
        fig.savefig(savename, bbox_inches='tight')
        print('Saved: ', savename)
        plt.close()
        fig.clear()

    else:

        # Only fit 7-9 micron zone.
        trim = np.where((wave > 6.0) & (wave < 10))

        # Try 6-components.
        yfit, results, p0 = fit_6gauss(wave, flux, fluxerr, trim, basename)

        # # Try 6, with LMFIT!
        # yfit2, results2, p02 = fit_6gauss_lmfit(wave, flux, fluxerr, trim)
        # st()

        # Plot results.
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace=0.3)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        ##############################
        # Upper panel.
        ##############################

        flux77 = sum([results[x]['integrated_flux']
                      for x in ('g76', 'g78', 'g82')])
        spec77 = results['g76']['spectrum'] + results['g78']['spectrum'] + \
            results['g82']['spectrum']

        centroid77 = np.sum(spec77 * wave) / np.sum(spec77)
        model_label = \
            r'Model (g1-3: {:.2f} µm, {:.2e} W/m$^2$)'.format(centroid77,
                                                              flux77)
        ax1.errorbar(wave[trim], flux[trim], yerr=fluxerr[trim], label='Data')
        # ax1.plot(wave[trim], flux[trim], label='Data')

        ax1.plot(wave[trim], yfit[1], label=model_label, zorder=1000)
        for index, key in enumerate(results):
            ax1.fill_between(wave[trim], wave[trim] * 0,
                             results[key]['spectrum'][trim],
                             lw=0.5, alpha=0.3)
        ax1.axvline(x=centroid77, color='k', ls='-', lw=0.5)
        ax1.axhline(y=0, ls='--', lw=0.5, color='k')
        ax1.axvline(x=6.9, color='k', ls='-', lw=0.5)
        ax1.axvline(x=7.25, color='k', ls='-', lw=0.5)
        ax1.legend(loc=0, fontsize=8)
        xmin, xmax = ax1.get_xlim()

        ##############################
        # Lower panel.
        ##############################

        f72_69 = results['line72']['integrated_flux'] / results['line69']['integrated_flux']

        ax2.plot(wave[trim], flux[trim] - yfit[1], label='Residuals (7.2/6.9 = {})'.format(quant_str(f72_69, precision="0.01")))
        ax2.axvline(x=6.9, color='k', ls='-', lw=0.5)
        ax2.axvline(x=7.25, color='k', ls='-', lw=0.5)

        param_OK_list = [True]
        for index, key in enumerate(results):
            line = results[key]
            label = '{:.2f} µm, {:.2e} W/m^2, FWHM={:.2f} µm'.format(
                line['position'], line['integrated_flux'],
                to_fwhm(line['sigma'])
            )
            param_OK_list.append(param_constraints_OK(p0, line, index))
            ax2.fill_between(wave[trim], wave[trim] * 0,
                             results[key]['spectrum'][trim],
                             lw=0.5, alpha=0.3, label=label)
        ax2.axhline(y=0, ls='--', lw=0.5, color='k')
        mylegend = ax2.legend(loc=0, fontsize=8)

        for index, text in enumerate(mylegend.get_texts()):
            if not param_OK_list[index]:
                text.set_color("red")

        # Save.
        savename = output_dir + 'fullspec/' + basename + '_6gauss.pdf'
        fig.savefig(savename, bbox_inches='tight')
        print('Saved: ', savename)
        plt.close()
        fig.clear()


        # Insert the 7.7 results.
        results['pah77'] = {
            'flux': flux77,
            'centroid': centroid77,
        }

        pkl_name = output_dir + 'numeric/' + basename + '.pkl'
        # Record results to disk.
        with open(pkl_name, 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
            print('Saved: ', pkl_name)

    return


def measure_112(basename, wave, flux, fluxerr, rms, output_dir,
                fnameStr='temp_label'):

    lim11 = np.where((wave >= 10.8) & (wave <= 12))
    waveLim11 = wave[lim11]
    fluxLim11 = flux[lim11]
    errLim11 = fluxerr[lim11]

    feature11 = np.where((waveLim11 >= 11.1) & (
        waveLim11 <= 11.85))  # Define actual feature
    # Start at 11.1, change end point as needed

    area11 = simps(fluxLim11[feature11], waveLim11[feature11])  # Integrate

    # print('Integrated flux of 11.2 feature: ', area11)

    fig5, ax = plt.subplots()  # Define figure

    ax.plot(waveLim11, fluxLim11, '-r', label=fnameStr, lw=2)
    ax.errorbar(
        waveLim11,
        fluxLim11,
        errLim11,
        color='r',
        ecolor='0.45',
        lw=2,
        elinewidth=1)
    ax.fill_between(
        waveLim11[feature11],
        fluxLim11[feature11],
        facecolor='red',
        alpha=0.15)
    ax.axhline(y=0, color='k', ls='-', zorder=-10, lw=2)

    ax.legend(loc=0, fontsize='small')
    ax.set_title(fnameStr + ' -- 11.2 feature')
    ax.set_xlabel('Wavelength (microns)')
    ax.set_ylabel('Flux (W/m^2)')
    ax.grid()
    ax.minorticks_on()

    ensure_exists(output_dir)
    pdf_filename = output_dir + basename + '_11.pdf'
    print('Saved: ', pdf_filename)
    fig5.savefig(pdf_filename, format='pdf', bbox_inches='tight')

    plt.close()
    fig5.clear()

    return area11


def save_fit_parameters(output_dir, results):

    ensure_exists(output_dir)

    fitAli, fit77, basename, waveLim, waveLim77, area69, \
        area72, area77, area11, SNR69, SNR72, feature = results

    fnameStr = basename

    # Save all fit parameters in one file
    arrParamsID = np.append(fitAli[0] * 0, fit77[0] * 0 + 1)
    arrParams = np.append(fitAli[0], fit77[0])
    # arrParamsErr = np.append(fitAli[2], fit77[2])

    txt_filename = output_dir + fnameStr + '_fitParams.txt'
    print('Saved: ', txt_filename)
    np.savetxt(
        txt_filename,
        np.c_[
            arrParamsID,
            arrParams],
        delimiter=',',
        header='Gaussian fit parameters\n col1: ID - 0=aliphatic, '
               '1=7.7 complex, col2: parameters, col3: fit error')
    # np.savetxt(fnameStr + '_fitParams.txt',
    # np.c_[arrParamsID, arrParams, arrParamsErr], delimiter=',',
    # header='Gaussian fit parameters\n col1: ID - 0=aliphatic, 1=7.7 complex,
    # col2: parameters, col3: fit error')

    # Save all fit models in one file
    arrID = np.append(fitAli[1] * 0, fit77[1] * 0 + 1)
    arrWave = np.append(waveLim, waveLim77[feature])
    arrFluxDensity = np.append(fitAli[1], fit77[1])

    txt_filename = output_dir + fnameStr + '_fitModel.txt'
    print('Saved: ', txt_filename)
    np.savetxt(
        txt_filename,
        np.c_[
            arrID,
            arrWave,
            arrFluxDensity],
        delimiter=',',
        header='Full model fit\n col1: ID - 0=aliphatic, 1=7.7 complex, '
               'col2: wavelength, col3: flux density')

    # arrIntegratedFluxes = np.array([area69, area72, area77, area77blue,
    # area77red, area11, lambdaC]) # Save all integrated fluxes in one file
    # Save all integrated fluxes in one file
    # arrIntegratedFluxes = np.array([area69, area72, area77, area11])

    txt_filename = output_dir + fnameStr + '_intFlux.txt'
    print('Saved: ', txt_filename)
    np.savetxt(
        txt_filename,
        np.c_[
            area69,
            area72,
            area77,
            area11],
        delimiter=',',
        header='Integrated fluxes of features\n col1: 6.9 microns, '
               'col2: 7.2 microns, col3: total 7.7 complex, col4: blue 7.7, '
               'col5: red 7.7, col6: 11.2 feature, '
               'col7: central wavelength of 7.7 (microns)')

    txt_filename = output_dir + fnameStr + '_SNR.txt'
    print('Saved: ', txt_filename)
    np.savetxt(
        txt_filename,
        np.c_[
            SNR69,
            SNR72],
        delimiter=',',
        header='col1:SNR69, col2: SNR72, col3: SNR77')

    txt_filename = output_dir + fnameStr + 'Full.txt'
    workFile = open(txt_filename, 'w')  # Write all data into single file
    workFile.write(fnameStr + 'Fitting and integrated flux data\n\n')

    workFile.write('Section 1: Aliphatic features\n')
    workFile.write(
        'Gaussian fitting parameters for 6.9 and 7.2 micron features\n')
    workFile.write(str(fitAli[0]) + '\n')
    workFile.write('Errors on aliphatic fitting parameters\n')
    workFile.write(str(fitAli[2]) + '\n')
    workFile.write('Fit model - aliphatic features\n')
    workFile.write(str(waveLim) + '\n' + str(fitAli[1]) + '\n\n')
    workFile.write('Integrated fluxes of aliphatic features\n')
    workFile.write('6.9 micron feature ' + str(area69) + '\n')
    workFile.write('7.2 micron feature ' + str(area72) + '\n')
    workFile.write('S/N of 6.9: ' + str(SNR69) + '\n')
    workFile.write('S/N of 7.2: ' + str(SNR72) + '\n\n')

    workFile.write('Section 2: 7.7 micron complex\n')
    workFile.write('Gaussian fitting parameters for 7.7 micron complex\n')
    workFile.write(str(fit77[0]) + '\n')
    workFile.write('Errors on fitting parameters\n')
    workFile.write(str(fit77[2]) + '\n')
    workFile.write('Fit model - 7.7 micron complex\n')
    workFile.write(str(waveLim77[feature]) + '\n' + str(fit77[1]) + '\n')
    workFile.write('Integrated fluxes\n')
    workFile.write('Total integrated flux of complex: ' + str(area77) + '\n')
    # workFile.write('Blue flux: ' + str(area77blue) + '\n')
    # workFile.write('Red flux: ' + str(area77red) + '\n')
    # workFile.write('SNR77: ' + str(SNR77) + '\n')
    # workFile.write('Central wavelength of 7.7 complex:' + str(lambdaC) +
    # '\n\n')

    workFile.write('Section 3: 11.2 micron feature\n')
    workFile.write('Integrated flux of 11.2 micron feature:\n')
    workFile.write(str(area11) + '\n')

    workFile.close()
    print('Saved: ', txt_filename)

    return
