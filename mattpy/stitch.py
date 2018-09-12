#!/usr/bin/env python3
"""
stitch.py

Functions for stitching Spitzer modules (SL, LL).
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp


def stitch_SL_LL(SLwave, LLwave, SLflux, LLflux, SLfluxerr, LLfluxerr,
                 source='', cpos=0, stitch_savename='', saveplot=False,
                 savespec=False, force_scale1=False):

    # Find SL and LL overlap.
    rxLL = np.where((LLwave <= SLwave[-1]))
    rxSL = np.where((SLwave >= LLwave[0]))
    overlapWaveLL = LLwave[rxLL]
    overlapFluxLL = LLflux[rxLL]
    # meanLL = np.mean(overlapFluxLL)
    overlapWaveSL = SLwave[rxSL]
    overlapFluxSL = SLflux[rxSL]
    # meanSL = np.mean(overlapFluxSL)
    # scaleLL = meanLL / meanSL

    # Compute scale factor.
    newWave = np.arange(overlapWaveSL[0], overlapWaveLL[-1], 0.001)
    splLL = interp.splrep(overlapWaveLL, overlapFluxLL, k=1)
    splSL = interp.splrep(overlapWaveSL, overlapFluxSL, k=1)
    # splineWaveLL = newWave
    splineFluxLL = interp.splev(newWave, splLL)
    # splineWaveSL = newWave
    splineFluxSL = interp.splev(newWave, splSL)
    nscaleLL = np.mean(splineFluxSL / splineFluxLL)
    # nscale2 = 0.4

    if force_scale1:
        nscaleLL = 1

    # Stitch.
    WaveLL = LLwave
    FluxLL = LLflux
    FluxLLerr = LLfluxerr
    cdxSL = np.where(SLwave <= LLwave[0])
    cutWaveSL = SLwave[cdxSL]
    cutFluxSL = SLflux[cdxSL]
    cutFluxSLerr = SLfluxerr[cdxSL]
    finalWave = np.concatenate((cutWaveSL, WaveLL))
    finalFlux = np.concatenate((cutFluxSL, FluxLL * nscaleLL))
    finalFluxerr = np.concatenate((cutFluxSLerr, FluxLLerr * nscaleLL))

    if saveplot:
        plt.errorbar(SLwave, SLflux, SLfluxerr,
                     label='SL', lw=1, ecolor='0.7', capsize=0)
        plt.errorbar(LLwave, LLflux, LLfluxerr,
                     label='LL', lw=1, ecolor='0.7', capsize=0)
        plt.errorbar(LLwave, LLflux * nscaleLL, LLfluxerr**nscaleLL,
                     label='LL (scaled)', lw=1, ecolor='0.7', capsize=0)
        # plt.errorbar(finalWave, finalFlux, finalFluxerr, label='final')
        plt.axvline(x=newWave[0], ls='-', color='k')
        plt.axvline(x=newWave[-1], ls='-', color='k')

        plt.legend(loc=0)
        plt.title(str(source + ' - cpos ' + str(cpos) + ' - ' + str(nscaleLL)))
        plt.savefig(stitch_savename, format='pdf', bbox_inches='tight')
        # plt.show()
        plt.close()

    if savespec:
        dasave = stitch_savename.split('.pdf')[0] + '_LLtoSL_factor.txt'
        np.savetxt(dasave, [nscaleLL], delimiter=',',
                   header='scale factor, LL stitch to SL (multiply by this)')

    return finalWave, finalFlux, finalFluxerr, cutWaveSL, WaveLL
