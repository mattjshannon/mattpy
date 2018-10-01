#!/usr/bin/env python3
"""

measure_features.py

Measure the PAH and atomic emission bands in a spectrum.

NOTE:
requires the pip packages...
pip install astropy-helpers
pip install https://github.com/keflavich/gaussfitter/archive/master.zip
"""

import numpy as np
import scipy as sp
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from mattpy import continuum_fluxes as cf
from mattpy import mpfit
from mattpy.utils import norm, get_home_dir, \
    compute_feature_uncertainty, to_sigma

from gaussfitter import onedgaussian, onedgaussfit, multigaussfit


def rms_bounds(module):
    if module == 'SL':
        wlow = 10.
        whigh = 10.4
    elif module == 'LL':
        wlow = 27
        whigh = 27.5
    else:
        raise SystemExit("module unrecognized.")

    return wlow, whigh


def jy(f, w):
    return w**2 * f / 3e-12


def si(f, w):
    return 3e-12 * f / w**2


def measureFluxes(waveL, csubL, csubLerr, waveS, csubS, csubSerr, cpos):

    def measure_112_RMS(wave, csub):
        xmin, xmax = rms_bounds('SL')
        myrange = np.where((wave >= xmin) & (wave <= xmax))
        csub_mr = csub[myrange]
        rms = np.sqrt(np.nanmean(csub_mr**2))

        return rms

    def measure_25_RMS(wave, csub):
        xmin, xmax = rms_bounds('LL')
        myrange = np.where((wave >= xmin) & (wave <= xmax))
        csub_mr = csub[myrange]
        rms = np.sqrt(np.nanmean(csub_mr**2))

        return rms

    def fit_110(wave, csubSI, csuberrSI, rms):

        # print("-----------")
        # print("FITTING 11.0")
        # print("-----------")

        is_SH_data = 0

        wave_feat = wave
        flux_feat = csubSI
        fluxerr_feat = csuberrSI

        if is_SH_data == 0:
            my_ind = np.where((wave_feat > 10.6) & (wave_feat < 11.8))
            w = wave_feat[my_ind]
            f = flux_feat[my_ind]
            e = fluxerr_feat[my_ind]

            amp_guess = max(f[np.where(w < 11.1)])
            params = [10.989, to_sigma(0.1538), amp_guess,
                      11.258, to_sigma(0.236), max(f)],
            limitedmin = [True, True, True,
                          True, True, True],
            limitedmax = [True, True, False,
                          True, True, False],
            minpars = [10.97, to_sigma(0.12), amp_guess / 20.,
                       11.2, to_sigma(0.226), 0.],
            maxpars = [11.05, to_sigma(0.19), 0.,
                       11.3, to_sigma(0.246), 0.],

            yfit = multigaussfit(w, f, err=e, ngauss=2,
                                 params=params,
                                 limitedmin=limitedmin,
                                 limitedmax=limitedmax,
                                 minpars=minpars,
                                 maxpars=maxpars,
                                 quiet=True,
                                 shh=True)

        else:
            my_ind = np.where((wave_feat > 10.6) & (wave_feat < 11.3))
            w = wave_feat[my_ind]
            f = flux_feat[my_ind]
            e = fluxerr_feat[my_ind]

            amp_guess = max(f[np.where(w < 11.1)])
            params = [10.989, to_sigma(0.1538), amp_guess,
                      11.258, to_sigma(0.236), max(f)],
            limitedmin = [True, True, True,
                          True, True, True],
            limitedmax = [True, True, False,
                          True, True, False],
            minpars = [10.97, to_sigma(0.02), amp_guess / 20.,
                       11.2, to_sigma(0.16), 0.],
            maxpars = [11.05, to_sigma(0.19), 0.,
                       11.3, to_sigma(0.246), 0.],

            yfit = multigaussfit(w, f, err=e, ngauss=2,
                                 params=params,
                                 limitedmin=limitedmin,
                                 limitedmax=limitedmax,
                                 minpars=minpars,
                                 maxpars=maxpars,
                                 quiet=True,
                                 shh=True)

        y1r = onedgaussian(w, 0, yfit[0][2], yfit[0][0], yfit[0][1])
        # y2r = onedgaussian(w, 0, yfit[0][5], yfit[0][3], yfit[0][4])

        # 11.0
        small_gauss_area = sp.integrate.trapz(y1r, x=w)
        position = yfit[0][0]
        sigma = yfit[0][1]
        amp = yfit[0][2]
        # position_err = yfit[2][0]
        sigma_err = yfit[2][1]
        amp_err = yfit[2][2]
        small_gauss_area_err = np.sqrt(
            (amp_err / amp)**2 + (sigma_err / sigma)**2) * small_gauss_area
        myrange = [position - (3. * sigma), position + (3. * sigma)]
        N = np.where((wave_feat >= myrange[0]) & (wave_feat <= myrange[1]))[0]
        dl = w[1] - w[0]
        measured_flux_noise110 = (rms * np.sqrt(len(N)) * dl * 2)

        # 11.2
        # gauss_area = sp.integrate.trapz(y2r, x=w)
        position = yfit[0][3]
        sigma = yfit[0][4]
        amp = yfit[0][5]
        # position_err = yfit[2][3]
        sigma_err = yfit[2][4]
        amp_err = yfit[2][5]
        # gauss_area_err = np.sqrt(
        #     (amp_err / amp)**2 + (sigma_err / sigma)**2) * gauss_area
        myrange = [position - (3. * sigma), position + (3. * sigma)]
        N = np.where((wave_feat >= myrange[0]) & (wave_feat <= myrange[1]))[0]
        dl = w[1] - w[0]
        measured_flux_noise112 = (rms * np.sqrt(len(N)) * dl * 2)

        ######################################
        new_flux_feat = f - y1r
        trap_flux_high = sp.integrate.trapz(new_flux_feat + e, x=w)
        trap_flux_low = sp.integrate.trapz(new_flux_feat - e, x=w)
        trap_flux = np.mean([trap_flux_high, trap_flux_low])
        # trap_flux_std = 0.67 * np.std([trap_flux_high, trap_flux_low])
        ######################################

        FINAL_112_FLUX = trap_flux  # full_trap_flux - small_gauss_area
        FINAL_112_FLUX_ERR = measured_flux_noise112  # + trap_flux_std
        FINAL_110_FLUX = small_gauss_area
        FINAL_110_FLUX_ERR = small_gauss_area_err + measured_flux_noise110

        return FINAL_110_FLUX, FINAL_110_FLUX_ERR, \
            FINAL_112_FLUX, FINAL_112_FLUX_ERR, myrange

    def fit_127(wave, csubSI, csuberrSI):

        def make_127_profile(home_dir, valmin=12.22, valmax=13.2):

            spectrum_file = \
                'Dropbox/code/Python/fitting_127/average_pah_spectrum.dat'
            data = np.loadtxt(str(home_dir + spectrum_file))
            data = data.T

            n = np.where((data[0] > valmin) & (data[0] < valmax))[0]
            wave = data[0][n]
            flux = data[1][n]
            fluxerr = data[2][n]
            band = data[3][n]

            flux = flux - 0.05  # offset to zero it

            for i in range(len(wave)):
                if i != len(wave) - 1:
                    if wave[i + 1] - wave[i] <= 0:
                        # print i, wave[i]
                        cut_it = i

            w1 = wave[:cut_it + 1]
            f1 = flux[:cut_it + 1]
            fe1 = fluxerr[:cut_it + 1]
            b1 = band[:cut_it + 1]

            w2 = wave[cut_it + 1:]
            f2 = flux[cut_it + 1:]
            fe2 = fluxerr[cut_it + 1:]
            b2 = band[cut_it + 1:]

            c1 = np.where(w1 < w2[0])[0]
            w1 = w1[c1]
            f1 = f1[c1]
            fe1 = fe1[c1]
            b1 = b1[c1]

            # tie together
            wave = np.concatenate((w1, w2), axis=0)
            flux = np.concatenate((f1, f2), axis=0)
            fluxerr = np.concatenate((fe1, fe2), axis=0)
            band = np.concatenate((b1, b2), axis=0)
            # flux -= np.amin(flux)

            wave1 = np.reshape(wave, (len(wave), 1))
            flux1 = np.reshape(flux, (len(flux), 1))
            fluxerr1 = np.reshape(fluxerr, (len(fluxerr), 1))
            band1 = np.reshape(band, (len(band), 1))

            all_cols = np.concatenate((wave1, flux1, fluxerr1, band1), axis=1)

            np.savetxt(
                "profile_127.dat",
                all_cols,
                header="Wave (um), Flux (Jy), Fluxerr (Jy), Band")

            return wave, flux, fluxerr, band

        def scale_127(scale_factor):
            # return flux_127 * scale_factor
            global hony_downsample_flux_trim
            return hony_downsample_flux_trim * scale_factor

        def scale_profile(xax, data, err=(), params=[1], fixed=[False],
                          limitedmin=[False], limitedmax=[False],
                          minpars=[0], maxpars=[10],
                          quiet=True, shh=True):

            def myfunc(x, y, err):
                if len(err) == 0:
                    print("OOOOOOOOOO")

                    def f(p, fjac=None):
                        return [0, (y - scale_127(*p))]
                else:
                    print("KKKKKKKKK")

                    def f(p, fjac=None):
                        return [0, (y - scale_127(*p)) / err]
                return f

            parinfo = [{'n': 0,
                        'value': params[0],
                        'limits':[minpars[0], maxpars[0]],
                        'limited':[limitedmin[0], limitedmax[0]],
                        'fixed':fixed[0],
                        'parname':"scale_factor", 'error':0}]

            mp = mpfit.mpfit(
                myfunc(
                    xax,
                    data,
                    err),
                parinfo=parinfo,
                quiet=quiet)
            mpp = mp.params
            mpperr = mp.perror
            chi2 = mp.fnorm

            if mp.status == 0:
                raise Exception(mp.errmsg)

            if not shh:
                for i, p in enumerate(mpp):
                    parinfo[i]['value'] = p
                    print((parinfo[i]['parname'], p, " +/- ", mpperr[i]))
                print(("Chi2: ", mp.fnorm, " Reduced Chi2: ",
                       mp.fnorm / len(data), " DOF:", len(data) - len(mpp)))

            return mpp, scale_127(*mpp), mpperr, chi2

        # Choose range for fitting.
        rx = np.where((wave >= 11.8) & (wave <= 13.5))

        # Isolate region.
        wavein = wave[rx]
        fluxin = csubSI[rx]
        # fluxerrin = csuberrSI[rx]
        rms = measure_112_RMS(wave, csubSI)

        # Read in Hony's spectrum.
        wave_127, flux_127, _, _ = make_127_profile(home_dir, 11.5, 14)
        wave127 = wave_127
        flux127 = si(flux_127, wave_127)

        # Regrid hony's to the data.
        spl = interp.splrep(wave127, flux127)
        honyWave = wavein
        honyFlux = norm(interp.splev(honyWave, spl))

        ##########################################################
        # Hony
        # Isolate the 12.4 - 12.6 region for fitting the scale factor, both my
        # data and Hony's template spectrum.
        global lower_fitting_boundary
        global upper_fitting_boundary
        lower_fitting_boundary = 12.4
        upper_fitting_boundary = 12.6

        global hony_downsample_flux_trim
        index = np.where(
            (wavein > lower_fitting_boundary) & (
                wavein < upper_fitting_boundary))[0]

        # check for poorly sampled data
        # (i.e. no indices meet the above condition)

        if len(index) == 0:
            # return flux127, fluxerr127, flux128, fluxerr128, amp, position,
            # sigma, integration_wave, integration_flux
            return 0, 0, 0, 0, 0, 0, 0, 0, 0

        # hony_downsample_wave_trim = honyWave[index]
        hony_downsample_flux_trim = honyFlux[index]
        SL_flux_trim = fluxin[index]
        SL_wave_trim = wavein[index]

        # Compute scale factor for Hony template spectrum
        yfit = scale_profile(
            SL_wave_trim, SL_flux_trim, params=[
                np.nanmax(fluxin)])  # False], maxpars=maxpars[10])
        my_scale_factor = yfit[0][0]

        # Subtract scaled Hony spectrum
        scaled_hony_downsample_flux = honyFlux * my_scale_factor
        final_flux = fluxin - scaled_hony_downsample_flux
        final_wave = wavein

        ##########################################################
        # 12.8

        # Fit remainder with gaussian (Neon line at 12.8)
        # params - Fit parameters: Height of background, Amplitude, Shift,
        # Width

        fwhm_min = 0.08
        fwhm_max = 0.12
        fwhm_start = 0.1

        params_in = [0, np.amax(final_flux), 12.813, to_sigma(fwhm_start)]
        limitedmin = [True, True, True, True]
        limitedmax = [True, True, True, True]
        fixed = [True, False, False, False]
        minpars = [0, 0, 12.763, to_sigma(fwhm_min)]
        maxpars = [0.01, np.amax(final_flux) * 1.1, 12.881, to_sigma(fwhm_max)]

        yfit = onedgaussfit(
            final_wave,
            final_flux,
            params=params_in,
            limitedmin=limitedmin,
            minpars=minpars,
            limitedmax=limitedmax,
            fixed=fixed,
            maxpars=maxpars)
        # plt.plot(x,onedgaussian(x,0,5,42,3),'-g',linewidth=3,label='input')
        # print yfit[0]
        amp = yfit[0][1]
        position = yfit[0][2]
        sigma = yfit[0][3]
        # fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        amp_err = yfit[2][1]
        # position_err = yfit[2][2]
        sigma_err = yfit[2][3]

        myrange = [position - 3. * sigma, position + 3. * sigma]
        N = np.where((final_wave >= myrange[0]) & (final_wave <= myrange[1]))
        dl = final_wave[1] - final_wave[0]

        fluxNeII = onedgaussian(final_wave, 0, amp, position, sigma)
        gauss_128_rms = (rms * np.sqrt(len(N)) * dl * 2)
        gauss_128_flux = amp * np.abs(np.sqrt(2) * sigma) * np.sqrt(np.pi)

        if np.sqrt((amp_err / amp)**2 + (sigma_err / sigma)**2) >= 10:
            gauss_128_flux_err = gauss_128_rms
        else:
            frac_err = np.sqrt((amp_err / amp)**2 + (sigma_err / sigma)**2)
            gauss_128_flux_err = frac_err * gauss_128_flux + gauss_128_rms

        ###########################################################
        # 12.7

        # Quantities of interest
        pah_127 = fluxin - fluxNeII
        pah_wave = wavein
        pah_127_watts = pah_127

        # Using the measured 12.7.
        cont_low = 12.2  # find_nearest(cont_wave,12.2)
        cont_hi = 13.0  # find_nearest(cont_wave,13.0)
        idx = np.where((pah_wave >= cont_low) & (pah_wave <= cont_hi))[0]
        integration_wave = pah_wave[idx]
        integration_flux = pah_127_watts[idx]
        trap_flux = sp.integrate.simps(integration_flux, integration_wave)

        # Using Hony's 12.7.
        cont_low = 12.2  # find_nearest(cont_wave,12.2)
        cont_hi = 13.0  # find_nearest(cont_wave,13.0)
        idx = np.where((pah_wave >= cont_low) & (pah_wave <= cont_hi))[0]
        integrationHony_wave = wavein[idx]
        integrationHony_flux = scaled_hony_downsample_flux[idx]
        trap_fluxHony = sp.integrate.simps(
            integrationHony_flux, integrationHony_wave)

        # ONLY IF USING 12.4 - 12.6 !!!!!!!!!!!!
        corrected_pah_trap_flux = trap_flux
        dl3 = integration_wave[1] - integration_wave[0]
        corrected_pah_trap_flux_err = (
            rms * np.sqrt(len(integration_wave)) * dl3 * 2)

        ################################################################
        # Plot to check

        print((gauss_128_flux, gauss_128_flux_err, gauss_128_rms))

        print()
        print(("12.7 flux (using infer. curve): ", trap_flux))
        print(("12.7 flux (using hony's curve): ", trap_fluxHony))

        # STUFF TO RETURN....
        flux127 = corrected_pah_trap_flux
        fluxerr127 = corrected_pah_trap_flux_err
        flux128 = gauss_128_flux
        fluxerr128 = gauss_128_flux_err

        if (trap_flux - trap_fluxHony) / trap_flux * 100 >= 30:
            flux127 = trap_fluxHony
            fluxerr127 = flux127 * 1e10

        return flux127, fluxerr127, flux128, fluxerr128, amp, position, \
            sigma, integration_wave, integration_flux

    def fitLine(atomWave, waveMargin, wave, csub, csuberr, fitMargin,
                rms, dobreak=0):

        def atomicFit(atomWave, cutWave, cutFlux, cutFluxerr, fitMargin):

            if np.all(cutFluxerr == 0.):
                theerr = None
            else:
                theerr = cutFluxerr

            spectral_resolution_hi = atomWave / 60
            spectral_resolution_lo = atomWave / 120
            spectral_resolution_med = np.nanmean(
                (spectral_resolution_lo, spectral_resolution_hi))

            fwhm_guess = spectral_resolution_med
            fwhm_min = spectral_resolution_lo
            fwhm_max = spectral_resolution_hi

            sigma_guess = to_sigma(fwhm_guess)
            sigma_min = to_sigma(fwhm_min)
            sigma_max = to_sigma(fwhm_max)

            sigma_min = sigma_guess * 0.9
            sigma_max = sigma_guess * 1.1

            yfit = onedgaussfit(cutWave, cutFlux, err=theerr,
                                params=[0, np.nanmax(cutFlux), atomWave,
                                        sigma_guess],
                                fixed=[True, False, False, False],
                                limitedmin=[True, True, True, True],
                                limitedmax=[True, False, True, True],
                                minpars=[
                                    0, 0, atomWave - fitMargin, sigma_min],
                                maxpars=[
                                    0,
                                    np.nanmax(cutFlux) *
                                    1.5,
                                    atomWave +
                                    fitMargin,
                                    sigma_max],
                                quiet=True, shh=True)

            g1 = onedgaussian(cutWave, *yfit[0])
            flux_g1 = sp.integrate.simps(g1, cutWave)

            return flux_g1, yfit

        # Prep regions for fitting
        rx = np.where(
            (wave >= atomWave -
             waveMargin) & (
                wave <= atomWave +
                waveMargin))
        cutWave = wave[rx]
        cutFlux = csub[rx]
        cutFluxerr = csuberr[rx]

        if dobreak == 1:
            raise SystemExit("BREAK")

        if np.nanmean(cutFlux) < 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Fit with a single gauss.
        fluxGauss, params = atomicFit(
            atomWave, cutWave, cutFlux, cutFluxerr, fitMargin)

        amp = params[0][1]
        position = params[0][2]
        sigma = params[0][3]
        # fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

        if params[2] is None:
            return fluxGauss, 0, 0, amp, position, sigma

        amp_err = params[2][1]
        # position_err = params[2][2]
        sigma_err = params[2][3]
        gaussAreaErr = np.sqrt((amp_err / amp)**2 +
                               (sigma_err / sigma)**2) * fluxGauss
        gaussFeatErr = compute_feature_uncertainty(
            position, sigma, cutWave, rms)
        gaussSNR = gaussAreaErr / fluxGauss

        if gaussSNR >= 1e5:
            gaussTotalErr = gaussFeatErr
        else:
            gaussTotalErr = gaussFeatErr + gaussAreaErr

        snr = fluxGauss / gaussTotalErr
        # spectrumGauss = onedgaussian(cutWave, 0, amp, position, sigma)

        return fluxGauss, gaussTotalErr, snr, amp, position, sigma

    def pahIntRanges(pahWave):
        if pahWave == 6.2:
            intmin = 6
            intmax = 6.5
        elif pahWave == 7.7:
            intmin = 7.2
            intmax = 8
        elif pahWave == 8.6:
            intmin = 8.2
            intmax = 8.85
        elif pahWave == 11.2:
            intmin = 10.8
            intmax = 11.65
        elif pahWave == 12.0:
            intmin = 11.8
            intmax = 12.1
        elif pahWave == 16.4:
            intmin = 16.15
            intmax = 16.7
        elif pahWave == 17.4:
            intmin = 17.2
            intmax = 17.55
        elif pahWave == 17.8:
            intmin = 17.7
            intmax = 18.2
        else:
            print(("unknown pah: ", pahWave))
            raise SystemExit()

        return intmin, intmax

    def linesToFit():

        atomicLines = [
            9.66,
            10.51,
            12.81,
            15.55,
            17.03,
            18.71,
            33.48,
            34.82,
            28.22,
            25.89]

        atomicLineNames = [
            'H2_97',
            'S4_105',
            'Ne2_128',
            'Ne3_155',
            'H2_17',
            'S3_187',
            'S3_335',
            'Si2_348',
            'H2_282',
            'Fe2_259']

        atomicPlotNames = [
            r'H$_2$ 9.66',
            '[S ɪᴠ] 10.5',
            '[Ne ɪɪ] 12.8',
            '[Ne ɪɪɪ] 15.5',
            r'H$_2$ 17.0',
            '[S ɪɪɪ] 18.7',
            '[S ɪɪɪ] 33.5',
            '[Si ɪɪ] 34.8',
            r'H$_2$ 28.2',
            '[O ɪᴠ] 25.9']

        return atomicLines, atomicLineNames, atomicPlotNames

    def pahsToFit():
        pahLines = np.array(
            [6.2, 7.7, 8.6, 11.2, 12.0, 12.7, 15.8, 16.4, 17.4, 17.8])
        pahLineNames = pahLines.astype(str)
        pahPlotNames = pahLines.astype(str)
        pahPlotNames = np.array(['PAH 6.2',
                                 'PAH 7.7',
                                 'PAH 8.6',
                                 'PAH 11.2',
                                 'PAH 12.0',
                                 'PAH 12.7',
                                 'PAH 15.8',
                                 'PAH 16.4',
                                 'PAH 17.4',
                                 'PAH 17.8'])
        return pahLines, pahLineNames, pahPlotNames

    home_dir = get_home_dir()

    # Which atomic lines to fit.
    atomicLines, atomicLineNames, atomicPlotNames = linesToFit()
    lineArr = []

    # Which PAH features to fit.
    pahLines, pahLineNames, pahPlotNames = pahsToFit()
    pahArr = []

    # RMS.
    rmsSL = measure_112_RMS(waveS, csubS)
    # rmsLL = measure_25_RMS(waveL, csubL)

    # // ATOMIC LINES
    for j in range(len(atomicLines)):
        atomWave = atomicLines[j]
        # atomName = atomicLineNames[j]
        waveMargin = 0.4
        fitMargin = 0.15

        if atomWave in [12.81, 15.55]:
            continue  # HANDLE BLENDED LINES SEPARATELY

        irms = rmsSL
        wave = waveS
        csub = csubS
        csuberr = csubSerr

        dobreak = 0
        lineFlux, lineFluxerr, lineSNR, amp, position, sigma = \
            fitLine(atomWave, waveMargin, wave, csub, csuberr, fitMargin,
                    irms, dobreak=dobreak)

        if atomWave < waveS[-1] and cpos == 5:
            lineArr.append((j, atomWave, 0, 0, 0, amp, position, sigma))
        else:
            lineArr.append(
                (j,
                 atomWave,
                 lineFlux,
                 lineFluxerr,
                 lineSNR,
                 amp,
                 position,
                 sigma))

    # // PAH FEATURES
    for j in range(len(pahLines)):
        pahWave = pahLines[j]
        # pahName = pahLineNames[j]

        if pahWave in [12.7]:
            continue  # BLENDED, DO SEPARATELY

        irms = rmsSL
        wave = waveS
        csub = csubS
        csuberr = csubSerr

        # Figure out integration boundaries.
        intmin, intmax = pahIntRanges(pahWave)

        intRange = np.where((wave >= intmin) & (wave <= intmax))
        wRange = wave[intRange]
        cRange = csub[intRange]
        pahFlux = sp.integrate.simps(cRange, wRange)
        pahFluxErr = compute_feature_uncertainty(
            cRange * 0,
            cRange * 0,
            wRange,
            irms,
            manual_range=[
                intmin,
                intmax])

        if pahWave < waveS[-1] and cpos == 5:
            pahArr.append((j, pahWave, 0, 0, 0, 0, 0))
        else:
            pahArr.append(
                (j,
                 pahWave,
                 pahFlux,
                 pahFluxErr,
                 pahFlux /
                 pahFluxErr,
                 wRange,
                 cRange))

    # // BLENDED FEATURES -- tack onto appropriate atomic/pah lists
    # after finished.
    # 11.0, 6.2, 12.7?

    return np.array(lineArr), np.array(pahArr)


def plotResults(savename, fitResults, wave, flux, fluxerr, plotUnits='jy',
                zoom=0):

    fluxAtomics, fluxPAHs, fluxCont, fluxPlats, csub, csuberr, spline, \
        pahArrRanges, wknots, fknots, finalPlatWave, finalPlatFlux, \
        finalPlatFluxerr, allpahflux, allpahfluxerr, allplatflux, \
        allplatfluxerr = fitResults

    if zoom:
        z = np.where((wave > 5) & (wave < zoom))
        wave = wave[z]
        flux = flux[z]
        fluxerr = fluxerr[z]
        spline = spline[z]
        csub = csub[z]
        csuberr = csuberr[z]
        fknots = fknots[np.where((wknots > 5) & (wknots < zoom))]
        wknots = wknots[np.where((wknots > 5) & (wknots < zoom))]

        if not isinstance(finalPlatFlux, int):
            finalPlatFlux = finalPlatFlux[z]
            finalPlatWave = finalPlatWave[z]
        # st()
        # ax1.set_xlim(xmax=20)
        # ax1.set_ylim(ymin=-100, ymax=1500)

    if plotUnits == 'jy':
        flux = flux
        fluxerr = fluxerr
        spline = spline
        csub = csub
        csuberr = csuberr
        fknots = fknots
        fplat = finalPlatFlux

    elif plotUnits == 'si':
        flux = si(flux, wave)
        fluxerr = si(fluxerr, wave)
        spline = si(spline, wave)
        csub = si(csub, wave)
        csuberr = si(csuberr, wave)
        fknots = si(fknots, wknots)
        fplat = si(finalPlatFlux, finalPlatWave)

    # ================== #
    # Make figure.
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    gs.update(wspace=0.025, hspace=0.00)  # set the spacing between axes.
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    # ================== #
    # AX ZERO
    #
    # Plot plateaus.
    if np.nanmax(fplat) > 0:
        ax0.plot(finalPlatWave, fplat, '--', color='red', lw=1.5)
        ax0.fill_between(finalPlatWave, fplat, spline, color='pink',
                         edgecolor='0.5', lw=0)
    #
    # Plot spectrum to evaluate fits.
    ax0.errorbar(wave, flux, yerr=fluxerr, label='flux', lw=0.5, zorder=1,
                 ecolor='0.6', capsize=0)
    ax0.plot(wave, flux, label='flux', lw=2, color='0.15')
    ax0.plot(wave, spline, label='spline', color='red', lw=1.5)
    ax0.plot(wknots, fknots, 'o', ms=5, color='deepskyblue', mew=1)

    #
    # RMS zones, and tidy up.
    ax0.minorticks_on()
    # ax0.set_ylabel('Flux density (MJy/sr)', fontsize=12)
    ax0.set_ylabel(r'Surface brightness ($MJy/sr$)', fontsize=12)

    rmsminS, rmsmaxS = rms_bounds('SL')
    rmsminL, rmsmaxL = rms_bounds('LL')
    ax1.axvspan(rmsminS, rmsmaxS, color='c', alpha=0.3)
    if zoom == 0:
        ax1.axvspan(rmsminL, rmsmaxL, color='c', alpha=0.3)
    elif zoom == 17.5:
        ax0.set_ylim(ymin=-50, ymax=10000)
        ax1.set_ylim(ymin=-50, ymax=10000)
    # else:
        # ax0.set_ylim(ymin=-50,ymax=10000)
        # ax1.set_ylim(ymin=-50,ymax=10000)
        # ax1.set_ylim(ymin=0)

    # ================== #
    # AX ONE
    ax1.errorbar(wave, csub, yerr=csuberr, label='fluxcsub', lw=0.5, zorder=1,
                 ecolor='0.6', capsize=0)
    ax1.plot(wave, csub, lw=2, color='0.2')

    if fluxPAHs.shape[0] != len(pahArrRanges):
        print("SOMETHING WENT WRONG.")
        raise SystemExit()

    for i in range(len(fluxPAHs)):
        # inpah = fluxPAHs[i]
        featWave = pahArrRanges[i][0]
        featFlux = pahArrRanges[i][1]
        if len(featWave) > 0:
            if plotUnits == 'jy':
                ax1.fill_between(featWave, 0, jy(featFlux, featWave),
                                 color='lightgreen')
            elif plotUnits == 'si':
                ax1.fill_between(featWave, 0, featFlux, color='lightgreen')

    for i in range(len(fluxAtomics)):
        inatom = fluxAtomics[i]
        if plotUnits == 'jy':
            realOG = jy(onedgaussian(wave, 0, inatom[5], inatom[6], inatom[7]),
                        wave)
        elif plotUnits == 'si':
            realOG = onedgaussian(wave, 0, inatom[5], inatom[6], inatom[7])
        ax1.fill_between(wave, 0, realOG, color='salmon')

    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_xlabel('Wavelength (microns)', fontsize=12)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    # ================== #
    # Clean up.
    print(("savename: ", savename))
    fig.savefig(savename, format='pdf', bbox_inches='tight')
    fig.clear()
    plt.close()

    return


def saveResults(fluxdir, fitResults, wave, flux, fluxerr, source, uniqID,
                module='', savePlats=1):

    fluxAtomics, fluxPAHs, fluxCont, fluxPlats, csub, csuberr, spline, \
        pahArrRanges, wknots, fknots, finalPlatWave, finalPlatFlux, \
        finalPlatFluxerr, allpahflux, allpahfluxerr, allplatflux, \
        allplatfluxerr = fitResults

    if module != '':
        fmod = '_' + module
    else:
        fmod = ''

    np.savetxt(fluxdir + source + '_' + uniqID + fmod + '_pahs.txt',
               fluxPAHs, delimiter=',',
               header='j, pahWave, flux, fluxerr, SNR')

    np.savetxt(fluxdir + source + '_' + uniqID + fmod + '_atoms.txt',
               fluxAtomics, delimiter=',',
               header='j, atomWave, flux, fluxerr, SNR, amp, position, sigma')

    np.savetxt(fluxdir + source + '_' + uniqID + fmod + '_conts.txt',
               fluxCont, delimiter=',',
               header='wave, flux, fluxerr (for 2 continuum measurements).')

    np.savetxt(fluxdir + source + '_' + uniqID + fmod + '_total_pahflux.txt',
               np.array([allpahflux, allpahfluxerr]).T, delimiter=',',
               header='total pah flux (W/m^2), total pah flux err (W/m^2)')

    np.savetxt(fluxdir + source + '_' + uniqID + fmod + '_total_platflux.txt',
               np.array([allplatflux, allplatfluxerr]).T, delimiter=',',
               header='total plat flux (W/m^2), total pah flux err (W/m^2)')

    if savePlats == 1:
        header = 'wave, flux, fluxerr, snr for (5, 10, 15 um) plateaus.'
        np.savetxt(fluxdir + source + '_' + uniqID + fmod + '_plats.txt',
                   fluxPlats, delimiter=',',
                   header=header)

    return 0


def saveSpectra(specdir, fitResults, wave, flux, fluxerr, source, uniqID,
                module=''):

    fluxAtomics, fluxPAHs, fluxCont, fluxPlats, csub, csuberr, spline, \
        pahArrRanges, wknots, fknots, finalPlatWave, finalPlatFlux, \
        finalPlatFluxerr, allpahflux, allpahfluxerr, allplatflux, \
        allplatfluxerr = fitResults

    savearr = np.array([wave, flux, fluxerr, spline, csub, csuberr]).T
    np.savetxt(specdir + source + '.txt', savearr, delimiter=',',
               header='wave, flux, fluxerr, spline, csub, csuberr')

    return 0
