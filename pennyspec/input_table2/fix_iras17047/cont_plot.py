#Continuum Plotting
#Pernille Ahlmann Jensen
#21/03/16

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import pyfits as pf
import glob

from ipdb import set_trace as st

def find_nearest(array,value,forcefloor=0):
    idx=(np.abs(array-value)).argmin()
    if forcefloor==1:
        if array[idx] > value:
            idx -= 1
    return idx

def smooth(x,window_len=50,window='hanning'):
    if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
            return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
            w=np.ones(window_len,'d')
    else:
            w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

def continuumPoints(start):
    if start > 6.:
        waveKnots = [7.6,8.25,9.15,9.9,
                    10.35,10.9,11.75,12.23,13.10,13.8]
    else:
        # waveKnots = [2.90,3.90,5.25,5.49,5.71,6.04,6.54,6.70,9.35,9.88,
        #             10.27,10.47,11.63,
        #             11.70,12.20,13.04,13.57,13.93]#,14.5]
        waveKnots = [2.90,3.90,4.6,5.49,5.8,6.00,6.54,6.75,7.15,9.3,10,
                    10.8,
                    11.70,12.20,13.04,13.57,13.93]#,14.5]
        # waveKnots = [5.25,5.49,5.71,6.04,6.54,6.70,9.35,9.88,
        #             10.27,10.47,11.63,12.20,13.0]#,14.5]

    return waveKnots

def getWindowSize():
    return 0.1

def getWindowLen():
    return 11

def continuumFluxes(waveKnots, wave, flux, dosmooth=0, findmin=0):

    holdWave = []
    holdFlux = []

    if dosmooth == 1:
        desiredWindowLen = getWindowLen()
        flux = smooth(flux, window_len=desiredWindowLen)

    if findmin == 1:
        for x in waveKnots:
            dL = getWindowSize()
            winDX = np.where((wave >= x-dL) & (wave <= x+dL))
            winWave = wave[winDX]
            winFlux = flux[winDX]
            minWave = winWave[np.nanargmin(winFlux)]
            minFlux = np.nanmin(winFlux)
            holdWave.append(minWave)
            holdFlux.append(minFlux)
        return holdWave, holdFlux

    else:
        fluxKnots = [flux[find_nearest(wave, x)] for x in waveKnots]
        return waveKnots, fluxKnots

def fitContinuum(wave, flux, dosmooth=0):

    # Retreive continuum points
    waveKnots = continuumPoints(wave[0])

    # st()

    waveKnotsFinal, fluxKnotsFinal = continuumFluxes(waveKnots, wave, flux, dosmooth, findmin=0)
    spl = interp.splrep(waveKnotsFinal, fluxKnotsFinal)

    # st()

    # plt.plot(wave, flux)
    # plt.plot(waveKnotsFinal, fluxKnotsFinal, 'o')
    # plt.show()
    # plt.close()

    # Evaluate spline
    splineWave = wave
    splineFlux = interp.splev(splineWave, spl)

    # st()

    return splineWave, splineFlux, waveKnotsFinal, fluxKnotsFinal


dataDir = ''
#allFiles = glob.glob(dataDir + '*_conv.txt')

#for i in range(len(allFiles)):
#for i in [4]:

    #skipList = [3, 76, 60, 22, 63, 43, 20, 16, 55, 37]
    #if i in skipList:
    #   continue

    #if i == 5: # Spectra/Tables/SMPLMC75_HR_final.txt
        #continue
    #if i == 19: # Spectra/Tables/IRAS13416-6243_final.txt
        #continue

    #fname = allFiles[i]
    #fnameStr = fname.split('_conv.txt')[0].split('/')[-1]

    #print i, fname

    # Read data
data = np.loadtxt(dataDir+'iras17047.txt', delimiter=',').T
wave, flux, fluxerr = data[0], data[1], data[2]

#flux = smooth(flux,window_len=11,window='hanning')

# idx = np.where((wave >= 4.5) & (wave <= 13.5))
# wave = wave[idx]
# flux = flux[idx]
# fluxerr = fluxerr[idx]


splineWave, splineFlux, waveKnots, fluxKnots = fitContinuum(wave, flux, dosmooth=1)
splineWaveSmoo, splineFluxSmoo, waveKnotsSmoo, fluxKnotsSmoo = fitContinuum(wave, flux, dosmooth=1)

# st()




wavelim=np.where(wave<=13.9)
# Plot fit
fig, ax = plt.subplots()
ax.errorbar(wave[wavelim], flux[wavelim], fluxerr[wavelim], color='r', ecolor='0.45', label='Data', lw=2, elinewidth=1)
ax.plot(wave[wavelim], flux[wavelim], color='r', label='Data', lw=2)

ax.plot(splineWave[wavelim], splineFlux[wavelim], 'g-', label='Spline fit', lw=2, zorder=1000)
ax.plot(waveKnots, fluxKnots, 'ko', ms=4, zorder=1000)
# ax.plot(splineWaveSmoo[wavelim], splineFluxSmoo[wavelim], 'b-', label='Spline fit', lw=2)
# ax.plot(waveKnotsSmoo, fluxKnotsSmoo, 'o', color='cyan', ms=4)
ax.set_xlabel('Wavelength ($\mu$m)')
ax.set_ylabel('Flux (Jy)')
ax.set_title('iras17047-- Continuum')
ax.grid()
ax.minorticks_on()
# ax.axvspan(6.75, 6.95, color='c', alpha=0.3, lw=0)
# ax.axvspan(7.1, 7.3, color='c', alpha=0.3, lw=0)
ax.axvline(x=6.9)
ax.axvline(x=7.25)
plt.legend(loc=0)
fig.savefig('iras17047_CWcont.pdf', format='pdf', bbox_inches='tight')
    #fig.savefig('Spectra/Figures/continuum2/' + fnameStr + '_cont.pdf', format='pdf', bbox_inches='tight')
plt.close()
fig.clear()




wavelim1 = np.where((wave>=5)&(wave<=8))

fig1, ax = plt.subplots()
ax.errorbar(wave[wavelim1], flux[wavelim1], fluxerr[wavelim1], color='r', ecolor='0.45', label='Data', lw=2, elinewidth=1)
ax.plot(wave[wavelim1], flux[wavelim1], color='r', label='Data', lw=2)
ax.plot(splineWave[wavelim1], splineFlux[wavelim1], 'g-', label='Spline fit', lw=2, zorder=1000)
ax.plot(waveKnots, fluxKnots, 'ko', ms=4, zorder=1000)
# ax.plot(splineWaveSmoo[wavelim], splineFluxSmoo[wavelim], 'b-', label='Spline fit', lw=2)
# ax.plot(waveKnotsSmoo, fluxKnotsSmoo, 'o', color='cyan', ms=4)
ax.set_xlabel('Wavelength (microns)')
ax.set_ylabel('Flux (Jy)')
ax.set_title('iras17047 -- Continuum - zoom')
ax.grid(ls=':', color='0.5', lw=0.5)
ax.minorticks_on()
# ax.axvspan(6.75, 6.95, color='c', alpha=0.3, lw=0)
# ax.axvspan(7.1, 7.3, color='c', alpha=0.3, lw=0)
ax.axvline(x=6.9)
ax.axvline(x=7.25)
plt.legend(loc=0)
ax.set_xlim(left=5, right=8)
fig1.savefig('iras17047_CWcont_zoom1.pdf', format='pdf', bbox_inches='tight')
    #fig.savefig('Spectra/Figures/continuum2/' + fnameStr + '_cont.pdf', format='pdf', bbox_inches='tight')
plt.close()
fig1.clear()



# wavelim2 = np.where((wave>=6.6)&(wave<=9))
wavelim2 = np.where((wave>=5)&(wave<=10))

fig2, ax = plt.subplots()
ax.errorbar(wave[wavelim2], flux[wavelim2], fluxerr[wavelim2], color='r', ecolor='0.45', label='Data', lw=2, elinewidth=1)
ax.plot(wave[wavelim2], flux[wavelim2], color='r', label='Data', lw=2)

ax.plot(splineWave, splineFlux, 'g-', label='Spline fit', lw=2, zorder=1000)
# ax.plot(splineWave[wavelim2], splineFlux[wavelim2], 'g-', label='Spline fit', lw=2)
ax.plot(waveKnots, fluxKnots, 'ko', ms=4, zorder=1000)
# ax.plot(splineWaveSmoo[wavelim], splineFluxSmoo[wavelim], 'b-', label='Spline fit', lw=2)
# ax.plot(waveKnotsSmoo, fluxKnotsSmoo, 'o', color='cyan', ms=4)
ax.set_xlabel('Wavelength (microns)')
ax.set_ylabel('Flux (Jy)')
ax.set_title('iras17047 -- Continuum - zoom')
# ax.grid()
ax.grid(ls=':', color='0.5', lw=0.5)
ax.minorticks_on()
# ax.axvspan(6.75, 6.95, color='c', alpha=0.3, lw=0)
# ax.axvspan(7.1, 7.3, color='c', alpha=0.3, lw=0)
ax.axvline(x=6.9)
ax.axvline(x=7.25)
plt.legend(loc=0)
ax.set_xlim(left=6, right=10)
fig2.savefig('iras17047_CWcont_zoom2.pdf', format='pdf', bbox_inches='tight')
    #fig.savefig('Spectra/Figures/continuum2/' + fnameStr + '_cont.pdf', format='pdf', bbox_inches='tight')
plt.close()
fig2.clear()





wavelim3 = np.where((wave>=8)&(wave<=13))

fig3, ax = plt.subplots()
ax.errorbar(wave[wavelim3], flux[wavelim3], fluxerr[wavelim3], color='r', ecolor='0.45', label='Data', lw=2, elinewidth=1)
ax.plot(wave[wavelim3], flux[wavelim3], color='r', label='Data', lw=2)
ax.plot(splineWave[wavelim3], splineFlux[wavelim3], 'g-', label='Spline fit', lw=2, zorder=1000)
ax.plot(waveKnots, fluxKnots, 'ko', ms=4, zorder=1000)
# ax.plot(splineWaveSmoo[wavelim], splineFluxSmoo[wavelim], 'b-', label='Spline fit', lw=2)
# ax.plot(waveKnotsSmoo, fluxKnotsSmoo, 'o', color='cyan', ms=4)
ax.set_xlabel('Wavelength (microns)')
ax.set_ylabel('Flux (Jy)')
ax.set_title('iras17047 -- Continuum - zoom')
# ax.grid()
ax.grid(ls=':', color='0.5', lw=0.5)
ax.minorticks_on()
#ax.axvspan(6.75, 6.95, color='c', alpha=0.3, lw=0)
#ax.axvspan(7.1, 7.3, color='c', alpha=0.3, lw=0)
plt.legend(loc=0)
ax.set_xlim(left=8, right=13)
fig3.savefig('iras17047_CWcont_zoom3.pdf', format='pdf', bbox_inches='tight')
    #fig.savefig('Spectra/Figures/continuum2/' + fnameStr + '_cont.pdf', format='pdf', bbox_inches='tight')
plt.close()
fig1.clear()

np.savetxt('iras17047_convCWcont.txt', np.c_[splineWave[wavelim], splineFlux[wavelim], fluxerr[wavelim]], delimiter=',', header='col1: wavelength, col2: cont flux')
np.savetxt('iras17047_convCWknots.txt', np.c_[waveKnots, fluxKnots], delimiter=',', header='col1: waveknot, col2: fluxknot')
#np.savetxt('iras17047_convCWcont_all.txt', np.c_[wave[wavelim], flux[wavelim], fluxerr[wavelim], splineFlux[wavelim], delimiter=',', header='wavelength, flux, contflux, error')

newFlux = flux[wavelim]-splineFlux[wavelim]



# Plot
fig4, ax = plt.subplots()
ax.errorbar(wave[wavelim], newFlux, fluxerr[wavelim], color='r', ecolor='0.45', label='Data', lw=2, elinewidth=1)
ax.plot(wave[wavelim], newFlux, color='r', label='Data', lw=2)
ax.axhline(y=0,color='k',ls='-',zorder=-10,lw=2)

ax.set_xlabel('Wavelength (microns)')
ax.set_ylabel('Flux (Jy)')
ax.set_title('iras17047 -- ' + 'Continuum Subtracted CW')
# ax.grid()
ax.grid(ls=':', color='0.5', lw=0.5)
ax.minorticks_on()
# ax.axvspan(6.75, 6.95, color='c', alpha=0.3, lw=0)
# ax.axvspan(7.1, 7.3, color='c', alpha=0.3, lw=0)
ax.axvline(x=6.9)
ax.axvline(x=7.25)
# Save figure and data
fig4.savefig('iras17047_convCWsub.pdf', format='pdf', bbox_inches='tight')
np.savetxt('iras17047_convCWsub.txt', np.c_[wave[wavelim], newFlux, fluxerr[wavelim]], delimiter=',', header='col1: wavelength, col2: cont_sub flux, col3: Flux error')

plt.close()
fig4.clear()










