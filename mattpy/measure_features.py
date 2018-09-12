import numpy as np
import scipy as sp
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn.apionly as sns
import matplotlib.patches as mpatches

from . import mylineid as lid

from mattpy.utils import to_sigma, norm, get_home_dir, find_nearest, quant_str
from mattpy import continuum_fluxes as cf
from mattpy import stitch, mpfit


home_dir = get_home_dir()


def rms(module):
    if module == 'SL':
        wlow = 10.
        whigh = 10.4
    elif module == 'LL':
        wlow = 27
        whigh = 27.5
    elif module == 'SL14':
        wlow = 14.5
        whigh = 15.2
    else:
        raise SystemExit("module unrecognized.")

    return wlow, whigh


def compute_feature_uncertainty(gposition, gsigma, wave_feat, rms,
                                manual_range=0):

    if isinstance(manual_range, int):
        myrange = [gposition - (3. * gsigma), gposition + (3. * gsigma)]
    else:
        myrange = manual_range

    dl = wave_feat[1] - wave_feat[0]
    N = (myrange[1] - myrange[0]) / dl
    feature_uncertainty = (rms * np.sqrt(N) * dl * 2)

    return feature_uncertainty


def multigaussfit(xax,data,ngauss=1,err=(),params=[1,0,1],fixed=[False,False,False],limitedmin=[False,False,True],
        limitedmax=[False,False,False],minpars=[0,0,0],maxpars=[0,0,0],
        quiet=True,shh=True):
    """
    An improvement on onedgaussfit.  Lets you fit multiple gaussians.

    Inputs:
       xax - x axis
       data - y axis
       ngauss - How many gaussians to fit?  Default 1 (this could supercede onedgaussfit)
       err - error corresponding to data

     These parameters need to have length = 3*ngauss.  If ngauss > 1 and length = 3, they will
     be replicated ngauss times, otherwise they will be reset to defaults:
       params - Fit parameters: [amplitude, offset, width] * ngauss
              If len(params) % 3 == 0, ngauss will be set to len(params) / 3
       fixed - Is parameter fixed?
       limitedmin/minpars - set lower limits on each parameter (default: width>0)
       limitedmax/maxpars - set upper limits on each parameter

       quiet - should MPFIT output each iteration?
       shh - output final parameters?

    Returns:
       Fit parameters
       Model
       Fit errors
       chi2
    """

    if len(params) != ngauss and (len(params) / 3) > ngauss:
        ngauss = len(params) / 3

    # make sure all various things are the right length; if they're not, fix them using the defaults
    for parlist in (params,fixed,limitedmin,limitedmax,minpars,maxpars):
        if len(parlist) != 3*ngauss:
            # if you leave the defaults, or enter something that can be multiplied by 3 to get to the
            # right number of gaussians, it will just replicate
            if len(parlist) == 3:
                parlist *= ngauss
            elif parlist==params:
                parlist[:] = [1,0,1] * ngauss
            elif parlist==fixed or parlist==limitedmax:
                parlist[:] = [False,False,False] * ngauss
            elif parlist==limitedmin:
                parlist[:] = [False,False,True] * ngauss
            elif parlist==minpars or parlist==maxpars:
                parlist[:] = [0,0,0] * ngauss

    def mpfitfun(x,y,err):
        if len(err) == 0:
            def f(p,fjac=None): return [0,(y-n_gaussian(x,pars=p))]
        else:
            def f(p,fjac=None): return [0,(y-n_gaussian(x,pars=p))/err]
        return f

    # st()

    # if xax == None:
    #     xax = np.arange(len(data))

    parnames = {0:"Position",1:"Width",2:"Amp"}

    #parinfo = [ {'n':ii,'value':params[ii],'limits':[minpars[ii],maxpars[ii]],'limited':[limitedmin[ii],limitedmax[ii]],'fixed':fixed[ii],'parname':parnames[ii%3]+str(ii/3),'error':ii} for ii in range(len(params)) ]

    parinfo = [ {'step':0.01,'n':ii,'value':params[ii],'limits':[minpars[ii],maxpars[ii]],'limited':[limitedmin[ii],limitedmax[ii]],'fixed':fixed[ii],'parname':parnames[ii%3]+str(ii%3),'error':ii} for ii in range(len(params)) ]

    #print range(len(params))

    mp = mpfit.mpfit(mpfitfun(xax,data,err),parinfo=parinfo,quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm

    if mp.status == 0:
        raise Exception(mp.errmsg)

    if not shh:
        print()
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print((parinfo[i]['parname'],p," +/- ",mpperr[i]))
        print(("Chi2: ",mp.fnorm," Reduced Chi2: ",mp.fnorm/len(data)," DOF:",len(data)-len(mpp)))

    return mpp,n_gaussian(xax,pars=mpp),mpperr,chi2


def multigaussfit_back(xax,data,ngauss=1,err=None,params=[1,0,1],fixed=[False,False,False],limitedmin=[False,False,True],
        limitedmax=[False,False,False],minpars=[0,0,0],maxpars=[0,0,0],
        quiet=True,shh=True):
    """
    An improvement on onedgaussfit.  Lets you fit multiple gaussians.

    Inputs:
       xax - x axis
       data - y axis
       ngauss - How many gaussians to fit?  Default 1 (this could supercede onedgaussfit)
       err - error corresponding to data

     These parameters need to have length = 3*ngauss.  If ngauss > 1 and length = 3, they will
     be replicated ngauss times, otherwise they will be reset to defaults:
       params - Fit parameters: [amplitude, offset, width] * ngauss
              If len(params) % 3 == 0, ngauss will be set to len(params) / 3
       fixed - Is parameter fixed?
       limitedmin/minpars - set lower limits on each parameter (default: width>0)
       limitedmax/maxpars - set upper limits on each parameter

       quiet - should MPFIT output each iteration?
       shh - output final parameters?

    Returns:
       Fit parameters
       Model
       Fit errors
       chi2
    """

    if len(params) != ngauss and (len(params) / 3) > ngauss:
        ngauss = len(params) / 3

    # make sure all various things are the right length; if they're not, fix them using the defaults
    for parlist in (params,fixed,limitedmin,limitedmax,minpars,maxpars):
        if len(parlist) != 3*ngauss:
            # if you leave the defaults, or enter something that can be multiplied by 3 to get to the
            # right number of gaussians, it will just replicate
            if len(parlist) == 3:
                parlist *= ngauss
            elif parlist==params:
                parlist[:] = [1,0,1] * ngauss
            elif parlist==fixed or parlist==limitedmax:
                parlist[:] = [False,False,False] * ngauss
            elif parlist==limitedmin:
                parlist[:] = [False,False,True] * ngauss
            elif parlist==minpars or parlist==maxpars:
                parlist[:] = [0,0,0] * ngauss

    def mpfitfun(x,y,err):
        if err == None:
            def f(p,fjac=None): return [0,(y-n_gaussian(x,pars=p))]
        else:
            def f(p,fjac=None): return [0,(y-n_gaussian(x,pars=p))/err]
        return f

    if xax == None:
        xax = np.arange(len(data))

    parnames = {0:"Position",1:"Width",2:"Amp"}

    #parinfo = [ {'n':ii,'value':params[ii],'limits':[minpars[ii],maxpars[ii]],'limited':[limitedmin[ii],limitedmax[ii]],'fixed':fixed[ii],'parname':parnames[ii%3]+str(ii/3),'error':ii} for ii in range(len(params)) ]

    parinfo = [ {'step':0.01,'n':ii,'value':params[ii],'limits':[minpars[ii],maxpars[ii]],'limited':[limitedmin[ii],limitedmax[ii]],'fixed':fixed[ii],'parname':parnames[ii%3]+str(ii/3),'error':ii} for ii in range(len(params)) ]

    #print range(len(params))

    mp = mpfit.mpfit(mpfitfun(xax,data,err),parinfo=parinfo,quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm

    if mp.status == 0:
        raise Exception(mp.errmsg)

    if not shh:
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print((parinfo[i]['parname'],p," +/- ",mpperr[i]))
        print(("Chi2: ",mp.fnorm," Reduced Chi2: ",mp.fnorm/len(data)," DOF:",len(data)-len(mpp)))

    return mpp,n_gaussian(xax,pars=mpp),mpperr,chi2
def fwhm_to_sigma(fwhm):
    #fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma

def n_gaussian(x, pars=None,a=None,dx=None,sigma=None):
    """
    Returns a function that sums over N gaussians, where N is the length of
    dx,sigma,a *OR* N = len(pars) / 3

    The background "height" is assumed to be zero (you must "baseline" your
    spectrum before fitting)

    pars  - a list with len(pars) = 3n, assuming a,dx,sigma repeated
    dx    - offset (velocity center) values
    sigma - line widths
    a     - amplitudes
    """
    if len(pars) % 3 == 0:
        a = [pars[ii] for ii in range(2,len(pars),3)]
        dx = [pars[ii] for ii in range(0,len(pars),3)]
        sigma = [pars[ii] for ii in range(1,len(pars),3)]

    elif not(len(dx) == len(sigma) == len(a)):
        raise ValueError("Wrong array lengths! dx: %i  sigma: %i  a: %i" % (len(dx),len(sigma),len(a)))

    def g(x):
        v = np.zeros(len(x))
        for i in range(len(dx)):
            onegauss = a[i] * np.exp( - ( x - dx[i] )**2 / (2*sigma[i]**2) )
            v += onegauss

        return v
    return g(x)

def includePlat1821():
    return 0


def scale_profile(xax,data,err=(),params=[1],fixed=[False],limitedmin=[False],
        limitedmax=[False],minpars=[0],maxpars=[10],
        quiet=True,shh=True):

    #def mpfitfun(x,y,err):
    def myfunc(x,y,err):
        if len(err) == 0:
            print("OOOOOOOOOO")
            def f(p,fjac=None):
                return [0,(y-scale_127(*p))]
        else:
            print("KKKKKKKKK")
            def f(p,fjac=None):
                return [0,(y-scale_127(*p))/err]
        return f




    # if len(xax) == 0:
    #     xax = np.arange(len(data))

    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],'parname':"scale_factor",'error':0}]

    #parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],'parname':"HEIGHT",'error':0} ,
                #{'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],'parname':"AMPLITUDE",'error':0},
                #{'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2],'parname':"SHIFT",'error':0},
                #{'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3],'parname':"WIDTH",'error':0}]

    mp = mpfit.mpfit(myfunc(xax,data,err),parinfo=parinfo,quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm

    if mp.status == 0:
        raise Exception(mp.errmsg)

    if not shh:
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print((parinfo[i]['parname'],p," +/- ",mpperr[i]))
        print(("Chi2: ",mp.fnorm," Reduced Chi2: ",mp.fnorm/len(data)," DOF:",len(data)-len(mpp)))

    return mpp,scale_127(*mpp),mpperr,chi2

def scale_profile_back(xax,data,err=None,params=[1],fixed=[False],limitedmin=[False],
        limitedmax=[False],minpars=[0],maxpars=[10],
        quiet=True,shh=True):

    #def mpfitfun(x,y,err):
    def myfunc(x,y,err):
        if err == None:
            def f(p,fjac=None): return [0,(y-scale_127(*p))]
        else:
            def f(p,fjac=None): return [0,(y-scale_127(*p))/err]
        return f

    if xax == None:
        xax = np.arange(len(data))

    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],'parname':"scale_factor",'error':0}]

    #parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],'parname':"HEIGHT",'error':0} ,
                #{'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],'parname':"AMPLITUDE",'error':0},
                #{'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2],'parname':"SHIFT",'error':0},
                #{'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3],'parname':"WIDTH",'error':0}]

    mp = mpfit.mpfit(myfunc(xax,data,err),parinfo=parinfo,quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm

    if mp.status == 0:
        raise Exception(mp.errmsg)

    if not shh:
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print((parinfo[i]['parname'],p," +/- ",mpperr[i]))
        print(("Chi2: ",mp.fnorm," Reduced Chi2: ",mp.fnorm/len(data)," DOF:",len(data)-len(mpp)))

    return mpp,scale_127(*mpp),mpperr,chi2
def scale_127(scale_factor):
    #return flux_127 * scale_factor
    global hony_downsample_flux_trim
    return hony_downsample_flux_trim * scale_factor
def make_127_profile(home_dir, valmin=12.22,valmax=13.2):
    # st()
    data = np.loadtxt(str(home_dir + 'Dropbox/code/Python/fitting_127/average_pah_spectrum.dat'))
    data = data.T

    n = np.where((data[0] > valmin) & (data[0] < valmax))[0]
    wave = data[0][n]
    flux = data[1][n]
    fluxerr = data[2][n]
    band = data[3][n]

    # plt.plot(wave,flux)
    # plt.show()

    flux = flux - 0.05 #offset to zero it

    # plt.plot(wave,flux)
    # plt.show()

    # raise SystemExit("")

    for i in range(len(wave)):
        if i != len(wave)-1:
            if wave[i+1] - wave[i] <= 0:
                #print i, wave[i]
                cut_it = i

    w1 = wave[:cut_it+1]
    f1 = flux[:cut_it+1]
    fe1 = fluxerr[:cut_it+1]
    b1 = band[:cut_it+1]

    w2 = wave[cut_it+1:]
    f2 = flux[cut_it+1:]
    fe2 = fluxerr[cut_it+1:]
    b2 = band[cut_it+1:]

    c1 = np.where(w1 < w2[0])[0]
    w1 = w1[c1]
    f1 = f1[c1]
    fe1 = fe1[c1]
    b1 = b1[c1]


    # tie together
    wave = np.concatenate((w1,w2),axis=0)
    flux = np.concatenate((f1,f2),axis=0)
    fluxerr = np.concatenate((fe1,fe2),axis=0)
    band = np.concatenate((b1,b2),axis=0)
    # flux -= np.amin(flux)

    wave1 = np.reshape(wave,(len(wave),1))
    flux1 = np.reshape(flux,(len(flux),1))
    fluxerr1 = np.reshape(fluxerr,(len(fluxerr),1))
    band1 = np.reshape(band,(len(band),1))

    all_cols = np.concatenate((wave1,flux1,fluxerr1,band1),axis=1)

    np.savetxt("profile_127.dat",all_cols,header="Wave (um), Flux (Jy), Fluxerr (Jy), Band")

    return wave, flux, fluxerr, band
def fit127(wave, csubSI, csuberrSI):

    # Choose range for fitting.
    rx = np.where((wave >= 11.8) & (wave <= 13.5))

    # Isolate region.
    wavein = wave[rx]
    fluxin = csubSI[rx]
    fluxerrin = csuberrSI[rx]
    rms = measure_112_RMS(wave, csubSI)

    # Read in Hony's spectrum.
    # wave_127, flux_127, _, _ = make_127_profile('/home/koma/', 11.5,14)
    wave_127, flux_127, _, _ = make_127_profile(home_dir, 11.5,14)
    wave127 = wave_127
    flux127 = si(flux_127, wave_127)

    # Regrid hony's to the data.
    spl = interp.splrep(wave127, flux127)
    honyWave = wavein
    honyFlux = norm(interp.splev(honyWave, spl))



    ##########################################################
    ########################## Hony ##########################
    # Isolate the 12.4 - 12.6 region for fitting the scale factor, both my data and Hony's template spectrum.
    global lower_fitting_boundary
    global upper_fitting_boundary
    lower_fitting_boundary = 12.4
    upper_fitting_boundary = 12.6

    global hony_downsample_flux_trim
    index = np.where((wavein > lower_fitting_boundary) & (wavein < upper_fitting_boundary))[0]
    # check for poorly sampled data (i.e. no indices meet the above condition)
    # st()
    if len(index) == 0:
        # return flux127, fluxerr127, flux128, fluxerr128, amp, position, sigma, integration_wave, integration_flux
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    hony_downsample_wave_trim = honyWave[index]
    hony_downsample_flux_trim = honyFlux[index]
    SL_flux_trim = fluxin[index]
    SL_wave_trim = wavein[index]

    # Compute scale factor for Hony template spectrum
    yfit = scale_profile(SL_wave_trim, SL_flux_trim, params=[np.nanmax(fluxin)])#False], maxpars=maxpars[10])
    my_scale_factor = yfit[0][0]

    # Subtract scaled Hony spectrum
    scaled_hony_downsample_flux = honyFlux * my_scale_factor
    final_flux = fluxin - scaled_hony_downsample_flux
    final_wave = wavein



    ##########################################################
    ########################## 12.8 ##########################
    # Fit remainder with gaussian (Neon line at 12.8)
    #params - Fit parameters: Height of background, Amplitude, Shift, Width

    fwhm_min = 0.08
    fwhm_max = 0.12
    fwhm_start = 0.1

    params_in = [0,np.amax(final_flux),12.813,to_sigma(fwhm_start)]
    limitedmin = [True,True,True,True]
    limitedmax = [True,True,True,True]
    fixed = [True,False,False,False]
    minpars = [0,0,12.763,to_sigma(fwhm_min)]
    maxpars = [0.01,np.amax(final_flux)*1.1,12.881,to_sigma(fwhm_max)]

    yfit = onedgaussfit(final_wave, final_flux, params=params_in, limitedmin=limitedmin, minpars=minpars, limitedmax=limitedmax, fixed=fixed, maxpars=maxpars)
    #plt.plot(x,onedgaussian(x,0,5,42,3),'-g',linewidth=3,label='input')
    #print yfit[0]
    amp = yfit[0][1]
    position = yfit[0][2]
    sigma = yfit[0][3]
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    amp_err = yfit[2][1]
    position_err = yfit[2][2]
    sigma_err = yfit[2][3]

    myrange = [position-3.*sigma, position+3.*sigma]
    N = np.where((final_wave >= myrange[0]) & (final_wave <= myrange[1]))
    dl = final_wave[1] - final_wave[0]

    fluxNeII = onedgaussian(final_wave,0,amp,position,sigma)
    gauss_128_rms = (rms * np.sqrt(len(N)) * dl * 2)
    gauss_128_flux = amp * np.abs(np.sqrt(2)*sigma) * np.sqrt(np.pi)

    if np.sqrt((amp_err/amp)**2 + (sigma_err/sigma)**2) >= 10:
        gauss_128_flux_err = gauss_128_rms
    else:
        gauss_128_flux_err = np.sqrt((amp_err/amp)**2 + (sigma_err/sigma)**2) * gauss_128_flux + gauss_128_rms

    ###########################################################
    ####################### PAH 12.7 ##########################
    # Quantities of interest
    pah_127 = fluxin - fluxNeII
    pah_wave = wavein
    pah_127_watts = pah_127

    # Using the measured 12.7.
    cont_low = 12.2 #find_nearest(cont_wave,12.2)
    cont_hi = 13.0 #find_nearest(cont_wave,13.0)
    idx = np.where((pah_wave >= cont_low) & (pah_wave <= cont_hi))[0]
    integration_wave = pah_wave[idx]
    integration_flux = pah_127_watts[idx]
    trap_flux = sp.integrate.simps(integration_flux, integration_wave)

    # Using Hony's 12.7.
    cont_low = 12.2 #find_nearest(cont_wave,12.2)
    cont_hi = 13.0 #find_nearest(cont_wave,13.0)
    idx = np.where((pah_wave >= cont_low) & (pah_wave <= cont_hi))[0]
    integrationHony_wave = wavein[idx]
    integrationHony_flux = scaled_hony_downsample_flux[idx]
    trap_fluxHony = sp.integrate.simps(integrationHony_flux, integrationHony_wave)

    # CORRECT FOR SYSTEMATIC UNCERTAINTY!!!!!!!!!!!
    # measured_neon_to_pah_trap_flux = measured_gaussian_trap_flux / trap_flux
    # corrected_pah_trap_flux = trap_flux / Simple_SimpleEquation_18_Offset_model(measured_neon_to_pah_trap_flux)
    ## USING 12.4-12.6, CORRECTION IS 1!!!! i.e. no correction.
    ## corrected_pah_trap_flux = trap_flux / Simple_SimpleEquation_18_Offset_model(measured_neon_to_pah_trap_flux)
    corrected_pah_trap_flux = trap_flux # ONLY IF USING 12.4 - 12.6 !!!!!!!!!!!!
    dl3 = integration_wave[1] - integration_wave[0]
    corrected_pah_trap_flux_err = (rms * np.sqrt(len(integration_wave)) * dl3 * 2)
    #corrected_pah_trap_flux_err = corrected_pah_trap_flux / 10.




    ################################################################
    ####################### Plot to check ##########################

    print((gauss_128_flux, gauss_128_flux_err, gauss_128_rms))



    print()
    print(("12.7 flux (using infer. curve): ", trap_flux))
    print(("12.7 flux (using hony's curve): ", trap_fluxHony))

    #mlw = 3
    #plt.plot(final_wave, fluxin, label="SL", lw=mlw*4, color='b')
    #plt.plot(final_wave, honyFlux*my_scale_factor, label="Hony spectrum scaled", lw=mlw*2, color='k', ls='-')
    #plt.plot(final_wave, fluxNeII, label='Ne II', lw=mlw*2, color='k', ls='-')
    #plt.plot(final_wave, fluxNeII + honyFlux*my_scale_factor, label="Total", lw=mlw*2, color='k', ls='-')
    #plt.plot(final_wave, pah_127, label='Implied 12.7', lw=mlw, color='r')
    ##plt.plot(final_wave, fluxin - honyFlux*my_scale_factor, label="SL - Hony")
     ##plt.plot(hony_downsample_wave_trim, hony_downsample_flux_trim*my_scale_factor, label="Hony trimmed spectrum scaled")
    ##plt.plot(final_wave, final_flux, label="SL_flux - scaled Hony spectrum")
    #[plt.axvline(x, ls='-', color='k') for x in (lower_fitting_boundary, upper_fitting_boundary,cont_low,cont_hi)]
    #plt.axhline(y=0, ls='-', color='k')
    #plt.title(corrected_pah_trap_flux)
    #plt.legend()
    #plt.show()
    #plt.close()
    #st()


    # STUFF TO RETURN....
    flux127 = corrected_pah_trap_flux
    fluxerr127 = corrected_pah_trap_flux_err
    flux128 = gauss_128_flux
    fluxerr128 = gauss_128_flux_err

    if (trap_flux-trap_fluxHony)/trap_flux*100 >= 30:
        #fluxerr127 = flux127*10
        #fluxerr128 = flux128*10
        flux127 = trap_fluxHony
        fluxerr127 = flux127*1e10
        #st()

    return flux127, fluxerr127, flux128, fluxerr128, amp, position, sigma, integration_wave, integration_flux
def measure_112_RMS(wave,csub):
    xmin, xmax = rms('SL')
    #xmin = 10.
    #xmax = 10.4
    myrange = np.where((wave >= xmin) & (wave <= xmax))
    csub_mr = csub[myrange]
    rms = np.sqrt(np.nanmean(csub_mr**2))

    return rms
def measure_14_RMS(wave,csub,passflux=0):

    #plt.plot(wave, passflux)
    #plt.plot(wave, csub)
    #plt.show()
    #plt.close()

    #st()
    xmin, xmax = rms('SL14')
    #xmin = 10.
    #xmax = 10.4
    myrange = np.where((wave >= xmin) & (wave <= xmax))
    csub_mr = csub[myrange]
    rms = np.sqrt(np.nanmean(csub_mr**2))

    return rms

def measure_25_RMS(wave,csub):
    xmin, xmax = rms('LL')
    #xmin = 24
    #xmax = 25
    myrange = np.where((wave >= xmin) & (wave <= xmax))
    csub_mr = csub[myrange]
    rms = np.sqrt(np.nanmean(csub_mr**2))

    return rms
def measure_112_RMS2D(wave,csub):
    xmin, xmax = rms('SL')
    #xmin = 10.
    #xmax = 10.4

    #csub = flux - cont
    myrange = np.where((wave >= xmin) & (wave <= xmax))
    csub_mr = csub[myrange[0],:]

    allrms = []
    nPix = csub_mr.shape[1]

    for i in range(nPix):
        rms = np.sqrt(np.nanmean(csub_mr[:,i]**2))
        allrms.append(rms)
    rmsvec = np.array(allrms)

    return rmsvec
def measure_25_RMS2D(wave,csub):
    xmin, xmax = rms('LL')
    #xmin = 24
    #xmax = 25

    #csub = flux - cont
    myrange = np.where((wave >= xmin) & (wave <= xmax))
    csub_mr = csub[myrange[0],:]

    allrms = []
    nPix = csub_mr.shape[1]

    for i in range(nPix):
        rms = np.sqrt(np.nanmean(csub_mr[:,i]**2))
        allrms.append(rms)
    rmsvec = np.array(allrms)

    return rmsvec
def atomicFit(atomWave, cutWave, cutFlux, cutFluxerr, fitMargin):

    if np.all(cutFluxerr == 0.):
        theerr = None
    else:
        theerr = cutFluxerr

    spectral_resolution_hi = atomWave / 60
    spectral_resolution_lo = atomWave / 120
    spectral_resolution_med = np.nanmean((spectral_resolution_lo, spectral_resolution_hi))

    fwhm_guess = spectral_resolution_med
    fwhm_min = spectral_resolution_lo
    fwhm_max = spectral_resolution_hi

    sigma_guess = to_sigma(fwhm_guess)
    sigma_min = to_sigma(fwhm_min)
    sigma_max = to_sigma(fwhm_max)

    sigma_min = sigma_guess * 0.9
    sigma_max = sigma_guess * 1.1


    yfit = onedgaussfit(cutWave, cutFlux, err=theerr,
        params=[0,np.nanmax(cutFlux),atomWave,sigma_guess],
        fixed=[True,False,False,False],
        limitedmin=[True,True,True,True],
        limitedmax=[True,False,True,True],
        minpars=[0,0,atomWave-fitMargin,sigma_min],
        maxpars=[0,np.nanmax(cutFlux)*1.5,atomWave+fitMargin,sigma_max],
        quiet=True, shh=True)

    #print
    #print atomWave, ' sigma guess: ', sigma_guess
    #print atomWave, ' sigma min: ', sigma_min
    #print atomWave, ' sigma max: ', sigma_max
    #print atomWave, ' sigma fit: ', yfit[0][3]



    #if atomWave > 34:
        #newWave = np.arange(cutWave[0], cutWave[-1], 0.005)
        #g2 = onedgaussian(newWave, *yfit[0])
        #plt.plot(cutWave, cutFlux)
        #plt.plot(cutWave, yfit[1])
        #plt.plot(newWave, g2)
        #plt.show()
        #plt.close()
        #st()

    #print yfit[0]
    g1 = onedgaussian(cutWave, *yfit[0])
    flux_g1 = sp.integrate.simps(g1, cutWave)
    #print
    #print 'Flux of ', atomWave, ' atomic line: ', flux_g1, ' W/m^2'
    #print

    return flux_g1, yfit
def onedgaussian(x,H,A,dx,w):
    """
    Returns a 1-dimensional gaussian of form
    H+A*np.exp(-(x-dx)**2/(2*w**2))
    """
    return H+A*np.exp(-(x-dx)**2/(2*w**2))
def onedgaussfit_back(xax,data,err=None,params=[0,1,0,1],fixed=[False,False,False,False],limitedmin=[False,False,False,True],
        limitedmax=[False,False,False,False],minpars=[0,0,0,0],maxpars=[0,0,0,0],
        quiet=True,shh=True):
    """
    Inputs:
       xax - x axis
       data - y axis
       err - error corresponding to data

       params - Fit parameters: Height of background, Amplitude, Shift, Width
       fixed - Is parameter fixed?
       limitedmin/minpars - set lower limits on each parameter (default: width>0)
       limitedmax/maxpars - set upper limits on each parameter
       quiet - should MPFIT output each iteration?
       shh - output final parameters?

    Returns:
       Fit parameters
       Model
       Fit errors
       chi2
    """

    def mpfitfun(x,y,err):
        if err == None:
            def f(p,fjac=None): return [0,(y-onedgaussian(x,*p))]
        else:
            def f(p,fjac=None): return [0,(y-onedgaussian(x,*p))/err]
        return f


    #if xax == None:
        #xax = np.arange(len(data))

    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],'parname':"HEIGHT",'error':0} ,
                {'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],'parname':"AMPLITUDE",'error':0},
                {'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2],'parname':"SHIFT",'error':0},
                {'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3],'parname':"WIDTH",'error':0}]

    mp = mpfit.mpfit(mpfitfun(xax,data,err),parinfo=parinfo,quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm

    dof = len(xax) - 3

    #if mpperr==None:
        #print chi2, dof, mpperr
        ##mpperr = [1e99]*4
    #else:
    #mpperr = mpperr * np.sqrt(chi2/dof)

    if mp.status == 0:
        raise Exception(mp.errmsg)

    if not shh:
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print((parinfo[i]['parname'],p," +/- ",mpperr[i]))
        print(("Chi2: ",mp.fnorm," Reduced Chi2: ",mp.fnorm/len(data)," DOF:",len(data)-len(mpp)))


    return mpp,onedgaussian(xax,*mpp),mpperr,chi2,mp.fnorm/len(data)


def onedgaussfit(xax,data,err=(),params=[0,1,0,1],fixed=[False,False,False,False],limitedmin=[False,False,False,True],
        limitedmax=[False,False,False,False],minpars=[0,0,0,0],maxpars=[0,0,0,0],
        quiet=True,shh=True):
    """
    Inputs:
       xax - x axis
       data - y axis
       err - error corresponding to data

       params - Fit parameters: Height of background, Amplitude, Shift, Width
       fixed - Is parameter fixed?
       limitedmin/minpars - set lower limits on each parameter (default: width>0)
       limitedmax/maxpars - set upper limits on each parameter
       quiet - should MPFIT output each iteration?
       shh - output final parameters?

    Returns:
       Fit parameters
       Model
       Fit errors
       chi2
    """

    def mpfitfun(x,y,err):
        if len(err) == 0:
            def f(p,fjac=None): return [0,(y-onedgaussian(x,*p))]
        else:
            def f(p,fjac=None): return [0,(y-onedgaussian(x,*p))/err]
        return f


    #if xax == None:
        #xax = np.arange(len(data))

    parinfo = [ {'n':0,'value':params[0],'limits':[minpars[0],maxpars[0]],'limited':[limitedmin[0],limitedmax[0]],'fixed':fixed[0],'parname':"HEIGHT",'error':0} ,
                {'n':1,'value':params[1],'limits':[minpars[1],maxpars[1]],'limited':[limitedmin[1],limitedmax[1]],'fixed':fixed[1],'parname':"AMPLITUDE",'error':0},
                {'n':2,'value':params[2],'limits':[minpars[2],maxpars[2]],'limited':[limitedmin[2],limitedmax[2]],'fixed':fixed[2],'parname':"SHIFT",'error':0},
                {'n':3,'value':params[3],'limits':[minpars[3],maxpars[3]],'limited':[limitedmin[3],limitedmax[3]],'fixed':fixed[3],'parname':"WIDTH",'error':0}]

    mp = mpfit.mpfit(mpfitfun(xax,data,err),parinfo=parinfo,quiet=quiet)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm

    dof = len(xax) - 3

    #if mpperr==None:
        #print chi2, dof, mpperr
        ##mpperr = [1e99]*4
    #else:
    #mpperr = mpperr * np.sqrt(chi2/dof)

    if mp.status == 0:
        raise Exception(mp.errmsg)

    if not shh:
        for i,p in enumerate(mpp):
            parinfo[i]['value'] = p
            print((parinfo[i]['parname'],p," +/- ",mpperr[i]))
        print(("Chi2: ",mp.fnorm," Reduced Chi2: ",mp.fnorm/len(data)," DOF:",len(data)-len(mpp)))


    return mpp,onedgaussian(xax,*mpp),mpperr,chi2,mp.fnorm/len(data)


def fitLine(atomWave, waveMargin, wave, csub, csuberr, fitMargin, rms, dobreak=0):
    # Prep regions for fitting
    rx = np.where((wave >= atomWave-waveMargin) & (wave <= atomWave+waveMargin))
    cutWave = wave[rx]
    cutFlux = csub[rx]
    cutFluxerr = csuberr[rx]

    # print(atomWave)

    if dobreak == 1:
        print("BREAK")
        st()

    if np.nanmean(cutFlux) < 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Fit with a single gauss.
    fluxGauss, params = atomicFit(atomWave, cutWave, cutFlux, cutFluxerr, fitMargin)

    # if atomWave == 15.55:
    #     plt.close()
    #     plt.plot(cutWave, cutFlux)
    #     plt.show()
    #     plt.close()
    #     st()

    amp = params[0][1]
    position = params[0][2]
    sigma = params[0][3]
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    if params[2] is None:
        return fluxGauss, 0, 0, amp, position, sigma

    # st()
    amp_err = params[2][1]
    position_err = params[2][2]
    sigma_err = params[2][3]
    gaussAreaErr = np.sqrt((amp_err/amp)**2 + (sigma_err/sigma)**2) * fluxGauss
    gaussFeatErr = compute_feature_uncertainty(position, sigma, cutWave, rms)
    gaussSNR = gaussAreaErr/fluxGauss


    if gaussSNR >= 1e5:
        gaussTotalErr = gaussFeatErr
    else:
        gaussTotalErr = gaussFeatErr + gaussAreaErr

    snr = fluxGauss / gaussTotalErr
    waveGauss = cutWave
    spectrumGauss = onedgaussian(cutWave,0,amp,position,sigma)

    return fluxGauss, gaussTotalErr, snr, amp, position, sigma
def linesToFit():
    #atomicLines = [9.66, 10.51, 12.81, 15.55, 17.03, 18.71, 33.48, 34.82, 28.22, 25.9, 32.3238]
    #atomicLineNames = ['H2_97', 'S4_105', 'Ne2_128', 'Ne3_155', 'H2_17', 'S3_187', 'S3_335', 'Si2_348', 'H2_282', 'Fe2_259', 'Fe2_323']
    #atomicPlotNames = ['9.66 H2', '10.5 [S IV]', '12.8 [Ne II]', '15.5 [Ne III]', '17.0 H2', '18.7 [S III]', '33.5 [S III]', '34.8 [Si II]', '28.2 H2', '25.9 [Fe II]', '32.3 [Fe II]???']

    # atomicLines = [9.66, 10.51, 12.81, 15.55, 17.03, 18.71, 33.48, 34.82, 28.22, 25.9]
    # atomicLineNames = ['H2_97', 'S4_105', 'Ne2_128', 'Ne3_155', 'H2_17', 'S3_187', 'S3_335', 'Si2_348', 'H2_282', 'Fe2_259']
    # atomicPlotNames = [r'9.66 H$_2$', '10.5 [S IV]', '12.8 [Ne II]', '15.5 [Ne III]', '17.0 H2', '18.7 [S III]', '33.5 [S III]', '34.8 [Si II]', r'28.2 H$_2$', '25.9 [Fe II]']

    atomicLines = [9.66, 10.51, 12.81, 15.55, 17.03, 18.71, 33.48, 34.82, 28.22, 25.89]
    atomicLineNames = ['H2_97', 'S4_105', 'Ne2_128', 'Ne3_155', 'H2_17', 'S3_187', 'S3_335', 'Si2_348', 'H2_282', 'Fe2_259']
    # atomicPlotNames = [r'9.66 µm H$_2$', '10.51 µm [S ɪᴠ]', '12.81 µm [Ne ɪɪ]', '15.55 µm [Ne ɪɪɪ]', '17.03 µm H$_2$', '18.71 µm [S ɪɪɪ]', '33.48 µm [S ɪɪɪ]', '34.82 µm [Si ɪɪ]', r'28.22 µm H$_2$', '25.89 µm [O ɪᴠ]']
    atomicPlotNames = [r'H$_2$ 9.66', '[S ɪᴠ] 10.5', '[Ne ɪɪ] 12.8', '[Ne ɪɪɪ] 15.5', r'H$_2$ 17.0', '[S ɪɪɪ] 18.7', '[S ɪɪɪ] 33.5', '[Si ɪɪ] 34.8', r'H$_2$ 28.2', '[O ɪᴠ] 25.9']


    return atomicLines, atomicLineNames, atomicPlotNames
def pahsToFit():
    pahLines = np.array([6.2, 7.7, 8.6, 11.2, 12.0, 12.7, 15.8, 16.4, 17.4, 17.8])
    pahLineNames = pahLines.astype(str)
    pahPlotNames = pahLines.astype(str)
    pahPlotNames = np.array(['PAH 6.2', 'PAH 7.7', 'PAH 8.6', 'PAH 11.2', 'PAH 12.0', 'PAH 12.7', 'PAH 15.8', 'PAH 16.4', 'PAH 17.4', 'PAH 17.8'])
    return pahLines, pahLineNames, pahPlotNames
def measureFluxes(waveL, csubL, csubLerr, waveS, csubS, csubSerr, cpos):

    # Which atomic lines to fit.
    #atomicLines = [18.71, 33.48, 34.82, 28.22]
    #atomicLineNames = ['S3_187', 'S3_335', 'Si2_348', 'H2_282']
    #atomicPlotNames = ['18.7 [S III]', '33.5 [S III]', '34.8 [Si II]', '28.2 H2']
    atomicLines, atomicLineNames, atomicPlotNames = linesToFit()
    lineArr = []

    # Which PAH features to fit.
    #pahLines = np.array([6.2, 7.7, 8.6, 11.2, 12.0, 12.7, 16.4, 17.8, 26])
    #pahLineNames = pahLines.astype(str)
    #pahPlotNames = pahLines.astype(str)
    pahLines, pahLineNames, pahPlotNames = pahsToFit()
    pahArr = []

    # RMS.
    rmsSL = measure_112_RMS(waveS, csubS)
    rmsLL = measure_25_RMS(waveL, csubL)


    # // ATOMIC LINES
    for j in range(len(atomicLines)):
        atomWave = atomicLines[j]
        atomName = atomicLineNames[j]
        waveMargin = 0.4
        fitMargin = 0.15

        if atomWave in [12.81, 15.55]:
            continue # HANDLE BLENDED LINES SEPARATELY

        #if atomWave > waveL[0]:
            #irms = rmsLL
            #wave = waveL
            #csub = csubL
            #csuberr = csubLerr
        #else:
        irms = rmsSL
        wave = waveS
        csub = csubS
        csuberr = csubSerr

        dobreak = 0
        lineFlux, lineFluxerr, lineSNR, amp, position, sigma = fitLine(atomWave, waveMargin, wave, csub, csuberr, fitMargin, irms, dobreak=dobreak)

        if atomWave < waveS[-1] and cpos == 5:
            lineArr.append((j, atomWave, 0, 0, 0, amp, position, sigma))
        else:
            lineArr.append((j, atomWave, lineFlux, lineFluxerr, lineSNR, amp, position, sigma))



    # // PAH FEATURES
    for j in range(len(pahLines)):
        pahWave = pahLines[j]
        pahName = pahLineNames[j]

        if pahWave in [12.7]:
            continue # BLENDED, DO SEPARATELY

        #if pahWave > waveL[0]:
            #irms = rmsLL
            #wave = waveL
            #csub = csubL
            #csuberr = csubLerr
        #else:
        irms = rmsSL
        wave = waveS
        csub = csubS
        csuberr = csubSerr

        ## Figure out integration boundaries.
        intmin, intmax = pahIntRanges(pahWave)

        intRange = np.where((wave >= intmin) & (wave <= intmax))
        wRange = wave[intRange]
        cRange = csub[intRange]
        pahFlux = sp.integrate.simps(cRange, wRange)
        pahFluxErr = compute_feature_uncertainty(cRange*0, cRange*0, wRange, irms, manual_range=[intmin,intmax])

        if pahWave < waveS[-1] and cpos == 5:
            pahArr.append((j, pahWave, 0, 0, 0, 0, 0))
        else:
            pahArr.append((j, pahWave, pahFlux, pahFluxErr, pahFlux/pahFluxErr, wRange, cRange))


    # // BLENDED FEATURES -- tack onto appropriate atomic/pah lists after finished.


    return np.array(lineArr), np.array(pahArr)
def measureFluxes2D(wave, flux2D, fluxerr2D, spl2D, csub2D, csuberr2D, waveKnots2D, module):
    # Which atomic lines to fit.
    atomicLines = [18.71, 33.48, 34.82, 28.22]
    atomicLineNames = ['S3_187', 'S3_335', 'Si2_348', 'H2_282']
    atomicPlotNames = ['18.7 [S III]', '33.5 [S III]', '34.8 [Si II]', '28.2 H2']
    lineArr = []

    # Which PAH features to fit.
    pahLines = [11.2]
    pahLineNames = ["11.2"]
    pahPlotNames = ["11.2"]
    pahArr = []

    wrapArr = []

    # Measure RMS.
    if module == "SL":
        rms = measure_112_RMS2D(wave,csub2D)
    elif module == "LL":
        rms = measure_25_RMS2D(wave,csub2D)

    # Choose spatial position.
    nPix = csub2D.shape[1]
    if rms.shape[0] != nPix:
        raise SystemExit("pix don't match.")

    # Iterate over spatial pixels.
    for i in np.arange(0,nPix,1):
        flux = flux2D[:,i]
        fluxerr = fluxerr2D[:,i]
        spl = spl2D[:,i]
        csub = csub2D[:,i]
        csuberr = csuberr2D[:,i]
        irms = rms[i]
        waveKnots1D = waveKnots2D[:,i]


        # NaN nanny.
        if np.all(np.isnan(csub)):
            lineArr.append((i, np.nan, np.nan, np.nan, np.nan, np.nan))
            pahArr.append((i, np.nan, np.nan, np.nan, np.nan))
            wrapArr.append((i, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        # // ATOMIC LINES
        for j in range(len(atomicLines)):
            atomWave = atomicLines[j]
            atomName = atomicLineNames[j]
            waveMargin = 1
            fitMargin = 0.15

            if atomWave >= wave[-1]:
                continue
            if atomWave < wave[0]:
                continue

            #print module, i, j, atomWave, atomName
            dobreak = 0
            lineFlux, lineFluxerr, lineSNR, amp, position, sigma = fitLine(atomWave, waveMargin, wave, csub, csuberr, fitMargin, irms, dobreak=dobreak)
            #lineFlux, lineFluxerr, lineSNR, waveGauss, spectrumGauss = fitLine(atomWave, waveMargin, wave, csub, csuberr, fitMargin, irms, dobreak=dobreak)
            lineArr.append((i, j, atomWave, lineFlux, lineFluxerr, lineSNR))
            wrapArr.append((i, j, atomWave, amp, position, sigma, lineSNR))
            # xpixel, linej, atomwave, lineflux, linefluxerr, linesnr
            #print lineFluxerr, lineSNR


        # // PAH FEATURES
        for j in range(len(pahLines)):
            pahWave = pahLines[j]
            pahName = pahLineNames[j]

            if pahWave >= wave[-1]:
                continue
            if pahWave < wave[0]:
                continue

            #print
            #print "PAH, ", module, i, j, pahWave, pahName

            ## Figure out integration boundaries.
            #intRange, nearest = findRange(wave, waveKnots1D, 11.0, forcefloor=1)
            intRange = np.where((wave >= 10.8) & (wave <= 11.6))
            wRange = wave[intRange]
            cRange = csub[intRange]

            pahFlux = sp.integrate.simps(cRange, wRange)
            pahFluxErr = compute_feature_uncertainty(cRange*0, cRange*0, wRange, irms, manual_range=[10.8,11.6])
            pahArr.append((i, j, pahWave, pahFlux, pahFluxErr))

            #plt.plot(wRange, cRange)
            #plt.axhline(y=0)
            #plt.axvline(x=10.8)
            #plt.axvline(x=11.6)
            #plt.title(str(pahFlux))
            #plt.show()
            #plt.close()

    tt = np.array(lineArr)
    vv = np.array(pahArr)
    ww = np.array(wrapArr)

    return tt, vv, ww
def measureFluxes2Dnew(wave, flux2D, fluxerr2D, spl2D, csub2D, csuberr2D, waveKnots2D, module):
    # Which atomic lines to fit.
    atomicLines = [18.71, 33.48, 34.82, 28.22]
    atomicLineNames = ['S3_187', 'S3_335', 'Si2_348', 'H2_282']
    atomicPlotNames = ['18.7 [S III]', '33.5 [S III]', '34.8 [Si II]', '28.2 H2']
    lineArr = []

    # Which PAH features to fit.
    pahLines = [11.2]
    pahLineNames = ["11.2"]
    pahPlotNames = ["11.2"]
    pahArr = []

    wrapArr = []

    # Measure RMS.
    if module == "SL":
        rms = measure_112_RMS2D(wave,csub2D)
    elif module == "LL":
        rms = measure_25_RMS2D(wave,csub2D)

    # Choose spatial position.
    nPix = csub2D.shape[1]
    if rms.shape[0] != nPix:
        raise SystemExit("pix don't match.")

    # Iterate over spatial pixels.
    for i in np.arange(0,nPix,1):
        flux = flux2D[:,i]
        fluxerr = fluxerr2D[:,i]
        spl = spl2D[:,i]
        csub = csub2D[:,i]
        csuberr = csuberr2D[:,i]
        irms = rms[i]
        waveKnots1D = waveKnots2D[:,i]


        # NaN nanny.
        if np.all(np.isnan(csub)):
            lineArr.append((i, np.nan, np.nan, np.nan, np.nan, np.nan))
            pahArr.append((i, np.nan, np.nan, np.nan, np.nan))
            wrapArr.append((i, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        # // ATOMIC LINES
        for j in range(len(atomicLines)):
            atomWave = atomicLines[j]
            atomName = atomicLineNames[j]
            waveMargin = 1
            fitMargin = 0.15

            if atomWave >= wave[-1]:
                continue
            if atomWave < wave[0]:
                continue

            #print module, i, j, atomWave, atomName
            dobreak = 0
            lineFlux, lineFluxerr, lineSNR, amp, position, sigma = fitLine(atomWave, waveMargin, wave, csub, csuberr, fitMargin, irms, dobreak=dobreak)
            #lineFlux, lineFluxerr, lineSNR, waveGauss, spectrumGauss = fitLine(atomWave, waveMargin, wave, csub, csuberr, fitMargin, irms, dobreak=dobreak)
            lineArr.append((i, j, atomWave, lineFlux, lineFluxerr, lineSNR))
            wrapArr.append((i, j, atomWave, amp, position, sigma, lineSNR))
            # xpixel, linej, atomwave, lineflux, linefluxerr, linesnr
            #print lineFluxerr, lineSNR


        # // PAH FEATURES
        for j in range(len(pahLines)):
            pahWave = pahLines[j]
            pahName = pahLineNames[j]

            if pahWave >= wave[-1]:
                continue
            if pahWave < wave[0]:
                continue

            #print
            #print "PAH, ", module, i, j, pahWave, pahName

            ## Figure out integration boundaries.
            #intRange, nearest = findRange(wave, waveKnots1D, 11.0, forcefloor=1)
            intRange = np.where((wave >= 10.8) & (wave <= 11.6))
            wRange = wave[intRange]
            cRange = csub[intRange]

            pahFlux = sp.integrate.simps(cRange, wRange)
            pahFluxErr = compute_feature_uncertainty(cRange*0, cRange*0, wRange, irms, manual_range=[10.8,11.6])
            pahArr.append((i, j, pahWave, pahFlux, pahFluxErr))

            #plt.plot(wRange, cRange)
            #plt.axhline(y=0)
            #plt.axvline(x=10.8)
            #plt.axvline(x=11.6)
            #plt.title(str(pahFlux))
            #plt.show()
            #plt.close()

    tt = np.array(lineArr)
    vv = np.array(pahArr)
    ww = np.array(wrapArr)

    return tt, vv, ww
def jy(f,w):
    return w**2 * f / 3e-12
def si(f,w):
    return 3e-12 * f / w**2
def ctPoints(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  3.0,    4.0,    5.5,    5.9,    6.6,    7.2,    \
            8.20,   8.8,    9.4,   9.85,   \
            10.65,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      1,      1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      0,      0,      .12,    0,      \
            .16,    0,      0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   0,      0,      1,      1,      0,      1,      1,      \
            0,      0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm

def ctPoints_special(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  3.0,    4.0,    5.0,    5.5,    6.1,    7.4,    \
            8.20,   8.8,    9.4,   9.85,   \
            10.65,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      0,      0,      0,      0,      1,      \
            1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      0,      0,      .12,    0,      \
            .16,    0,      0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   0,      0,      0,      0,      1,      1,      0,      \
            0,      0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm


def ctPointsSolo(module):

    if module == 'SL':
        cl=[5.5,5.9,6.6,7.2,   8.20,8.8,9.15,9.85, \
                10.65,10.9,11.8,12.23,13.15, 13.8,14.7]
        find_min=[1,1,1,1,    1,1,1,1, \
                1,1,1,1,1, 1,1]
        width=[0,0,.12,0,    .16,0,0,0, \
                .10,.1,0.1,0,0, 0,0]
        smooth_flag=[1,1,1,1, 0,0,1,0, \
                1,1,0,0,0, 1,1]

    elif module == 'LL':
        cl = [14.3,15.,16.2,16.9,     17.6,18.1,18.5,19.,19.8,21] # 20 !!!!!!!!!!!!
        find_min = [1,0,1,1,       1,1,0,1,1,1]
        width = [0,.15,.11,.2,   0,0,0,.11,0,0]
        smooth_flag = [1,1,0,0,    0,0,0,0,1,1]

        cl+=[23,25,27.8,29,31.5,35.8,38.1]
        find_min+=[1,1,1,1,1,1,1]
        width+=[0,0,0,0,0,0.35,0.2]
        smooth_flag+=[1,1,0,1,1,1,1]

    return cl, find_min, width, smooth_flag
def ctpoints_iso_HD100546(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  5.5,    5.9,    6.6,    7.4,    \
            8.40,   9.0,    9.4,   9.85,   \
            10.65,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      .12,    0,      \
            .16,    0,      0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   1,      1,      0,      1,      1,      \
            0,      0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm
def ctpoints_spitzer_HD31293(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  5.5,    5.9,    6.6,    7.4,    \
            8.40,   9.05,    9.4,   9.85,   \
            10.65,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      .12,    0,      \
            .16,    0,      0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   1,      1,      0,      1,      1,      \
            0,      0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm

def ctpoints_spitzer_whatever(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  5.5,    5.9,    6.6,    7.4,    \
            8.40,   9.0,    9.4,   9.85,   \
            10.65,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      .12,    0,      \
            .16,    0,      0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   1,      1,      0,      1,      1,      \
            0,      0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm

def ctpoints_spitzer_HD100453(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  5.5,    5.9,    6.6,    7.4,    \
            8.40,   9.0,    9.4,   9.85,   \
            10.65,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      .12,    0,      \
            .16,    0,      0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   1,      1,      0,      1,      1,      \
            0,      0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm
def ctpoints_spitzer_HD72106(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  5.5,    5.9,    6.6,    7.4,    \
            8.40,   9.06,    9.4,   9.85,   \
            10.65,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      .12,    0,      \
            .16,    0,      0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   1,      1,      0,      1,      1,      \
            0,      0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm
def ctpoints_spitzer_HD36917(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  5.5,    5.9,    6.6,    7.4,    \
            8.40,   9.0,    9.85,   \
            10.65,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      1,      1,      1,      \
            1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      .12,    0,      \
            .16,    0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   1,      1,      0,      1,      1,      \
            0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm
def ctpoints_spitzer_HD135344(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  5.5,    5.9,    6.6,    7.4,    \
            8.47,   9.0,    9.4,   9.85,   \
            10.65,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      .12,    0,      \
            .16,    0,      0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   1,      1,      0,      1,      1,      \
            0,      0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm
def ctpoints_spitzer_HD141569(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  5.5,    5.9,    6.6,    7.4,    \
            8.40,   9.0,    9.4,   9.85,   \
            10.65,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      1,      1,      0,      \
            1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      .12,    0,      \
            .16,    0,      0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   1,      1,      0,      1,      1,      \
            0,      0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm
def ctpoints_spitzer_HD169142(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  5.5,    5.9,    6.6,    7.4,    \
            8.40,   9.0,    9.4,   9.85,   \
            10.65,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      1,      1,      0,      \
            1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      .12,    0,      \
            .16,    0,      0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   1,      1,      0,      1,      1,      \
            0,      0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm
def ctpoints_spitzer_HD245906(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  5.5,    5.9,    6.6,    7.05,    \
            8.45,   9.0,    9.4,   9.85,   \
            10.5,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      1,      0,      1,      \
            1,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      .12,    0,      \
            .16,    0,      0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   1,      1,      0,      1,      1,      \
            0,      0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm
def ctpoints_spitzer_V590_Mon(special_cut=0, skip_8um_bump=0):

    ########################################
    # Nominal anchor positions.
    cl = [  5.5,    5.9,    6.6,    7.3,    \
            8.40,   8.95,    9.4,   9.85,   \
            10.65,  10.9,   11.8,   12.23,  \
            13.15,  13.8]
    ####################
    cl+= [  14.5,   14.9,   15.35,  16.2,   16.9,   \
            17.6,   18.1,   18.5,   19.,    \
            19.8,   21]
    ####################
    cl+= [  23,     25,     27.8,   29,     \
            31.5,   35.8,   38.1]
    ########################################


    ########################################
    # Find minimum?
    fmn= [  1,      1,      1,      1,      1,      \
            0,      1,      1,      1,      \
            1,      1,      1,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      0,      1,      1,      \
            1,      1,      0,      1,      \
            1,      1]
    ####################
    fmn+=[  1,      1,      1,      1,      \
            1,      1,      1]
    ########################################


    ########################################
    # Search width.
    wth= [  0,      0,      0,      .12,    0,      \
            .16,    0,      0,      0,      \
            .10,    .1,     0.1,    0,      \
            0,      0]
    ####################
    wth+=[  0,      .15,    .11,    .2,     \
            0,      0,      0,      .11,    \
            0,      0]
    ####################
    wth+=[  0,      0,      0,      0,      \
            0,      0.35,   0.2]
    ########################################


    ########################################
    # Smooth spectra?
    sm= [   1,      1,      0,      1,      1,      \
            0,      0,      1,      0,      \
            1,      1,      0,      0,      \
            0,      1]
    ####################
    sm+=[   1,      1,      0,      0,      \
            0,      0,      0,      0,      \
            1,1]
    ####################
    sm+=[   1,      1,      0,      1,      \
            1,      1,      1]
    ########################################

    if special_cut == 1:
        thedx = np.where(np.array(cl) == 15.35)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    if skip_8um_bump == 1:
        thedx = np.where(np.array(cl) == 8.20)[0][0]
        del cl[thedx]
        del fmn[thedx]
        del wth[thedx]
        del sm[thedx]

    return cl, fmn, wth, sm

def ctPointsAmes(special_cut=0, skip_8um_bump=0, ames_data='0', myobjname='0'):

    print(myobjname)
    # st()

    if ames_data == 'iso':
        if myobjname == 'HD100546':
            cl, fmn, wth, sm = ctpoints_iso_HD100546(special_cut, skip_8um_bump)
        else:
            cl, fmn, wth, sm = ctPoints(special_cut, skip_8um_bump)

    elif ames_data == 'spitzer':
        if myobjname == 'HD31293':
            cl, fmn, wth, sm = ctpoints_spitzer_HD31293(special_cut, skip_8um_bump)
        elif myobjname == 'HD36917':
            cl, fmn, wth, sm = ctpoints_spitzer_HD36917(special_cut, skip_8um_bump)
        elif myobjname == 'HD72106':
            cl, fmn, wth, sm = ctpoints_spitzer_HD72106(special_cut, skip_8um_bump)
        elif myobjname == 'HD100453':
            cl, fmn, wth, sm = ctpoints_spitzer_HD100453(special_cut, skip_8um_bump)
        elif myobjname == 'HD135344':
            cl, fmn, wth, sm = ctpoints_spitzer_HD135344(special_cut, skip_8um_bump)
        elif myobjname == 'HD141569':
            cl, fmn, wth, sm = ctpoints_spitzer_HD141569(special_cut, skip_8um_bump)
        elif myobjname == 'HD169142':
            cl, fmn, wth, sm = ctpoints_spitzer_HD169142(special_cut, skip_8um_bump)
        elif myobjname == 'HD245906':
            cl, fmn, wth, sm = ctpoints_spitzer_HD245906(special_cut, skip_8um_bump)
        elif myobjname == 'V590_Mon':
            cl, fmn, wth, sm = ctpoints_spitzer_V590_Mon(special_cut, skip_8um_bump)

        else:
            cl, fmn, wth, sm = ctPoints(special_cut, skip_8um_bump)



    else:
        cl, fmn, wth, sm = ctPoints(special_cut, skip_8um_bump)

    return cl, fmn, wth, sm

def getPlateauSpl(wave, flux, fluxerr, splineFlux, waveKnotsFinal, fluxKnotsFinal):

    include_1821 = includePlat1821()

    if include_1821:
        dx5 = find_nearest.find_nearest(waveKnotsFinal, 5)
        dx10 = find_nearest.find_nearest(waveKnotsFinal, 9.5)
        dx15 = find_nearest.find_nearest(waveKnotsFinal, 15)
        dx18 = find_nearest.find_nearest(waveKnotsFinal, 18)
        dx21 = find_nearest.find_nearest(waveKnotsFinal, 21)

        w5, f5 = waveKnotsFinal[dx5], fluxKnotsFinal[dx5]
        w10, f10 = waveKnotsFinal[dx10], fluxKnotsFinal[dx10]
        w15, f15 = waveKnotsFinal[dx15], fluxKnotsFinal[dx15]
        w18, f18 = waveKnotsFinal[dx18], fluxKnotsFinal[dx18]
        w21, f21 = waveKnotsFinal[dx21], fluxKnotsFinal[dx21]

        lowwave = wave[np.where(wave < w5)]
        lowflux = flux[np.where(wave < w5)]
        lowfluxerr = fluxerr[np.where(wave < w5)]
        lowspline = splineFlux[np.where(wave < w5)]
        hiwave = wave[np.where(wave >= w21)]
        hiflux = flux[np.where(wave >= w21)]
        hifluxerr = fluxerr[np.where(wave >= w21)]
        hispline = splineFlux[np.where(wave >= w21)]

        z1 = np.polyfit([w5,w10],[f5,f10],1)
        p1 = np.poly1d(z1)
        dx1 = [np.where((wave >= w5) & (wave < w10))][0]
        w1 = wave[dx1]
        f1 = p1(w1)
        fe1 = fluxerr[dx1]

        z2 = np.polyfit([w10,w15],[f10,f15],1)
        p2 = np.poly1d(z2)
        dx2 = [np.where((wave >= w10) & (wave < w15))][0]
        w2 = wave[dx2]
        f2 = p2(w2)
        fe2 = fluxerr[dx2]

        z3 = np.polyfit([w15,w18],[f15,f18],1)
        p3 = np.poly1d(z3)
        dx3 = [np.where((wave >= w15) & (wave < w18))][0]
        w3 = wave[dx3]
        f3 = p3(w3)
        fe3 = fluxerr[dx3]

        z4 = np.polyfit([w18,w21],[f18,f21],1)
        p4 = np.poly1d(z4)
        dx4 = [np.where((wave >= w18) & (wave < w21))][0]
        w4 = wave[dx4]
        f4 = p4(w4)
        fe4 = fluxerr[dx4]

        finalPlatWave = np.concatenate((lowwave,w1,w2,w3,w4,hiwave))
        finalPlatFlux = np.concatenate((lowspline,f1,f2,f3,f4,hispline))
        finalPlatFluxerr = np.concatenate((lowfluxerr,fe1,fe2,fe3,fe4,hifluxerr))

        return finalPlatWave, finalPlatFlux, finalPlatFluxerr, dx1, dx2, dx3, dx4

    else:

        dx5 = find_nearest.find_nearest(waveKnotsFinal, 5)
        dx10 = find_nearest.find_nearest(waveKnotsFinal, 9.5)
        dx15 = find_nearest.find_nearest(waveKnotsFinal, 15)
        dx18 = find_nearest.find_nearest(waveKnotsFinal, 18)

        w5, f5 = waveKnotsFinal[dx5], fluxKnotsFinal[dx5]
        w10, f10 = waveKnotsFinal[dx10], fluxKnotsFinal[dx10]
        w15, f15 = waveKnotsFinal[dx15], fluxKnotsFinal[dx15]
        w18, f18 = waveKnotsFinal[dx18], fluxKnotsFinal[dx18]

        lowwave = wave[np.where(wave < w5)]
        lowflux = flux[np.where(wave < w5)]
        lowfluxerr = fluxerr[np.where(wave < w5)]
        lowspline = splineFlux[np.where(wave < w5)]
        hiwave = wave[np.where(wave >= w18)]
        hiflux = flux[np.where(wave >= w18)]
        hifluxerr = fluxerr[np.where(wave >= w18)]
        hispline = splineFlux[np.where(wave >= w18)]

        z1 = np.polyfit([w5,w10],[f5,f10],1)
        p1 = np.poly1d(z1)
        dx1 = [np.where((wave >= w5) & (wave < w10))][0]
        w1 = wave[dx1]
        f1 = p1(w1)
        fe1 = fluxerr[dx1]

        z2 = np.polyfit([w10,w15],[f10,f15],1)
        p2 = np.poly1d(z2)
        dx2 = [np.where((wave >= w10) & (wave < w15))][0]
        w2 = wave[dx2]
        f2 = p2(w2)
        fe2 = fluxerr[dx2]

        z3 = np.polyfit([w15,w18],[f15,f18],1)
        p3 = np.poly1d(z3)
        dx3 = [np.where((wave >= w15) & (wave < w18))][0]
        w3 = wave[dx3]
        f3 = p3(w3)
        fe3 = fluxerr[dx3]

        finalPlatWave = np.concatenate((lowwave,w1,w2,w3,hiwave))
        finalPlatFlux = np.concatenate((lowspline,f1,f2,f3,hispline))
        finalPlatFluxerr = np.concatenate((lowfluxerr,fe1,fe2,fe3,hifluxerr))

        return finalPlatWave, finalPlatFlux, finalPlatFluxerr, dx1, dx2, dx3, 0
def getSmoothSize():
    return 8
def getWindowSize():
    return 0.12
def measurePlateauFlux(wave, flux, fluxerr, splineFlux, waveKnotsFinal, fluxKnotsFinal, csub, csuberr):

        # Make a version of the spectrum for measuring the plateaus.
        finalPlatWave, finalPlatFlux, finalPlatFluxerr, dx1, dx2, dx3, dx4 = getPlateauSpl(wave, flux, fluxerr, splineFlux, waveKnotsFinal, fluxKnotsFinal)

        if np.nanmax(finalPlatWave) < 15:
            # st()
            return 0, 0, 0, 0, 0, 0, 0, 0, [0,0,0,0], [0,0,0,0], finalPlatWave, finalPlatWave*0, finalPlatWave*0
        else:

            if isinstance(dx4, list):

                # Subtract plat spline from normal spline.
                subWave = finalPlatWave
                subPlatFlux = splineFlux - finalPlatFlux
                subPlatFluxerr = finalPlatFluxerr

                subWave = subWave
                subPlatFlux = si(subPlatFlux, subWave)
                subPlatFluxerr = si(subPlatFluxerr, subWave)

                platflux1 = sp.integrate.simps(subPlatFlux[dx1], subWave[dx1])
                platflux2 = sp.integrate.simps(subPlatFlux[dx2], subWave[dx2])
                platflux3 = sp.integrate.simps(subPlatFlux[dx3], subWave[dx3])
                platflux4 = sp.integrate.simps(subPlatFlux[dx4], subWave[dx4])

                irms1 = measure_112_RMS(wave, si(csub,wave))
                irms2 = measure_25_RMS(wave, si(csub,wave))

                platfluxerr1 = compute_feature_uncertainty(subPlatFlux[dx1]*0, subPlatFlux[dx1]*0, subWave[dx1], irms1, manual_range=[subWave[dx1][0], subWave[dx1][-1]])
                platfluxerr2 = compute_feature_uncertainty(subPlatFlux[dx2]*0, subPlatFlux[dx2]*0, subWave[dx2], irms1, manual_range=[subWave[dx2][0], subWave[dx2][-1]])
                platfluxerr3 = compute_feature_uncertainty(subPlatFlux[dx3]*0, subPlatFlux[dx3]*0, subWave[dx3], irms1, manual_range=[subWave[dx3][0], subWave[dx3][-1]])
                platfluxerr4 = compute_feature_uncertainty(subPlatFlux[dx4]*0, subPlatFlux[dx4]*0, subWave[dx4], irms2, manual_range=[subWave[dx4][0], subWave[dx4][-1]])

                #print platflux1, platfluxerr1
                #print platflux2, platfluxerr2
                #print platflux3, platfluxerr3

                arrWaves = [subWave[dx1], subWave[dx2], subWave[dx3], subWave[dx4]]
                arrFluxes = [subPlatFlux[dx1], subPlatFlux[dx2], subPlatFlux[dx3], subPlatFlux[dx4]]

            else:

                # Subtract plat spline from normal spline.
                subWave = finalPlatWave
                subPlatFlux = splineFlux - finalPlatFlux
                subPlatFluxerr = finalPlatFluxerr

                subWave = subWave
                subPlatFlux = si(subPlatFlux, subWave)
                subPlatFluxerr = si(subPlatFluxerr, subWave)

                irms1 = measure_112_RMS(wave, si(csub,wave))
                irms2 = measure_25_RMS(wave, si(csub,wave))

                if len(dx1[0]) == 0:
                    platflux1 = 0
                    platfluxerr1 = 0
                else:
                    platflux1 = sp.integrate.simps(subPlatFlux[dx1], subWave[dx1])
                    platfluxerr1 = compute_feature_uncertainty(subPlatFlux[dx1]*0, subPlatFlux[dx1]*0, subWave[dx1], irms1, manual_range=[subWave[dx1][0], subWave[dx1][-1]])

                if len(dx2[0]) == 0:
                    platflux2 = 0
                    platfluxerr2 = 0
                else:
                    platflux2 = sp.integrate.simps(subPlatFlux[dx2], subWave[dx2])
                    platfluxerr2 = compute_feature_uncertainty(subPlatFlux[dx2]*0, subPlatFlux[dx2]*0, subWave[dx2], irms1, manual_range=[subWave[dx2][0], subWave[dx2][-1]])

                if len(dx3[0]) == 0:
                    platflux3 = 0
                    platfluxerr3 = 0
                else:
                    platflux3 = sp.integrate.simps(subPlatFlux[dx3], subWave[dx3])
                    platfluxerr3 = compute_feature_uncertainty(subPlatFlux[dx3]*0, subPlatFlux[dx3]*0, subWave[dx3], irms1, manual_range=[subWave[dx3][0], subWave[dx3][-1]])

                #print platflux1, platfluxerr1
                #print platflux2, platfluxerr2
                #print platflux3, platfluxerr3

                arrWaves = [subWave[dx1], subWave[dx2], subWave[dx3]]
                arrFluxes = [subPlatFlux[dx1], subPlatFlux[dx2], subPlatFlux[dx3]]

                platflux4 = 0
                platfluxerr4 = 1

            return platflux1, platfluxerr1, platflux2, platfluxerr2, platflux3, platfluxerr3, platflux4, platfluxerr4, arrWaves, arrFluxes, finalPlatWave, finalPlatFlux, finalPlatFluxerr
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
def lineids(ax, wave, flux):

    pahLines, pahLineNames, pahPlotNames = pahsToFit()
    atomicLines, atomicLineNames, atomicPlotNames = linesToFit()

    finalpahLines = []
    finalpahLineNames = []
    finalpahPlotNames = []

    for i in range(len(pahLines)):
        wavein = pahLines[i]
        if wave[0] <= wavein <= wave[-1]:
            finalpahLines.append(pahLines[i])
            finalpahLineNames.append(pahLineNames[i])
            finalpahPlotNames.append(pahPlotNames[i])

    finalatomicLines = []
    finalatomicLineNames = []
    finalatomicPlotNames = []

    for i in range(len(atomicLines)):
        wavein = atomicLines[i]
        if wave[0] <= wavein <= wave[-1]:
            finalatomicLines.append(atomicLines[i])
            finalatomicLineNames.append(atomicLineNames[i])
            finalatomicPlotNames.append(atomicPlotNames[i])

    allWave = np.hstack((finalpahLines,finalatomicLines))
    allNames = np.hstack((finalpahPlotNames,finalatomicPlotNames))

    # import matplotlib as mpl
    # mpl.get_backend()

    # lid.plot_line_ids(wave, flux, allWave, allNames, label1_size=[10]*len(allWave), extend=True, ax=ax)#, )
    # lid.plot_line_ids(wave, flux, allWave, allNames, label1_size=[10]*len(allWave), ax=ax)#, )

    # pk = lid.initial_plot_kwargs()
    # pk['color'] = "0.2"
    # pk['lw'] = 0.5
    # pk['linestyle'] = ":"
    # linestyle=":", color="0.2", lw=0.5,

    print()
    print()
    print(lid.__version__)
    print()
    print()

    lid.plot_line_ids(wave, flux, allWave, allNames,
        label1_size=[10]*len(allWave), extend=True, ax=ax)


    

    return
def doAnno(snr, atomWave, ax, plotUnits, ftype):
    if snr >= 3:
        pcolor = 'g'
    elif snr < 3:
        pcolor = 'r'
    else:
        pcolor = 'k'

    tsize = 12
    myoff = 0
    myha = 'center'
    # strsnr = str(decimal.Decimal(snr).quantize(decimal.Decimal("1")))
    strsnr = quant_str(snr, precision="1")

    if plotUnits == 'jy':
        #print "good"
        if ftype == 'pah':
            ylow = -15
        else:
            ylow = -10
    elif plotUnits == 'si':
        ymin, ymax = ax.get_ylim()
        yrange = ymax-ymin
        if ftype == 'pah':
            ylow = ymin + 0.05*yrange
        else:
            ylow = ymin + 0.10*yrange

    #print ylow
    #ylow = 0

    ax.annotate(
        strsnr,
        xy = (atomWave, ylow), xytext = (myoff,0), size=tsize, color='w',
        textcoords = 'offset points', ha = myha, va = 'bottom',
        bbox = dict(fc = 'w', alpha = 1))
    ax.annotate(
        strsnr,
        xy = (atomWave, ylow), xytext = (myoff,0), size=tsize,
        textcoords = 'offset points', ha = myha, va = 'bottom',
        bbox = dict(fc = pcolor, alpha = 0.3))

    #st()

    return
def plotResultsSolo(savename, fitResults, wave, flux, fluxerr, plotUnits='jy'):

    fluxAtomics, fluxPAHs, fluxCont, fluxPlats, csub, csuberr, spline, pahArrRanges, wknots, fknots, finalPlatWave, finalPlatFlux, finalPlatFluxerr, allpahflux, allpahfluxerr, allplatflux, allplatfluxerr = fitResults
    #plotUnits = 'jy'
    #plotUnits = 'si'

    sname = savename.split('.pdf')[0] + '_solo.pdf'


    if zoom != 0:
        z = np.where((wave > 5) & (wave < zoom))
        wave = wave[z]
        flux = flux[z]
        fluxerr = fluxerr[z]
        spline = spline[z]
        csub = csub[z]
        csuberr = csuberr[z]
        fknots = fknots[np.where((wknots > 5 ) & (wknots < zoom))]
        wknots = wknots[np.where((wknots > 5 ) & (wknots < zoom))]
        #st()
        finalPlatFlux = finalPlatFlux[z]
        finalPlatWave = finalPlatWave[z]
        #st()
        #ax1.set_xlim(xmax=20)
        #ax1.set_ylim(ymin=-100, ymax=1500)


    ### wave, flux, fluxerr, continuum, continuum_plateau
    ##data_in = np.loadtxt(file_in, delimiter=',', skiprows=1, dtype=str)
    ##data_in = data_in[:,0:-1].astype(float).T

    ### Identify input data
    ##wave = data_in[0]
    ##flux = data_in[1]
    ##fluxerr = data_in[2]
    ##cont = data_in[3]
    ##contplat = data_in[4]

    ### Convert inputs to W/m2/um
    ##flux = watts(wave,flux)
    ##fluxerr = watts(wave,fluxerr)
    ##cont = watts(wave,cont)
    ##contplat = watts(wave,contplat)
    ##mylw = 3

    ## Make fig
    ##fig = plt.figure(figsize=((18,6)))
    #fig = plt.figure(figsize=((9,6)))
    #ax = fig.add_subplot(111)

    ## Define colours
    #dacolorz = sns.color_palette("pastel")
    ##dacolorz = sns.color_palette()
    #color1 = dacolorz[0]
    #color2 = dacolorz[1]
    #color3 = dacolorz[4]
    #color4 = dacolorz[2]
    #datacolor = sns.color_palette()[0]

    ## Plot data
    #ax.errorbar(wave, flux*1e11, yerr=fluxerr*1e11, label='Data', lw=1, zorder=1)
    ##ax.fill_between(wave, 0, flux, alpha=0.4, color='0.5', zorder=1)
    #ax.fill_between(wave, 0, flux*1e11, color=dacolorz[0], zorder=1)
    #ax.plot(wave, cont*1e11, lw=2, color='0.35')
    #ax.fill_between(wave, 0, cont*1e11, alpha=0.8, color=dacolorz[1], zorder=2, hatch='///', edgecolor='0.35')

    ## Plot parameters
    #malpha = 0.7
    #linecolor = 'k'
    #mylw2 = 1
    #ax.set_ylabel(r'Flux density [$10^{-11} W/m^2/ \mu m$]', fontsize=16)
    #ax.set_xlabel(r'Wavelength [$\mu m$]', fontsize=16)
    #ax.set_axis_bgcolor("w")
    ##ax.tick_params(labelsize=14)
    ##ax.set_xlim(xmin=10,xmax=19.5)

    #ax.set_xlim(xmin=10,xmax=15)

    #ax.tick_params(which='minor', length=5, width=1)
    #ax.tick_params(which='major', length=8, width=1)
    #ax.minorticks_on()





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


    ## ================== #
    ## Make figure.
    #fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(12,8))






    # Make fig
    #fig = plt.figure(figsize=((18,6)))
    fig = plt.figure(figsize=((12,8)))
    fig.subplots_adjust(bottom=0.3)
    ax = fig.add_subplot(111)

    # Define colours
    dacolorz = sns.color_palette("pastel")
    #dacolorz = sns.color_palette()
    color1 = dacolorz[0]
    color2 = dacolorz[1]
    color3 = dacolorz[4]
    color4 = dacolorz[2]
    datacolor = sns.color_palette()[0]

    # Plot data
    #ax.fill_between(wave, 0, fplat, alpha=0.8, color='w', hatch='\\', edgecolor='k', lw=0.01, zorder=-5)
    ax.fill_between(wave, fplat, spline, alpha=0.8, color=dacolorz[1], hatch='', edgecolor='k', lw=0.01, zorder=-5)
    ax.fill_between(wave, spline, flux, color=dacolorz[0], zorder=1)
    ax.errorbar(wave, flux, yerr=fluxerr, label='Data', lw=1, zorder=1)


    ax.plot(wave, fplat, ls='-', lw=0.2, color='w', zorder=10)
    ax.plot(wave, fplat, ls='--', lw=0.2, color='0.35', zorder=10)
    ax.plot(wave, spline, ls='-', lw=0.2, color='k', zorder=10)


    lineids(ax, wave, flux)

    #st()

    #ax.fill_between(wave, 0, flux, alpha=0.4, color='0.5', zorder=1)
    #ax.plot(wave, fplat, lw=0.5, color='0.35', zorder=10)
    #ax.plot(wave, flux, lw=1, color='k')
    #ax.fill_between(wave, 0, fplat, alpha=0.8, color=dacolorz[3], zorder=2, hatch='', edgecolor='0.35')
    #ax.fill_between(wave, 0, fplat, alpha=1, color='w', zorder=2, hatch='', edgecolor='0.35')


    #ax.fill_between(wave, fplat, spline, alpha=1, color='w', zorder=2, hatch='///', edgecolor='0.35')




    # Plot parameters
    malpha = 0.7
    linecolor = 'k'
    mylw2 = 1
    #ax.set_ylabel(r'Flux density [$10^{-11} W/m^2/ \mu m$]', fontsize=16)
    #ax.set_ylabel(r'Flux density ($MJy/sr$)', fontsize=14)
    # ax.set_ylabel(r'Surface brightness ($MJy/sr$)', fontsize=14)
    # ax.set_xlabel(r'Wavelength ($\mu m$)', fontsize=14)
    
    ax.set_ylabel('Surface brightness (MJy/sr)', fontsize=14)
    ax.set_xlabel('Wavelength (µm)', fontsize=14)

    ax.set_axis_bgcolor("w")
    #ax.tick_params(labelsize=14)
    #ax.set_xlim(xmin=10,xmax=19.5)

    #ax.set_xlim(xmin=5,xmax=37)
    #ax.set_ylim(ymax=80)


    #if zoom == 1:
        #ax.set_xlim(xmax=20)
        #ax.set_ylim(ymin=-100, ymax=1500)


    #leg = ax.legend(loc=0)
    #st()

    #Create custom artists
    #simArtist = Rectangle((0,1),(0,0), color='k', marker='o', linestyle='')
    blackline = plt.Line2D((0,1),(0,0), color='k', lw=2)
    greyline = plt.Line2D((0,1),(0,0), color='0.35', ls='-', lw=1)

    green_patch = mpatches.Patch(color=dacolorz[1], label='Plateaus')
    #plt.legend(handles=[red_patch])

    #Create legend from custom artist/label lists
    ax.legend([green_patch, blackline, greyline], ['Plateaus', 'Local spline', 'Global spline'], loc=0, frameon=0, bbox_to_anchor=(0.97,0.24), fontsize=12)


    ax.tick_params(which='minor', length=5, width=1)
    ax.tick_params(which='major', length=8, width=1)
    ax.minorticks_on()



    ###### ================== #
    ###### AX ZERO
    ######
    ###### Plot plateaus.
    #####ax0.plot(finalPlatWave, fplat, '--', color='red', lw=1.5)
    #####ax0.fill_between(finalPlatWave, fplat, spline, color='pink', edgecolor='0.5', lw=0)
    ######
    ###### Plot spectrum to evaluate fits.
    #####ax0.errorbar(wave, flux, fluxerr, label='flux', lw=2, color='0.15')
    #####ax0.plot(wave, spline, label='spline', color='red', lw=1.5)
    #####ax0.plot(wknots, fknots, 'o', ms=5, color='deepskyblue', mew=1)
    #lineids(ax0, wave, flux)
    #
    ## RMS zones, and tidy up.
    #ax0.minorticks_on()
    #ax0.set_xticklabels([])
    #ax0.set_ylabel('Flux density (MJy/sr)', fontsize=12)
    #rmsminS, rmsmaxS = rms('SL')
    #rmsminL, rmsmaxL = rms('LL')
    #ax1.axvspan(rmsminS, rmsmaxS, color='c', alpha=0.3)
    #ax1.axvspan(rmsminL, rmsmaxL, color='c', alpha=0.3)


    ## ================== #
    ## AX ONE
    ##
    ## Plot crap.
    #ax1.errorbar(wave, csub, csuberr, label='fluxcsub', lw=1, color='0.2')
    ##
    ## Plot features.
    #for i in range(len(fluxAtomics)):
        ##lineArr.append((j, atomWave, lineFlux, lineFluxerr, lineSNR, amp, position, sigma))
        #inatom = fluxAtomics[i]
        #if plotUnits == 'jy':
            #realOG = jy(onedgaussian(wave,0,inatom[5],inatom[6],inatom[7]), wave)
        #elif plotUnits == 'si':
            #realOG = onedgaussian(wave,0,inatom[5],inatom[6],inatom[7])
        #ax1.fill_between(wave, 0, realOG, color='salmon')
        #linesnr = inatom[4]
        #linewave = inatom[1]
        ##doAnno(linesnr, linewave, ax, plotUnits, ftype='atomic')
    ##
    #if fluxPAHs.shape[0] != len(pahArrRanges):
        #raise SystemExit()
    ##
    #for i in range(len(fluxPAHs)):
        ##pahArr.append((j, pahWave, pahFlux, pahFluxErr, pahFlux/pahFluxErr))
        #inpah = fluxPAHs[i]
        #featWave = pahArrRanges[i][0]
        #featFlux = pahArrRanges[i][1]
        #if plotUnits == 'jy':
            #ax1.fill_between(featWave, 0, jy(featFlux, featWave), color='lightgreen')
        #elif plotUnits == 'si':
            #ax1.fill_between(featWave, 0, featFlux, color='lightgreen')
        #pahsnr = inpah[4]
        #pahwave = inpah[1]
        ##doAnno(pahsnr, pahwave, ax, plotUnits, ftype='pah')
    ##
    ##ax1.axhline(y=0, color='k', ls='-')
    #ax1.set_ylabel('Residuals', fontsize=12)
    #ax1.set_xlabel('Wavelength (microns)', fontsize=12)
    #yticks = ax1.yaxis.get_major_ticks()
    ##yticks[0].label1.set_visible(False)
    #yticks[-1].label1.set_visible(False)


    # ================== #
    # Clean up.
    #fig.tight_layout()
    fig.savefig(sname, format='pdf', bbox_inches='tight')
    fig.clear()
    plt.close()

    return
def plotResultsSoloNew(savename, fitResults, wave, flux, fluxerr, plotUnits='jy', zoom=0, zody_flux=None, zody_scale_factor=1, ymax=None):

    fluxAtomics, fluxPAHs, fluxCont, fluxPlats, csub, csuberr, spline, pahArrRanges, wknots, fknots, finalPlatWave, finalPlatFlux, finalPlatFluxerr, allpahflux, allpahfluxerr, allplatflux, allplatfluxerr = fitResults
    #plotUnits = 'jy'
    #plotUnits = 'si'

    #sname = savename.split('.pdf')[0] + '_solo.pdf'
    sname = savename

    if zoom != 0:
        z = np.where((wave > 5) & (wave < zoom))
        wave = wave[z]
        flux = flux[z]
        fluxerr = fluxerr[z]
        spline = spline[z]
        csub = csub[z]
        csuberr = csuberr[z]
        fknots = fknots[np.where((wknots > 5 ) & (wknots < zoom))]
        wknots = wknots[np.where((wknots > 5 ) & (wknots < zoom))]
        #st()
        finalPlatFlux = finalPlatFlux[z]
        finalPlatWave = finalPlatWave[z]
        #st()
        #ax1.set_xlim(xmax=20)
        #ax1.set_ylim(ymin=-100, ymax=1500)

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

    # Make fig
    #fig = plt.figure(figsize=((18,6)))
    fig = plt.figure(figsize=((12,8)))
    fig.subplots_adjust(bottom=0.3)
    ax = fig.add_subplot(111)

    # Define colours
    dacolorz = sns.color_palette("pastel")
    #dacolorz = sns.color_palette()
    color1 = dacolorz[0]
    color2 = dacolorz[1]
    color3 = dacolorz[4]
    color4 = dacolorz[2]
    datacolor = sns.color_palette()[0]

    # Plot data
    #ax.fill_between(wave, 0, fplat, alpha=0.8, color='w', hatch='\\', edgecolor='k', lw=0.01, zorder=-5)
    ax.fill_between(wave, fplat, spline, alpha=0.8, color=dacolorz[1], hatch='', edgecolor='k', lw=0.01, zorder=-5)
    ax.fill_between(wave, spline, flux, color=dacolorz[0], zorder=1)


    ax.errorbar(wave, flux, yerr=fluxerr, label='Data', lw=0.5, zorder=1, ecolor='0.6', capsize=0)


    #ax.plot(wave, fplat, ls='-', lw=0.2, color='w', zorder=10)
    #ax.plot(wave, fplat, ls='--', lw=0.2, color='0.35', zorder=10)
    ax.plot(wave, spline, ls='-', lw=2, color='k')
    ax.plot(wknots, fknots, 'o', ms=5, color='deepskyblue', mew=1, mec='k')


    # NEW #
    # scalefac = 1.3
    scalefac = zody_scale_factor
    # ax.plot(wave, zody_flux, color='r', ls='-')  # label='Zodiacal light (SST tool)'
    ax.plot(wave, zody_flux, color='k', ls='--')  # label='Zodiacal light (SST tool)'
    # ax.plot(wave, zody_flux/scalefac, color='orange', ls='-', zorder=-5)  # label='Zodiacal light (SST tool)'
    #st()

    # print("TURNING OFF LINEIDS")
    lineids(ax, wave, flux)

    # ax.plot(wave, (flux-zody_flux/scalefac)/2., color='0.5', lw=1)

    # ax.plot(wave, (flux-zody_flux), color='0.5', lw=1)


    # Plot parameters
    malpha = 0.7
    linecolor = 'k'
    mylw2 = 1
    #ax.set_ylabel(r'Flux density [$10^{-11} W/m^2/ \mu m$]', fontsize=16)
    #ax.set_ylabel(r'Flux density ($MJy/sr$)', fontsize=14)
    # ax.set_ylabel(r'Surface brightness ($MJy/sr$)', fontsize=14)
    # ax.set_xlabel(r'Wavelength ($\mu m$)', fontsize=14)

    ax.set_ylabel('Surface brightness (MJy/sr)', fontsize=14)
    ax.set_xlabel('Wavelength (µm)', fontsize=14)

    ax.set_axis_bgcolor("w")
    #Create custom artists
    #simArtist = Rectangle((0,1),(0,0), color='k', marker='o', linestyle='')
    blackline = plt.Line2D((0,1),(0,0), color='k', lw=2)
    zod_est = plt.Line2D((0,1),(0,0), color='r', ls='-', lw=2)

    zod_est2 = plt.Line2D((0,1),(0,0), color='k', ls='--', lw=1)

    zod_est_scaled = plt.Line2D((0,1),(0,0), color='orange', ls='-', lw=2)
    sub_line = plt.Line2D((0,1),(0,0), color='0.5', ls='-', lw=2)
    greyline = plt.Line2D((0,1),(0,0), color='0.35', ls='-', lw=1)
    green_patch = mpatches.Patch(color=dacolorz[1], label='Plateaus')
    #plt.legend(handles=[red_patch])
    #Create legend from custom artist/label lists
    #ax.legend([green_patch, blackline, greyline], ['Plateaus', 'Local spline', 'Global spline'], loc=0, frameon=0, bbox_to_anchor=(0.97,0.24), fontsize=12)
    ax.set_ylim(ymin=0)

    if ymax:
        ax.set_ylim(ymax=ymax)

    ax.set_xlim(xmin=5)

    # ax.legend([green_patch, blackline], ['Plateaus', 'Local spline'], loc=0, frameon=0, bbox_to_anchor=(0.97,0.24), fontsize=12)
    zod_str = 'Zodiacal estimate / ' + str(scalefac)
    sub_str = '0.5 * (Data - Zodiacal estimate / ' + str(scalefac) + ')'
    # ax.legend([green_patch, blackline, zod_est, zod_est_scaled, sub_line], ['Plateaus', 'Local spline', 'Zodiacal estimate', zod_str, sub_str], loc=0, frameon=0, fontsize=12)

    sub_str = 'Data - Zodiacal estimate'
    # ax.legend([green_patch, blackline, zod_est, sub_line], ['Plateaus', 'Local spline', 'Zodiacal estimate', sub_str], loc=0, frameon=0, fontsize=12)

    ax.legend([green_patch, blackline, zod_est2], ['Plateaus', 'Local spline', 'Zodiacal estimate'], loc=0, frameon=0, fontsize=12, ncol=3)

    ax.tick_params(which='minor', length=5, width=1)
    ax.tick_params(which='major', length=8, width=1)
    ax.minorticks_on()


    # ================== #
    # Clean up.
    #fig.tight_layout()
    fig.savefig(sname, format='pdf', bbox_inches='tight')

    print()
    print(("SAVED: ", sname))
    print()

    fig.clear()
    plt.close()


    return
def plotResults(savename, fitResults, wave, flux, fluxerr, plotUnits='jy', zoom=0):

    fluxAtomics, fluxPAHs, fluxCont, fluxPlats, csub, csuberr, spline, pahArrRanges, wknots, fknots, finalPlatWave, finalPlatFlux, finalPlatFluxerr, allpahflux, allpahfluxerr, allplatflux, allplatfluxerr = fitResults
    #plotUnits = 'jy'
    #plotUnits = 'si'

    if zoom != 0:
        z = np.where((wave > 5) & (wave < zoom))
        wave = wave[z]
        flux = flux[z]
        fluxerr = fluxerr[z]
        spline = spline[z]
        csub = csub[z]
        csuberr = csuberr[z]
        fknots = fknots[np.where((wknots > 5 ) & (wknots < zoom))]
        wknots = wknots[np.where((wknots > 5 ) & (wknots < zoom))]

        if not isinstance(finalPlatFlux, int):
            finalPlatFlux = finalPlatFlux[z]
            finalPlatWave = finalPlatWave[z]
        #st()
        #ax1.set_xlim(xmax=20)
        #ax1.set_ylim(ymin=-100, ymax=1500)


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
    #fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(20,24))
    #ax0 = axarr[0]
    #ax1 = axarr[1]
    fig = plt.figure(figsize=(16,7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
    gs.update(wspace=0.025, hspace=0.00) # set the spacing between axes.
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)



    # ================== #
    # AX ZERO
    #
    # Plot plateaus.
    if np.nanmax(fplat) > 0:
        ax0.plot(finalPlatWave, fplat, '--', color='red', lw=1.5)
        ax0.fill_between(finalPlatWave, fplat, spline, color='pink', edgecolor='0.5', lw=0)
    #
    # Plot spectrum to evaluate fits.
    #ax0.errorbar(wave, flux, fluxerr, label='flux', color='0.15')
    ax0.errorbar(wave, flux, yerr=fluxerr, label='flux', lw=0.5, zorder=1, ecolor='0.6', capsize=0)
    ax0.plot(wave, flux, label='flux', lw=2, color='0.15')
    ax0.plot(wave, spline, label='spline', color='red', lw=1.5)
    ax0.plot(wknots, fknots, 'o', ms=5, color='deepskyblue', mew=1)
    # print("TURNING OFF LINEIDS")
    lineids(ax0, wave, flux)
    #
    # RMS zones, and tidy up.
    ax0.minorticks_on()
    #ax0.set_xticklabels([])
    #ax0.set_ylabel('Flux density (MJy/sr)', fontsize=12)
    ax0.set_ylabel(r'Surface brightness ($MJy/sr$)', fontsize=12)

    rmsminS, rmsmaxS = rms('SL')
    rmsminL, rmsmaxL = rms('LL')
    ax1.axvspan(rmsminS, rmsmaxS, color='c', alpha=0.3)
    if zoom == 0:
        ax1.axvspan(rmsminL, rmsmaxL, color='c', alpha=0.3)
    elif zoom == 17.5:
        ax0.set_ylim(ymin=-50,ymax=10000)
        ax1.set_ylim(ymin=-50,ymax=10000)
    #else:
        #ax0.set_ylim(ymin=-50,ymax=10000)
        #ax1.set_ylim(ymin=-50,ymax=10000)
        #ax1.set_ylim(ymin=0)




    # ================== #
    # AX ONE
    #
    # Plot crap.
    #ax1.errorbar(wave, csub, csuberr, label='fluxcsub', lw=1, color='0.2')
    ax1.errorbar(wave, csub, yerr=csuberr, label='fluxcsub', lw=0.5, zorder=1, ecolor='0.6', capsize=0)
    ax1.plot(wave, csub, lw=2, color='0.2')

    #
    # Plot features.

    if fluxPAHs.shape[0] != len(pahArrRanges):
        print("SOMETHING WENT WRONG.")
        raise SystemExit()

    for i in range(len(fluxPAHs)):
        #pahArr.append((j, pahWave, pahFlux, pahFluxErr, pahFlux/pahFluxErr))
        inpah = fluxPAHs[i]
        featWave = pahArrRanges[i][0]
        featFlux = pahArrRanges[i][1]
        # st()
        if len(featWave) > 0:
            if plotUnits == 'jy':
                ax1.fill_between(featWave, 0, jy(featFlux, featWave), color='lightgreen')
            elif plotUnits == 'si':
                ax1.fill_between(featWave, 0, featFlux, color='lightgreen')
        pahsnr = inpah[4]
        pahwave = inpah[1]
        #doAnno(pahsnr, pahwave, ax, plotUnits, ftype='pah')

    for i in range(len(fluxAtomics)):
        #lineArr.append((j, atomWave, lineFlux, lineFluxerr, lineSNR, amp, position, sigma))
        inatom = fluxAtomics[i]
        if plotUnits == 'jy':
            realOG = jy(onedgaussian(wave,0,inatom[5],inatom[6],inatom[7]), wave)
        elif plotUnits == 'si':
            realOG = onedgaussian(wave,0,inatom[5],inatom[6],inatom[7])
        ax1.fill_between(wave, 0, realOG, color='salmon')
        linesnr = inatom[4]
        linewave = inatom[1]
        #doAnno(linesnr, linewave, ax, plotUnits, ftype='atomic')

    #ax1.axhline(y=0, color='k', ls='-')
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_xlabel('Wavelength (microns)', fontsize=12)
    yticks = ax1.yaxis.get_major_ticks()
    #yticks[0].label1.set_visible(False)
    yticks[-1].label1.set_visible(False)



    # ================== #
    # Clean up.
    #fig.tight_layout()
    print(("savename: ", savename))
    fig.savefig(savename, format='pdf', bbox_inches='tight')
    fig.clear()
    plt.close()

    return
def saveResults(fluxdir, fitResults, wave, flux, fluxerr, source, uniqID, module='', savePlats=1):

    fluxAtomics, fluxPAHs, fluxCont, fluxPlats, csub, csuberr, spline, pahArrRanges, wknots, fknots, finalPlatWave, finalPlatFlux, finalPlatFluxerr, allpahflux, allpahfluxerr, allplatflux, allplatfluxerr = fitResults

    if module != '':
        fmod = '_' + module
    else:
        fmod = ''

    np.savetxt(fluxdir + source + '_' + uniqID + fmod + '_pahs.txt', fluxPAHs, delimiter=',', header='j, pahWave, flux, fluxerr, SNR')
    np.savetxt(fluxdir + source + '_' + uniqID + fmod + '_atoms.txt', fluxAtomics, delimiter=',', header='j, atomWave, flux, fluxerr, SNR, amp, position, sigma')
    np.savetxt(fluxdir + source + '_' + uniqID + fmod + '_conts.txt', fluxCont, delimiter=',', header='wave, flux, fluxerr (for 2 continuum measurements).')
    np.savetxt(fluxdir + source + '_' + uniqID + fmod + '_total_pahflux.txt', np.array([allpahflux,allpahfluxerr]).T, delimiter=',', header='total pah flux (W/m^2), total pah flux err (W/m^2)')
    np.savetxt(fluxdir + source + '_' + uniqID + fmod + '_total_platflux.txt', np.array([allplatflux,allplatfluxerr]).T, delimiter=',', header='total plat flux (W/m^2), total pah flux err (W/m^2)')

    if savePlats == 1:
        np.savetxt(fluxdir + source + '_' + uniqID + fmod + '_plats.txt', fluxPlats, delimiter=',', header='wave, flux, fluxerr, snr for (5, 10, 15 um) plateaus.')

    return 0


def saveSpectra(specdir, fitResults, wave, flux, fluxerr, source, uniqID, module=''):

    fluxAtomics, fluxPAHs, fluxCont, fluxPlats, csub, csuberr, spline, pahArrRanges, wknots, fknots, finalPlatWave, finalPlatFlux, finalPlatFluxerr, allpahflux, allpahfluxerr, allplatflux, allplatfluxerr = fitResults


    savearr = np.array([wave, flux, fluxerr, spline, csub, csuberr]).T
    np.savetxt(specdir + source + '.txt', savearr, delimiter=',', header='wave, flux, fluxerr, spline, csub, csuberr')

    return 0



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
        yfit = multigaussfit(w, f, err=e, ngauss=2,
                     params=[10.989,fwhm_to_sigma(0.1538),max(f[np.where(w < 11.1)]), 11.258,fwhm_to_sigma(0.236),max(f)],
                     limitedmin=[True,True,True,True,True,True],
                     limitedmax=[True,True,False,True,True,False],
                     minpars=[10.97,fwhm_to_sigma(0.12),max(f[np.where(w < 11.1)])/20.,     11.2,fwhm_to_sigma(0.226),0.],
                     maxpars=[11.05,fwhm_to_sigma(0.19),0.,     11.3,fwhm_to_sigma(0.246),0.],
                     quiet=True,
                     shh=True)
    else:
        my_ind = np.where((wave_feat > 10.6) & (wave_feat < 11.3))
        w = wave_feat[my_ind]
        f = flux_feat[my_ind]
        e = fluxerr_feat[my_ind]
        yfit = multigaussfit(w, f, err=e, ngauss=2,
                 params=[10.989,fwhm_to_sigma(0.1538),max(f[np.where(w < 11.1)]), 11.258,fwhm_to_sigma(0.236),max(f)],
                 limitedmin=[True,True,True,True,True,True],
                 limitedmax=[True,True,False,True,True,False],
                 minpars=[10.97,fwhm_to_sigma(0.02),max(f[np.where(w < 11.1)])/20.,     11.2,fwhm_to_sigma(0.16),0.],
                 maxpars=[11.05,fwhm_to_sigma(0.19),0.,     11.3,fwhm_to_sigma(0.246),0.],
                 quiet=True,
                 shh=True)

    y1r = onedgaussian(w,0,yfit[0][2],yfit[0][0],yfit[0][1])
    y2r = onedgaussian(w,0,yfit[0][5],yfit[0][3],yfit[0][4])

    # 11.0
    small_gauss_area = sp.integrate.trapz(y1r, x=w)
    position = yfit[0][0]
    sigma = yfit[0][1]
    amp = yfit[0][2]
    position_err = yfit[2][0]
    sigma_err = yfit[2][1]
    amp_err = yfit[2][2]
    small_gauss_area_err = np.sqrt((amp_err/amp)**2 + (sigma_err/sigma)**2) * small_gauss_area
    myrange = [position-(3. * sigma), position+(3.*sigma)]
    N = np.where((wave_feat >= myrange[0]) & (wave_feat <= myrange[1]))[0]
    dl = w[1] - w[0]
    measured_flux_noise110 = (rms * np.sqrt(len(N)) * dl * 2)

    # 11.2
    gauss_area = sp.integrate.trapz(y2r, x=w)
    position = yfit[0][3]
    sigma = yfit[0][4]
    amp = yfit[0][5]
    position_err = yfit[2][3]
    sigma_err = yfit[2][4]
    amp_err = yfit[2][5]
    gauss_area_err = np.sqrt((amp_err/amp)**2 + (sigma_err/sigma)**2) * gauss_area
    myrange = [position-(3. * sigma), position+(3.*sigma)]
    N = np.where((wave_feat >= myrange[0]) & (wave_feat <= myrange[1]))[0]
    dl = w[1] - w[0]
    measured_flux_noise112 = (rms * np.sqrt(len(N)) * dl * 2)


    ######################################
    new_flux_feat = f - y1r
    trap_flux_high = sp.integrate.trapz(new_flux_feat+e, x=w)
    trap_flux_low = sp.integrate.trapz(new_flux_feat-e, x=w)
    trap_flux = np.mean([trap_flux_high, trap_flux_low])
    trap_flux_std = 0.67 * np.std([trap_flux_high, trap_flux_low])
    ######################################

    #plt.close()
    #plt.plot(w, f, label='data')
    #plt.plot(w, yfit[1], label='total fit')
    #plt.plot(w, y1r, label='y1r')
    #plt.plot(w, y2r, label='y2r')
    #plt.plot(w, new_flux_feat, label='feat - y1r')
    #plt.legend(loc=0)

    ##print(small_gauss_area, small_gauss_area_err, measured_flux_noise110)
    ##print(gauss_area, gauss_area_err, measured_flux_noise112)
    ##print(trap_flux, trap_flux_std + measured_flux_noise112)

    #plt.axvline(x=w[0])
    #plt.axvline(x=w[-1])
    #plt.show()
    #plt.close()
    ##st()


    FINAL_112_FLUX = trap_flux # full_trap_flux - small_gauss_area
    FINAL_112_FLUX_ERR = measured_flux_noise112 # + trap_flux_std
    FINAL_110_FLUX = small_gauss_area
    FINAL_110_FLUX_ERR = small_gauss_area_err + measured_flux_noise110

    #st()

    return FINAL_110_FLUX, FINAL_110_FLUX_ERR, FINAL_112_FLUX, FINAL_112_FLUX_ERR, myrange

def measureSpectrum(wave, flux, fluxerr, special_cut=0, source=0, cpos=0, skip127=0, skip_8um_bump=0, ames_data='0', myobjname='', suppress=None):

    # # Convert units if needed.
    # if np.nanmean(flux) > 1e-6:
    #     print("I think this spectrum is in MJy/sr, or Jy.")
    #     #st()
    # else:
    #     print(np.nanmean(flux))
    #     st()
    #     print("I think this spectrum is in W/m^2/um. Converting to MJy/sr...")
    #     flux = jy(flux,wave)
    #     fluxerr = jy(fluxerr,wave)
    if not suppress:
        print("Assuming Jy. If it's not (e.g. W/m^2/um), come back to measureFeatures.py!")

    # Determine nominal anchor locations.
    if ames_data == '0':
        anchorWaves, findMin, width, smoothFlag = ctPoints(special_cut=special_cut, skip_8um_bump=skip_8um_bump)
    else:
        anchorWaves, findMin, width, smoothFlag = ctPointsAmes(special_cut=special_cut, skip_8um_bump=skip_8um_bump, ames_data=ames_data, myobjname=myobjname)



    # Determine actual anchor points.
    waveKnotsFinal, fluxKnotsFinal = cf.measure(anchorWaves, wave, flux, findMin, width, smoothFlag, smoothsize=getSmoothSize(), windowsize=getWindowSize())


    # Remove any NaN anchor points!!!!
    badindices = np.where(np.isnan(waveKnotsFinal))[0]
    waveKnotsFinal = np.delete(waveKnotsFinal, badindices)
    fluxKnotsFinal = np.delete(fluxKnotsFinal, badindices)

    # Create spline.
    spl = interp.splrep(waveKnotsFinal, fluxKnotsFinal)
    splineWave = wave
    splineFlux = interp.splev(splineWave, spl)
    csub = flux - splineFlux
    csuberr = fluxerr

    passflux = si(flux, wave)
    csubSI = si(csub, wave)
    csuberrSI = si(csuberr, wave)

    # st()

    # Measure plateau fluxes.
    platflux1, platfluxerr1, platflux2, platfluxerr2, platflux3, platfluxerr3, platflux4, platfluxerr4, platwavearr, platfluxarr, finalPlatWave, finalPlatFlux, finalPlatFluxerr = measurePlateauFlux(wave, flux, fluxerr, splineFlux, waveKnotsFinal, fluxKnotsFinal, csub, csuberr)
    include_1821 = includePlat1821()
    if include_1821:
        holdplatwave = np.array([510, 1015, 1518, 1821])
        holdplatflux = np.array([platflux1, platflux2, platflux3, platflux4])
        holdplatfluxerr = np.array([platfluxerr1, platfluxerr2, platfluxerr3, platfluxerr4])
        fluxPlats = np.vstack((holdplatwave,holdplatflux,holdplatfluxerr,holdplatflux/holdplatfluxerr)).T
    else:
        holdplatwave = np.array([510, 1015, 1518])
        holdplatflux = np.array([platflux1, platflux2, platflux3])
        holdplatfluxerr = np.array([platfluxerr1, platfluxerr2, platfluxerr3])
        fluxPlats = np.vstack((holdplatwave,holdplatflux,holdplatfluxerr,holdplatflux/holdplatfluxerr)).T


    # Measure atomic lines.
    atomicLines, atomicLineNames, atomicPlotNames = linesToFit()
    lineArr = []
    # // ATOMIC LINES
    for j in range(len(atomicLines)):
        atomWave = atomicLines[j]
        atomName = atomicLineNames[j]
        waveMargin = 0.4
        fitMargin = 0.15

        if atomWave > np.amax(wave):
            continue

        if atomWave in [12.81]:
            continue # HANDLE BLENDED LINES SEPARATELY

        if atomWave < 15:
            irms = measure_112_RMS(wave, csubSI)
        elif atomWave >= 15 and atomWave < 20:
            irms = measure_14_RMS(wave, csubSI)
        elif atomWave >= 20:
            irms = measure_25_RMS(wave, csubSI)

        dobreak = 0
        # print((j,atomWave,atomName))
        # st()
        if atomWave <= wave[0]:
            lineArr.append((j, atomWave, 0, 0, 0, 0, 0, 0))
        else:
            lineFlux, lineFluxerr, lineSNR, amp, position, sigma = fitLine(atomWave, waveMargin, wave, csubSI, csuberrSI, fitMargin, irms, dobreak=dobreak)
            lineArr.append((j, atomWave, lineFlux, lineFluxerr, lineSNR, amp, position, sigma))



    # Measure PAH features.
    pahLines, pahLineNames, pahPlotNames = pahsToFit()
    pahArr = []
    pahArrRanges = []
    # // PAH FEATURES
    for j in range(len(pahLines)):
        pahWave = pahLines[j]
        pahName = pahLineNames[j]

        if pahWave > np.amax(wave):
            continue

        if pahWave in [12.7, 15.8]:
            continue # BLENDED, DO SEPARATELY

        if pahWave > 15:
            #irms = measure_25_RMS(wave, csubSI)
            irms = measure_14_RMS(wave, csubSI, passflux=passflux)
        else:
            irms = measure_112_RMS(wave, csubSI)

        # TRY SPLITTING 11.0, 11.2.
        if pahWave == 11.2:
            flux110, flux110err, flux112, flux112err, somerange = fit_110(wave, csubSI, csuberrSI, irms)
            daRange = np.where((wave >= somerange[0]) & (wave <= somerange[-1]))
            wRange = wave[daRange]
            cRange = csubSI[daRange]
            pahArr.append((999, 11.0, flux110, flux110err, flux110/flux110err))
            pahArrRanges.append((wRange, cRange))

            daRange = np.where((wave >= somerange[-1]) & (wave <= 11.6))
            wRange = wave[daRange]
            cRange = csubSI[daRange]
            pahArr.append((j, 11.2, flux112, flux112err, flux112/flux112err))
            pahArrRanges.append((wRange, cRange))


        else:
            ## Figure out integration boundaries.
            intmin, intmax = pahIntRanges(pahWave)
            intRange = np.where((wave >= intmin) & (wave <= intmax))
            wRange = wave[intRange]
            cRange = csubSI[intRange]

            doTrust = testTrust(source, cpos, pahWave)

            if doTrust and len(wRange) > 1:
                # print((pahWave, pahName))
                # st()
                pahFlux = sp.integrate.simps(cRange, wRange)
                if np.isnan(pahFlux):
                    pahFlux = sp.integrate.trapz(cRange, wRange)
                # st()
                pahFluxErr = compute_feature_uncertainty(cRange*0, cRange*0, wRange, irms, manual_range=[intmin,intmax], dopause=0)
                pahArr.append((j, pahWave, pahFlux, pahFluxErr, pahFlux/pahFluxErr))
                pahArrRanges.append((wRange, cRange))
            else:
                pahArr.append((j, pahWave, 0, 0, 0))
                pahArrRanges.append((wRange, cRange))

        #if pahWave == 6.2:
            #st()

    # Fit 12.7/12.8
    # if skip127 == 1:
    #     lineArr.append((99, 12.8, 0, 0, 0, 0, 0, 0))
    #     pahArr.append((99, 12.7, 0, 0, 0))
    #     pahArrRanges.append((wavecut127, speccut127))
    # else:
    # flux127, fluxerr127, flux128, fluxerr128, amp128, position128, sigma128, wavecut127, speccut127 = fit127(wave, csubSI, csuberrSI)
    flux127 = 0
    flux128 = 0
    # CLUDGE
    if flux127 == 0 and flux128 == 0:
        lineArr.append((99, 12.8, 0, 0, 0, 0, 0, 0))
        pahArr.append((99, 12.7, 0, 0, 0))
        pahArrRanges.append(((), ()))

    else:
        doTrust = testTrust(source, cpos, 12.7)
        if doTrust:
            lineArr.append((99, 12.8, flux128, fluxerr128, flux128/fluxerr128, amp128, position128, sigma128))
            pahArr.append((99, 12.7, flux127, fluxerr127, flux127/fluxerr127))
            pahArrRanges.append((wavecut127, speccut127))
        else:
            lineArr.append((99, 12.8, 0, 0, 0, 0, 0, 0))
            pahArr.append((99, 12.7, 0, 0, 0))
            pahArrRanges.append((wavecut127, speccut127))



    # Measure continuum strength (at 14 microns?).
    wc1 = 10
    wc2 = 15.2
    fc1 = flux[find_nearest.find_nearest(wave,wc1)]
    fc2 = flux[find_nearest.find_nearest(wave,wc2)]
    fc1err = fluxerr[find_nearest.find_nearest(wave,wc1)]
    fc2err = fluxerr[find_nearest.find_nearest(wave,wc2)]

    # Convert to SI.
    fc1 = si(fc1, wc1)
    fc2 = si(fc2, wc2)
    fc1err = si(fc1err, wc1)
    fc2err = si(fc2err, wc2)

    # FINAL FLUXES TO RETURN.
    fluxAtomics = np.array(lineArr)
    fluxPAHs = np.array(pahArr)
    fluxCont = np.array([[wc1,wc2], [fc1,fc2], [fc1err, fc2err]]).T


    allpahflux = np.nansum(fluxPAHs.T[2])
    allpahfluxerr = np.nansum(fluxPAHs.T[3])

    allplatflux = np.nansum(fluxPlats.T[1])
    allplatfluxerr = np.nansum(fluxPlats.T[2])

    ## Plot stuff if desired.
    #plt.plot(wave, flux, 'b', lw=2)
    #plt.plot(splineWave, splineFlux, 'g', lw=2)
    #plt.plot(waveKnotsFinal, fluxKnotsFinal, 'ro', ms=5)
    #plt.plot(finalPlatWave, finalPlatFlux, '0.5', label='Plateau spec.', lw=2)
    #plt.plot(platwavearr[0], platfluxarr[0], label='platsub1', lw=4, color='0.5')
    #plt.plot(platwavearr[1], platfluxarr[1], label='platsub2', lw=4, color='k')
    #plt.plot(platwavearr[2], platfluxarr[2], label='platsub3', lw=4, color='0.5')
    #plt.plot(wc1, fc1, color='w', mew=4, marker='D', ms=8)
    #plt.plot(wc2, fc2, color='w', mew=4, marker='D', ms=8)
    #plt.axhline(y=0, color='0.3', ls='-')
    #plt.legend(loc=0)
    #plt.show()
    #plt.close()

    return fluxAtomics, fluxPAHs, fluxCont, fluxPlats, csub, csuberr, splineFlux, pahArrRanges, waveKnotsFinal, fluxKnotsFinal, finalPlatWave, finalPlatFlux, finalPlatFluxerr, allpahflux, allpahfluxerr, allplatflux, allplatfluxerr
def measureSpectrumOneModule(wave, flux, fluxerr, module):

    # Convert units if needed.
    if np.nanmean(flux) > 1e-6:
        print("I think this spectrum is in MJy/sr, or Jy.")
        #st()
    else:
        print((np.nanmean(flux)))
        st()
        print("I think this spectrum is in W/m^2/um. Converting to MJy/sr...")
        flux = jy(flux,wave)
        fluxerr = jy(fluxerr,wave)

    #if module == 'SL':
        #if np.amax(wave) > 20:
            #print "uh oh"
            #st()
    #if module == 'LL':
        #if np.amax(wave) < 20:
            #print "uh oh"
            #st()

    # Determine nominal anchor locations.
    anchorWaves, findMin, width, smoothFlag = ctPointsSolo(module)

    # Determine actual anchor points.
    waveKnotsFinal, fluxKnotsFinal = cf.measure(anchorWaves, wave, flux, findMin, width, smoothFlag, smoothsize=getSmoothSize(), windowsize=getWindowSize())

    # Create spline.
    spl = interp.splrep(waveKnotsFinal, fluxKnotsFinal)
    splineWave = wave
    splineFlux = interp.splev(splineWave, spl)
    csub = flux - splineFlux
    csuberr = fluxerr

    # Convert to SI, measure RMS.
    csubSI = si(csub, wave)
    csuberrSI = si(csuberr, wave)
    if module == 'SL':
        irms = measure_112_RMS(wave, csubSI)
    elif module == 'LL':
        irms = measure_25_RMS(wave, csubSI)



    ## Plot stuff if desired.
    #plt.plot(wave, flux, 'b', lw=2)
    #plt.plot(splineWave, splineFlux, 'g', lw=2)
    #plt.plot(waveKnotsFinal, fluxKnotsFinal, 'ro', ms=5)
    ##plt.plot(finalPlatWave, finalPlatFlux, '0.5', label='Plateau spec.', lw=2)
    ##plt.plot(platwavearr[0], platfluxarr[0], label='platsub1', lw=4, color='0.5')
    ##plt.plot(platwavearr[1], platfluxarr[1], label='platsub2', lw=4, color='k')
    ##plt.plot(platwavearr[2], platfluxarr[2], label='platsub3', lw=4, color='0.5')
    ##plt.plot(wc1, fc1, color='w', mew=4, marker='D', ms=8)
    ##plt.plot(wc2, fc2, color='w', mew=4, marker='D', ms=8)
    #plt.axhline(y=0, color='0.3', ls='-')
    #plt.legend(loc=0)
    #plt.show()
    #plt.close()


    # Measure atomic lines.
    atomicLines, atomicLineNames, atomicPlotNames = linesToFit()
    lineArr = []

    # // ATOMIC LINES
    for j in range(len(atomicLines)):

        atomWave = atomicLines[j]
        atomName = atomicLineNames[j]
        waveMargin = 1
        fitMargin = 0.15
        if wave[0] <= atomWave <= wave[-1]:
            if atomWave in [12.81]:
                continue # HANDLE BLENDED LINES SEPARATELY

            dobreak = 0
            lineFlux, lineFluxerr, lineSNR, amp, position, sigma = fitLine(atomWave, waveMargin, wave, csubSI, csuberrSI, fitMargin, irms, dobreak=dobreak)
            lineArr.append((j, atomWave, lineFlux, lineFluxerr, lineSNR, amp, position, sigma))


    # Measure PAH features.
    pahLines, pahLineNames, pahPlotNames = pahsToFit()
    pahArr = []
    pahArrRanges = []

    # // PAH FEATURES
    for j in range(len(pahLines)):

        pahWave = pahLines[j]
        pahName = pahLineNames[j]
        if wave[0] <= pahWave <= wave[-1]:
            if pahWave in [12.7, 15.8]:
                continue # BLENDED, DO SEPARATELY

            ## Figure out integration boundaries.
            intmin, intmax = pahIntRanges(pahWave)
            intRange = np.where((wave >= intmin) & (wave <= intmax))
            wRange = wave[intRange]
            cRange = csubSI[intRange]

            if len(wRange) > 1:
                pahFlux = sp.integrate.simps(cRange, wRange)
                pahFluxErr = compute_feature_uncertainty(cRange*0, cRange*0, wRange, irms, manual_range=[intmin,intmax])
                pahArr.append((j, pahWave, pahFlux, pahFluxErr, pahFlux/pahFluxErr))
                pahArrRanges.append((wRange, cRange))
            else:
                pahArr.append((j, pahWave, 0, 1, 0))
                pahArrRanges.append((wRange, cRange))


    if module == 'SL' and wave[-1] >= 13:
        # Fit 12.7/12.8
        flux127, fluxerr127, flux128, fluxerr128, amp128, position128, sigma128, wavecut127, speccut127 = fit127(wave, csubSI, csuberrSI)
        lineArr.append((99, 12.8, flux128, fluxerr128, flux128/fluxerr128, amp128, position128, sigma128))
        pahArr.append((99, 12.7, flux127, fluxerr127, flux127/fluxerr127))
        pahArrRanges.append((wavecut127, speccut127))

    if module == 'SL':
        wc = 10
        fc = flux[find_nearest.find_nearest(wave,wc)]
        fcerr = fluxerr[find_nearest.find_nearest(wave,wc)]
    elif module == 'LL':
        wc = 15.2
        fc = flux[find_nearest.find_nearest(wave,wc)]
        fcerr = fluxerr[find_nearest.find_nearest(wave,wc)]

    # Convert to SI.
    fc = si(fc, wc)
    fcerr = si(fcerr, wc)

    # FINAL FLUXES TO RETURN.
    fluxAtomics = np.array(lineArr)
    fluxPAHs = np.array(pahArr)
    fluxCont = np.array([[wc], [fc], [fcerr]]).T

    allpahflux = np.nansum(fluxPAHs.T[2])
    allpahfluxerr = np.nansum(fluxPAHs.T[3])

    allplatflux = 0
    allplatfluxerr = 0

    return fluxAtomics, fluxPAHs, fluxCont, 0, csub, csuberr, splineFlux, pahArrRanges, waveKnotsFinal, fluxKnotsFinal, 0, 0, 0, allpahflux, allpahfluxerr, allplatflux, allplatfluxerr

##############################


if __name__ == "__main__":

    source = 'c32'
    cpos = 1
    uniqID = 'p' + str(cpos).zfill(2)
    meanSpecDir = '/home/koma/Dropbox/code/Python/galacticCentre/analysis/meanSpectra/'
    inS = meanSpecDir + 'spectra/' + source + '/' + source + '_SLmeanspec_' + uniqID + '.txt'
    inL = meanSpecDir + 'spectra/' + source + '/' + source + '_LLmeanspec_' + uniqID + '.txt'

    # Read in SL and LL spectra.
    fluxL, fluxerrL, csubL, csubLerr, waveL = np.loadtxt(inL, delimiter=',').T
    fluxS, fluxerrS, csubS, csubSerr, waveS = np.loadtxt(inS, delimiter=',').T


    #/////////////////////////////////
    # TEST FULL SPECTRUM.
    #/////////////////////////////////

    # Stich 'em.
    wave, flux, fluxerr, _, _ = stitch.stitch_SL_LL(waveS, waveL, fluxS, fluxL, fluxerrS, fluxerrL)

    # Measure features.
    fitResults = measureSpectrum(wave, flux, fluxerr)

    # Plot results.
    savename = meanSpecDir + 'testdir/' + source + '/' + 'test.pdf'
    plotResults(savename, fitResults, wave, flux, fluxerr, plotUnits='jy')

    # Save fluxes.
    fluxdir = meanSpecDir + 'testdir/' + source + '/'
    saveResults(fluxdir, fitResults, wave, flux, fluxerr, source, uniqID)


    #/////////////////////////////////
    # TEST HALF SPECTRUM - SL
    #/////////////////////////////////

    savename = meanSpecDir + 'testdir/' + source + '/' + 'testSL.pdf'
    fluxdir = meanSpecDir + 'testdir/' + source + '/'
    fitResults = measureSpectrumOneModule(waveS, fluxS, fluxerrS, "SL")
    plotResultsSoloNew(savename, fitResults, waveS, fluxS, fluxerrS, plotUnits='jy')
    saveResults(fluxdir, fitResults, waveS, fluxS, fluxerrS, source, uniqID, module='SL', savePlats=0)


    #/////////////////////////////////
    # TEST HALF SPECTRUM - LL
    #/////////////////////////////////

    savename = meanSpecDir + 'testdir/' + source + '/' + 'testLL.pdf'
    fluxdir = meanSpecDir + 'testdir/' + source + '/'
    fitResults = measureSpectrumOneModule(waveL, fluxL, fluxerrL, "LL")
    plotResultsSoloNew(savename, fitResults, waveL, fluxL, fluxerrL, plotUnits='jy')
    saveResults(fluxdir, fitResults, waveL, fluxL, fluxerrL, source, uniqID, module='LL', savePlats=0)







