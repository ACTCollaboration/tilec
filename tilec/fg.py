from __future__ import print_function
import numpy as np
import yaml
import os
"""
Utilities for unit conversions and foreground SED modeling, including Planck and ACT bandpasses.
SEDs included: CMB, kSZ, tSZ, rSZ, mu, CIB
"""
######################################
# global constants
# MKS units, except electron rest mass-energy
######################################
TCMB = 2.726 #Kelvin
TCMB_uK = 2.726e6 #micro-Kelvin
hplanck=6.626068e-34 #MKS
kboltz=1.3806503e-23 #MKS
clight=299792458.0 #MKS
m_elec = 510.999 #keV
######################################

######################################
# various unit conversions
######################################
# conversion from specific intensity to Delta T units (i.e., 1/dBdT|T_CMB)
#   i.e., from W/m^2/Hz/sr (Jy/sr) --> uK_CMB
#   i.e., you would multiply a map in Jy/sr by this factor to get an output map in uK_CMB
def ItoDeltaT(nu_ghz):
    nu = 1.e9*np.asarray(nu_ghz)
    X  = hplanck*nu/(kboltz*TCMB)
    return (1.0 / (2.0 * X**4.0 * np.exp(X) * (kboltz**3.0*TCMB**2.0) / (hplanck * clight)**2.0 / (np.exp(X) - 1.0)**2.0)) * 1.e6 #the 1e6 takes you from K to uK

# conversion from antenna temperature to CMB thermodynamic temperature
def antenna2thermoTemp(nu_ghz):
    nu = 1.e9*np.asarray(nu_ghz)
    x = hplanck*nu/(kboltz*TCMB)
    return (2.0*np.sinh(x/2.0)/x)**2.0 #from http://adsabs.harvard.edu/abs/2000ApJ...530..133T Eq. 1

# function needed for Planck bandpass integration/conversion following approach in Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf
# blackbody derivative
# units are Jy/sr/uK_CMB
def dBnudT(nu_ghz):
    nu = 1.e9*np.asarray(nu_ghz)
    X = hplanck*nu/(kboltz*TCMB)
    return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB_uK
######################################

######################################
# dictionary of parameter values needed for some component SEDs
######################################
def read_param_dict_from_yaml(yaml_file):
    assert(yaml_file != None)
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    return config
# default case
fpath = os.path.dirname(__file__)
default_dict = read_param_dict_from_yaml(fpath+'/../input/fg_SEDs_default_params.yml')
######################################


######################################
# spectral functions of physical components, evaluated for specific frequencies (i.e., delta-function bandpasses)
# N.B. overall amplitudes are (generally) not meaningful; this function gives relative conversions between frequencies, for each component SED
# convention is that the maps being modeled are in uK_CMB units
# nu_ghz can contain entries that are None, which correspond to maps that have no CMB-relevant (or CIB) signals in them (e.g., HI maps)
######################################
def get_mix(nu_ghz, comp, param_dict_file=None): #nu_ghz = array of frequencies in GHz; comp = string containing component name; param_dict_file = dictionary of SED parameters and values (optional, and only needed for some SEDs)
    assert (comp != None)
    nu_ghz = np.atleast_1d(nu_ghz) #catch possible scalar input
    assert (len(nu_ghz) > 0)
    if (comp == 'CMB' or comp == 'kSZ'): #CMB (or kSZ)
        resp = np.ones(len(np.asarray(nu_ghz))) #this is unity by definition, since we're working in Delta T units [uK_CMB]; output ILC map will thus also be in uK_CMB
        resp[np.where(nu_ghz == None)] = 0. #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in nu_ghz
        return resp
    elif (comp == 'tSZ'): #Thermal SZ (y-type distortion)
        nu = 1.e9*np.asarray(nu_ghz).astype(float)
        X = hplanck*nu/(kboltz*TCMB)
        resp = (X / np.tanh(X/2.0) - 4.0) * TCMB_uK #put explicitly into uK_CMB units, so that output ILC map is in Compton-y
        resp[np.where(nu_ghz == None)] = 0. #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in nu_ghz
        return resp
    elif (comp == 'mu'): #mu-type distortion
        nu = 1.e9*np.asarray(nu_ghz).astype(float)
        X = hplanck*nu/(kboltz*TCMB)
        resp = (X / 2.1923 - 1.0)/X * TCMB_uK #put explicitly into uK_CMB units, so that output ILC map is in terms of \mu (analogous to y above)
        resp[np.where(nu_ghz == None)] = 0. #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in nu_ghz
        return resp
    elif (comp == 'rSZ'): #relativistic thermal SZ (to 3rd order in kT_e/(m_e*c^2) using expressions from Nozawa+2006)
        # relativistic SZ parameter choice in dict file: kT_e_keV [keV] (temperature of electrons)
        if param_dict_file is None:
            p = default_dict
        else:
            p = read_param_dict_from_yaml(param_dict_file)
        nu = 1.e9*np.asarray(nu_ghz).astype(float)
        kTe = p['kT_e_keV'] / m_elec #kT_e/(m_e*c^2)
        X = hplanck*nu/(kboltz*TCMB)
        Xtwid=X*np.cosh(0.5*X)/np.sinh(0.5*X)
        Stwid=X/np.sinh(0.5*X)
        #Y0=Xtwid-4.0 #non-relativistic tSZ (same as 'tSZ' above)
        Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
        Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+0.3666667*Xtwid**5.0+Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+4.7666667*Xtwid**3.0)+Stwid**4.0*(-8.8+3.11666667*Xtwid)
        Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0-531.257142857*Xtwid**4.0+86.1357142857*Xtwid**5.0-6.09523809524*Xtwid**6.0+0.15238095238*Xtwid**7.0+Stwid**2.0*(-709.8+2850.6*Xtwid-2921.91428571*Xtwid**2.0+1119.76428571*Xtwid**3.0-173.714285714*Xtwid**4.0+9.14285714286*Xtwid**5.0)+Stwid**4.0*(-531.257142857+732.153571429*Xtwid-274.285714286*Xtwid**2.0+29.2571428571*Xtwid**3.0)+Stwid**6.0*(-25.9047619048+9.44761904762*Xtwid)
        # leave out non-rel. tSZ, as we only want the rel. terms here
        resp = (Y1*kTe+Y2*kTe**2.+Y3*kTe**3.) * TCMB_uK #put explicitly into uK_CMB units, analogous to non-rel. tSZ above
        resp[np.where(nu_ghz == None)] = 0. #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in nu_ghz
        return resp
    elif (comp == 'CIB'):
        # CIB SED parameter choices in dict file: Tdust_CIB [K], beta_CIB, nu0_CIB [GHz]
        # N.B. overall amplitude is not meaningful here; output ILC map (if you tried to preserve this component) would not be in sensible units
        if param_dict_file is None:
            p = default_dict
        else:
            p = read_param_dict_from_yaml(param_dict_file)
        nu = 1.e9*np.asarray(nu_ghz).astype(float)
        X_CIB = hplanck*nu/(kboltz*(p['Tdust_CIB']))
        nu0_CIB = p['nu0_CIB_ghz']*1.e9
        X0_CIB = hplanck*nu0_CIB/(kboltz*(p['Tdust_CIB']))
        resp = (nu/nu0_CIB)**(3.0+(p['beta_CIB'])) * ((np.exp(X0_CIB) - 1.0) / (np.exp(X_CIB) - 1.0)) * (ItoDeltaT(np.asarray(nu_ghz).astype(float))/ItoDeltaT(p['nu0_CIB_ghz']))
        resp[np.where(nu_ghz == None)] = 0. #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in nu_ghz
        return resp
    elif (comp == 'CIB_Jysr'): #same as CIB above but in Jy/sr instead of uK_CMB
        # CIB SED parameter choices in dict file: Tdust_CIB [K], beta_CIB, nu0_CIB [GHz]
        # N.B. overall amplitude is not meaningful here; output ILC map (if you tried to preserve this component) would not be in sensible units
        if param_dict_file is None:
            p = default_dict
        else:
            p = read_param_dict_from_yaml(param_dict_file)
        nu = 1.e9*np.asarray(nu_ghz).astype(float)
        X_CIB = hplanck*nu/(kboltz*(p['Tdust_CIB']))
        nu0_CIB = p['nu0_CIB_ghz']*1.e9
        X0_CIB = hplanck*nu0_CIB/(kboltz*(p['Tdust_CIB']))
        resp = (nu/nu0_CIB)**(3.0+(p['beta_CIB'])) * ((np.exp(X0_CIB) - 1.0) / (np.exp(X_CIB) - 1.0))
        resp[np.where(nu_ghz == None)] = 0. #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in nu_ghz
        return resp
    else:
        print("unknown component specified")
        raise NotImplementedError
######################################


######################################
# spectral functions of physical components, evaluated for non-trivial bandpasses
# N.B. overall amplitudes are (generally) not meaningful; this function gives relative conversions between frequencies, for each component SED
# convention is that the maps being modeled are in uK_CMB units
# bandpass file columns should be [freq (GHz)] [transmission]  (any other columns are ignored)
# bp_list can contain entries that are None, which correspond to maps that have no CMB-relevant (or CIB) signals in them (e.g., HI maps)
######################################
def get_mix_bandpassed(bp_list, comp, param_dict_file=None):
    #bp_list = list containing strings of bandpass filenames; comp = string containing component name; param_dict_file = dictionary of SED parameters and values (optional, and only needed for some SEDs)
    assert (comp != None)
    assert (bp_list != None)
    N_freqs = len(bp_list)
    output = np.zeros(N_freqs)
    if (comp == 'CMB' or comp == 'kSZ'): #CMB (or kSZ)
        output = np.ones(N_freqs) #this is unity by definition, since we're working in Delta T units [uK_CMB]; output ILC map will thus also be in uK_CMB
        for i in range(N_freqs):
            if(bp_list[i] == None): #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in bp_list
                output[i] = 0.
        return output
    elif (comp == 'tSZ' or comp == 'mu' or comp == 'rSZ'): #Thermal SZ (y-type distortion) or mu-type distortion or relativistic tSZ
        # following Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf -- N.B. IMPORTANT TYPO IN THEIR EQ. 35 -- see https://www.aanda.org/articles/aa/pdf/2014/11/aa21531-13.pdf
        for i,bp in enumerate(bp_list):
            if (bp_list[i] != None):
                nu_ghz, trans = np.loadtxt(bp, usecols=(0,1), unpack=True)
                output[i] = np.trapz(trans * dBnudT(nu_ghz) * get_mix(nu_ghz, comp), nu_ghz) / np.trapz(trans * dBnudT(nu_ghz), nu_ghz)
            elif (bp_list[i] == None): #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in bp_list
                output[i] = 0.
        return output #this is the response at each frequency channel in uK_CMB for a signal with y=1 (or mu=1)
    elif (comp == 'CIB'):
        # following Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf -- N.B. IMPORTANT TYPO IN THEIR EQ. 35 -- see https://www.aanda.org/articles/aa/pdf/2014/11/aa21531-13.pdf
        # CIB SED parameter choices in dict file: Tdust_CIB [K], beta_CIB, nu0_CIB [GHz]
        # N.B. overall amplitude is not meaningful here; output ILC map (if you tried to preserve this component) would not be in sensible units
        if param_dict_file is None:
            p = default_dict
        else:
            p = read_param_dict_from_yaml(param_dict_file)
        for i,bp in enumerate(bp_list):
            if (bp_list[i] != None):
                nu_ghz, trans = np.loadtxt(bp, usecols=(0,1), unpack=True)
                # N.B. this expression follows from Eqs. 32 and 35 of https://www.aanda.org/articles/aa/pdf/2014/11/aa21531-13.pdf , and then noting that one also needs to first rescale the CIB emission in Jy/sr from nu0_CIB to the "nominal frequency" nu_c that appears in those equations (i.e., multiply by get_mix(nu_c, 'CIB_Jysr')).  The resulting cancellation leaves this simple expression which has no dependence on nu_c.
                output[i] = (np.trapz(trans * get_mix(nu_ghz, 'CIB_Jysr', param_dict_file), nu_ghz) / np.trapz(trans * dBnudT(nu_ghz), nu_ghz))
            elif (bp_list[i] == None): #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in bp_list
                output[i] = 0.
        return output/np.amax(output) #overall amplitude not meaningful, so divide by max to get numbers of order unity; output gives the relative conversion between CIB at different frequencies, for maps in uK_CMB
    else:
        print("unknown component specified")
        raise NotImplementedError
######################################


