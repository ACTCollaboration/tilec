from __future__ import print_function
import numpy as np
import yaml
import os

######################################
# global constants
# MKS units, except electron rest mass
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
def read_param_dict_from_yaml(yaml_file="input/fg_SEDs_default_params.yml"):
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    return config
# default case
fpath = os.path.dirname(__file__)
default_dict = read_param_dict_from_yaml(fpath+'/../input/fg_SEDs_default_params.yml')
#pdict = {}                                                                                                                                           
#pdict['beta'] = 1                                                                                                                                    
######################################


######################################
# spectral functions of physical components, evaluated for specific frequencies (i.e., delta-function bandpasses)
# N.B. overall amplitudes are (generally) not meaningful; this function gives relative conversions between frequencies, for each component SED
# convention is that the maps being modeled are in uK_CMB units
######################################
def get_mix(nu_ghz, comp, param_dict_file=None): #nu_ghz = array of frequencies in GHz; comp = string containing component name; param_dict_file = dictionary of SED parameters and values (optional, and only needed for non-first-principles SEDs)
    assert (comp != None)
    if (comp == 'CMB' or comp == 'kSZ'): #CMB (or kSZ)
        return np.ones(len(np.asarray(nu_ghz))) #this is unity by definition, since we're working in Delta T units [uK_CMB]; output ILC map will thus also be in uK_CMB
    elif (comp == 'tSZ'): #Thermal SZ (y-type distortion)
        nu = 1.e9*np.asarray(nu_ghz)
        X = hplanck*nu/(kboltz*TCMB)
        return (X / np.tanh(X/2.0) - 4.0) * TCMB_uK #put explicitly into uK_CMB units, so that output ILC map is in Compton-y
    elif (comp == 'mu'): #mu-type distortion
        nu = 1.e9*np.asarray(nu_ghz)
        X = hplanck*nu/(kboltz*TCMB)
        return (X / 2.1923 - 1.0)/X * TCMB_uK #put explicitly into uK_CMB units, so that output ILC map is in terms of \mu (analogous to y above)
#    elif (comp == 'rSZ'):
#        # relativistic SZ parameter choice in dict file: kT_e_keV [keV] (temperature of electrons)
#        if param_dict is None: param_dict = default_dict
#        p = param_dict
#        nu = 1.e9*np.asarray(nu_ghz)
#        
    elif (comp == 'CIB'):
        # CIB SED parameter choices in dict file: Tdust_CIB [K], beta_CIB, nu0_CIB [GHz]
        # N.B. overall amplitude is not meaningful here; output ILC map (if you tried to preserve this component) would not be in sensible units
        if param_dict_file is None:
            p = read_param_dict_from_yaml()
        else:
            p = read_param_dict_from_yaml(param_dict_file)
        nu = 1.e9*np.asarray(nu_ghz)
        X_CIB = hplanck*nu/(kboltz*(p['Tdust_CIB']))
        nu0_CIB = p['nu0_CIB_ghz']*1.e9
        X0_CIB = hplanck*nu0_CIB/(kboltz*(p['Tdust_CIB']))
        return (nu/nu0_CIB)**(3.0+(p['beta_CIB'])) * ((np.exp(X0_CIB) - 1.0) / (np.exp(X_CIB) - 1.0)) * (ItoDeltaT(np.asarray(nu_ghz))/ItoDeltaT(p['nu0_CIB_ghz']))
    elif (comp == 'CIB_Jysr'): #same as CIB above but in Jy/sr instead of uK_CMB
        # CIB SED parameter choices in dict file: Tdust_CIB [K], beta_CIB, nu0_CIB [GHz]
        # N.B. overall amplitude is not meaningful here; output ILC map (if you tried to preserve this component) would not be in sensible units
        if param_dict_file is None:
            p = read_param_dict_from_yaml()
        else:
            p = read_param_dict_from_yaml(param_dict_file)
        nu = 1.e9*np.asarray(nu_ghz)
        X_CIB = hplanck*nu/(kboltz*(p['Tdust_CIB']))
        nu0_CIB = p['nu0_CIB_ghz']*1.e9
        X0_CIB = hplanck*nu0_CIB/(kboltz*(p['Tdust_CIB']))
        return (nu/nu0_CIB)**(3.0+(p['beta_CIB'])) * ((np.exp(X0_CIB) - 1.0) / (np.exp(X_CIB) - 1.0))
    else:
        print("unknown component specified")
        quit()
######################################


######################################
# spectral functions of physical components, evaluated for non-trivial bandpasses
# N.B. overall amplitudes are (generally) not meaningful; this function gives relative conversions between frequencies, for each component SED
# convention is that the maps being modeled are in uK_CMB units
# bandpass file columns should be [freq (GHz)] [transmission (unit-integral-normalized)]
######################################
def get_mix_bandpassed(bp_list, comp, param_dict_file=None): #bp_list = list containing strings of bandpass filenames; comp = string containing component name; param_dict_file = dictionary of SED parameters and values (optional, and only needed for non-first-principles SEDs)
    assert (comp != None)
    assert (bp_list != None)
    N_freqs = len(bp_list)
    output = np.zeros(N_freqs)
    if (comp == 'CMB' or comp == 'kSZ'): #CMB (or kSZ)
        return np.ones(N_freqs) #this is unity by definition, since we're working in Delta T units [uK_CMB]; output ILC map will thus also be in uK_CMB
    elif (comp == 'tSZ' or comp == 'mu'): #Thermal SZ (y-type distortion) or mu-type distortion
        # following Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf -- N.B. IMPORTANT TYPO IN THEIR EQ. 35 -- see https://www.aanda.org/articles/aa/pdf/2014/11/aa21531-13.pdf
        i=0
        for bp in bp_list:
            nu_ghz, trans = np.loadtxt(bp, usecols=(0,1), unpack=True)
            output[i] = np.trapz(trans * dBnudT(nu_ghz) * get_mix(nu_ghz, comp), nu_ghz) / np.trapz(trans * dBnudT(nu_ghz), nu_ghz)
            i+=1
        return output #this is the response at each frequency channel in uK_CMB for a signal with y=1 (or mu=1)
    elif (comp == 'CIB'):
        # following Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf -- N.B. IMPORTANT TYPO IN THEIR EQ. 35 -- see https://www.aanda.org/articles/aa/pdf/2014/11/aa21531-13.pdf
        # CIB SED parameter choices in dict file: Tdust_CIB [K], beta_CIB, nu0_CIB [GHz]
        # N.B. overall amplitude is not meaningful here; output ILC map (if you tried to preserve this component) would not be in sensible units
        if param_dict_file is None:
            p = read_param_dict_from_yaml()
        else:
            p = read_param_dict_from_yaml(param_dict_file)
        i=0
        for bp in bp_list:
            nu_ghz, trans = np.loadtxt(bp, usecols=(0,1), unpack=True)
            # N.B. this expression follows from Eqs. 32 and 35 of https://www.aanda.org/articles/aa/pdf/2014/11/aa21531-13.pdf , and then noting that one also needs to first rescale the CIB emission in Jy/sr from nu0_CIB to the "nominal frequency" nu_c that appears in those equations (i.e., multiply by get_mix(nu_c, 'CIB_Jysr')).  The resulting cancellation leaves this simple expression which has no dependence on nu_c.
            output[i] = (np.trapz(trans * get_mix(nu_ghz, 'CIB_Jysr', param_dict_file), nu_ghz) / np.trapz(trans * dBnudT(nu_ghz), nu_ghz))
            i+=1
        return output/np.amax(output) #overall amplitude not meaningful, so divide by max to get numbers of order unity; output gives the relative conversion between CIB at different frequencies, for maps in uK_CMB
    else:
        print("unknown component specified")
        quit()
######################################
