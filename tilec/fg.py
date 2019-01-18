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
# N.B. temperature here is in K, not uK!
def dBnudT_KCMB(nu_ghz):
    nu = 1.e9*np.asarray(nu_ghz)
    X = hplanck*nu/(kboltz*TCMB)
    return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB
######################################

######################################
# dictionary of parameter values needed for some component SEDs
######################################
def read_param_dict_from_yaml(filename):
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config
# default case
fpath = os.path.dirname(__file__)
default_dict = read_param_dict_from_yaml(fpath+'/../input/fg_SEDs_default_params.yml')
#pdict = {}                                                                                                                                           
#pdict['beta'] = 1                                                                                                                                    
######################################


######################################
# spectral functions of physical components
# N.B. overall amplitudes are not meaningful; these functions give relative conversions between frequencies
# convention is that the maps being modeled here are in uK_CMB units
######################################
def get_mix(nu_ghz, comp, param_dict=None): #comp = string containing component name; param_dict = dictionary of SED parameters and values (optional, and only needed for non-first-principles SEDs)
    if (comp == 'CMB' or comp == 'kSZ'): #CMB (or kSZ)
        return np.ones(len(np.asarray(nu_ghz))) #this is unity by definition, since we're working in Delta T units [uK_CMB]
    elif (comp == 'tSZ'): #Thermal SZ (y-type distortion)
        nu = 1.e9*np.asarray(nu_ghz)
        X = hplanck*nu/(kboltz*TCMB)
        return (X / np.tanh(X/2.0) - 4.0) * TCMB_uK #put explicitly into uK_CMB units
    elif (comp == 'mu'): #mu-type distortion
        nu = 1.e9*np.asarray(nu_ghz)
        X = hplanck*nu/(kboltz*TCMB)
        return (X / 2.1923 - 1.0)/X * TCMB_uK #put explicitly into uK_CMB units 
    elif (comp == 'CIB'):
        # CIB SED parameter choices in dict file: Tdust_CIB [K], beta_CIB, nu0_CIB [GHz]
        if param_dict is None: param_dict = default_dict
        p = param_dict
        nu = 1.e9*np.asarray(nu_ghz)
        X_CIB = hplanck*nu/(kboltz*(p['Tdust_CIB']))
        nu0_CIB = p['nu0_CIB_ghz']*1.e9
        X0_CIB = hplanck*nu0_CIB/(kboltz*(p['Tdust_CIB']))
        return (nu/nu0_CIB)**(3.0+(p['beta_CIB'])) * ((np.exp(X0_CIB) - 1.0) / (np.exp(X_CIB) - 1.0)) * (ItoDeltaT(np.asarray(nu_ghz))/ItoDeltaT(p['nu0_CIB_ghz']))
    else:
        print("unknown component specified")
        quit()
