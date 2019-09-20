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
# conversion from antenna temperature to CMB thermodynamic temperature
def antenna2thermoTemp(nu_ghz):
    nu = 1.e9*np.asarray(nu_ghz)
    x = hplanck*nu/(kboltz*TCMB)
    return (2.0*np.sinh(x/2.0)/x)**2.0 #from http://adsabs.harvard.edu/abs/2000ApJ...530..133T Eq. 1

# function needed for Planck bandpass integration/conversion following approach in Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf
# blackbody derivative
# units are 1e-26 Jy/sr/uK_CMB
def dBnudT(nu_ghz):
    nu = 1.e9*np.asarray(nu_ghz)
    X = hplanck*nu/(kboltz*TCMB)
    return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB_uK

# conversion from specific intensity to Delta T units (i.e., 1/dBdT|T_CMB)
#   i.e., from W/m^2/Hz/sr (1e-26 Jy/sr) --> uK_CMB
#   i.e., you would multiply a map in 1e-26 Jy/sr by this factor to get an output map in uK_CMB
def ItoDeltaT(nu_ghz):
    return 1./dBnudT(nu_ghz)

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
def get_mix(nu_ghz, comp, param_dict_file=None, param_dict_override=None): #nu_ghz = array of frequencies in GHz; comp = string containing component name; param_dict_file = dictionary of SED parameters and values (optional, and only needed for some SEDs)
    assert (comp != None)
    nu_ghz = np.atleast_1d(nu_ghz) #catch possible scalar input
    assert (len(nu_ghz) > 0)


    def _setp():
        if param_dict_override is not None:
            assert param_dict_file is None
            v = param_dict_override
        else:
            if param_dict_file is None:
                v = default_dict
            else:
                v = read_param_dict_from_yaml(param_dict_file)
        return v


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
        p = _setp()
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
        p = _setp()
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
        p = _setp()
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

def get_scaled_beams(ells,lbeam,cen_nu_ghz,nus_ghz,ccor_exp=-1):
    """
    Scale a beam specified at multipoles ells with beam transfer
    factors lbeam (normalized to 1 at ell=0) and central frequency
    cen_nu_ghz onto the target frequencies nus_ghz with an exponent
    ccor_exp.

    Parameters
    ----------

    ells : array_like
        A 1d (nells,) array specifying what multipoles correspond to 
        beam transfer factors in lbeam

    lbeam : array_like
        A 1d (nells,) array specifying beam transfer factors normalized
        such that lbeam(ell=0) = 1

    cen_nu_ghz : float
        The "central frequency" in GHz to which lbeam corresponds

    nus_ghz : array_like
        A 1d (nfreqs,) array of frequencies in GHz on to which the 
        beam lbeam should be scaled to

    ccor_exp : float, optional
        The exponent of the beam scaling. Defaults to -1, corresponding
        to diffraction limited optics.

    """
    from orphics import maps
    fbnus = maps.interp(ells,lbeam[None,:],fill_value=(lbeam[0],lbeam[-1]))
    bnus = fbnus(((cen_nu_ghz/nus_ghz)**(-ccor_exp))*ells[:,None])[0].swapaxes(0,1)
    bnus = bnus / bnus[:,:1]
    return bnus



######################################
# spectral functions of physical components, evaluated for non-trivial bandpasses
# N.B. overall amplitudes are (generally) not meaningful; this function gives relative conversions between frequencies, for each component SED
# convention is that the maps being modeled are in uK_CMB units
# bandpass file columns should be [freq (GHz)] [transmission]  (any other columns are ignored)
# bp_list can contain entries that are None, which correspond to maps that have no CMB-relevant (or CIB) signals in them (e.g., HI maps)
######################################
def get_mix_bandpassed(bp_list, comp, param_dict_file=None,bandpass_shifts=None,
                       ccor_cen_nus=None, ccor_beams=None, ccor_exps = None, 
                       normalize_cib=True,param_dict_override=None):

    """
    Get mixing factors for a given component that have "color corrections" that account for
    a non-delta-function bandpass and for possible variation of the beam within the bandpass.
    If the latter is provided, the resulting output is of shape [Nfreqs,nells], otherwise
    the output is of shape [Nfreqs,].

    Parameters
    ----------
    bp_list : list of strings
        a list of strings of length Nfreqs where each string is the filename for a file
        containing a specification of the bandpass for that frequency channel. For each
        file, the first column is frequency in GHz and the second column is the transmission
        whose overall normalization does not matter.

    comp : string
        a string specifying the component whose mixing is requested. Currently, the following are
        supported (1) CMB or kSZ (considered identical, and always returns ones) 
        (2) tSZ (3) mu (4) rSZ (5) CIB


    param_dict_file : string, optional
        filename of a YAML file used to create a dictionary of SED parameters and values 
        (only needed for some SEDs). If None, defaults to parameters specified in 
        input/fg_SEDs_default_params.yml.


    bandpass_shifts : list of floats, optional
        A list of floats of length [Nfreqs,] specifying how much in GHz to shift the 
        entire bandpass. Each value can be positive (shift right) or negative (shift left).
        If None, no shift is applied and the bandpass specified in the files is used as is.
    

    ccor_cen_nus : list of floats, optional
        If not None, this indicates that the dependence of the beam on frequency with the
        bandpass should be taken into account. ccor_cen_nus will then be interpreted as a
        [Nfreqs,] length list of the "central frequencies" of each bandpass in GHz.
        The provided beams in ccor_beams for each channel are then scaled by
        (nu/nu_central)**ccor_exp where ccor_exp defaults to -1.
    
    
    ccor_beams : list of array_like, optional
        Only used if ccor_cen_nus is not None. In that mode, ccor_beams is interpreted as
        an [Nfreqs,] length list where each element is a 1d numpy array specifying the
        beam transmission starting from ell=0 and normalized to one at ell=0.
        The provided beams for each channel are then scaled by
        (nu/nu_central)**ccor_exp where ccor_exp defaults to -1 and nu_central is specified
        through ccor_cen_nus. If any list element is None, no scale dependent color correction
        is applied for that frequency channel. See get_scaled_beams for more information.
    


    ccor_exps : list of floats, optional
        Only used if ccor_cen_nus is not None. Defaults to -1 for each frequncy channel. 
        This controls how the beam specified in ccor_beams for the central frequencies 
        specified in ccor_cen_nus is scaled to other frequencies.
    

    """

    if bandpass_shifts is not None and np.any(np.array(bandpass_shifts)!=0):
        print("WARNING: shifted bandpasses provided.")
    assert (comp != None)
    assert (bp_list != None)
    N_freqs = len(bp_list)

    if ccor_cen_nus is not None:
        assert len(ccor_cen_nus)==N_freqs
        assert len(ccor_beams)==N_freqs
        lmaxs = []
        for i in range(N_freqs):
            if ccor_beams[i] is not None:
                assert ccor_beams[i].ndim==1
                lmaxs.append( ccor_beams[i].size )
        if len(lmaxs)==0:
            ccor_cen_nus = None
            shape = N_freqs
        else:
            lmax = max(lmaxs)
            shape = (N_freqs,lmax)

        if ccor_exps is None: ccor_exps = [-1]*N_freqs

    else:
        shape = N_freqs


    if (comp == 'CIB'):
        if param_dict_file is None:
            p = default_dict
        else:
            p = read_param_dict_from_yaml(param_dict_file)


    if (comp == 'CMB' or comp == 'kSZ'): #CMB (or kSZ)
        output = np.ones(shape) #this is unity by definition, since we're working in Delta T units [uK_CMB]; output ILC map will thus also be in uK_CMB
        for i in range(N_freqs):
            if(bp_list[i] == None): #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in bp_list
                output[i] = 0.
        return output

    else:

        output = np.zeros(shape)

        for i,bp in enumerate(bp_list):
            if (bp_list[i] != None):
                nu_ghz, trans = np.loadtxt(bp, usecols=(0,1), unpack=True)
                if bandpass_shifts is not None: nu_ghz = nu_ghz + bandpass_shifts[i]

                
                lbeam = 1
                bnus = 1
                if ccor_cen_nus is not None: 
                    if ccor_beams[i] is not None:
                        lbeam = ccor_beams[i]
                        ells = np.arange(lbeam.size)
                        cen_nu_ghz = ccor_cen_nus[i]
                        bnus = get_scaled_beams(ells,lbeam,cen_nu_ghz,nu_ghz,ccor_exp=ccor_exps[i]).swapaxes(0,1)
                        assert np.all(np.isfinite(bnus))


                if (comp == 'tSZ' or comp == 'mu' or comp == 'rSZ'): 
                    # Thermal SZ (y-type distortion) or mu-type distortion or relativistic tSZ
                    # following Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf 
                    # -- N.B. IMPORTANT TYPO IN THEIR EQ. 35 -- see https://www.aanda.org/articles/aa/pdf/2014/11/aa21531-13.pdf
                    val = np.trapz(trans * dBnudT(nu_ghz) * bnus * get_mix(nu_ghz, comp), nu_ghz) / np.trapz(trans * dBnudT(nu_ghz), nu_ghz) / lbeam
                    # this is the response at each frequency channel in uK_CMB for a signal with y=1 (or mu=1)
                elif (comp == 'CIB'):
                    # following Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf 
                    # -- N.B. IMPORTANT TYPO IN THEIR EQ. 35 -- see https://www.aanda.org/articles/aa/pdf/2014/11/aa21531-13.pdf
                    # CIB SED parameter choices in dict file: Tdust_CIB [K], beta_CIB, nu0_CIB [GHz]
                    # N.B. overall amplitude is not meaningful here; output ILC map (if you tried to preserve this component) would not be in sensible units
                    val = (np.trapz(trans * get_mix(nu_ghz, 'CIB_Jysr', param_dict_file=param_dict_file,param_dict_override=param_dict_override) * bnus , nu_ghz) / np.trapz(trans * dBnudT(nu_ghz), nu_ghz)) / lbeam
                    # N.B. this expression follows from Eqs. 32 and 35 of 
                    # https://www.aanda.org/articles/aa/pdf/2014/11/aa21531-13.pdf , 
                    # and then noting that one also needs to first rescale the CIB emission 
                    # in Jy/sr from nu0_CIB to the "nominal frequency" nu_c that appears in 
                    # those equations (i.e., multiply by get_mix(nu_c, 'CIB_Jysr')).  
                    # The resulting cancellation leaves this simple expression which has no dependence on nu_c.
                else:
                    print("unknown component specified")
                    raise NotImplementedError

                if (ccor_cen_nus is not None) and (ccor_beams[i] is not None): val[lbeam==0] = 0
                output[i] = val
                assert np.all(np.isfinite(val))

            elif (bp_list[i] is None): 
                #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in bp_list
                output[i] = 0.        


        if (comp == 'CIB') and normalize_cib:        
            #overall amplitude not meaningful, so divide by max to get numbers of order unity; 
            # output gives the relative conversion between CIB at different frequencies, for maps in uK_CMB
            omax = output.max(axis=0)
            ret = output / omax
            if (ccor_cen_nus is not None): ret[:,omax==0] = 0
        else:
            ret = output

        
        assert np.all(np.isfinite(ret))
        return ret


######################################


def get_test_fdict():
    import glob
    nus = np.geomspace(10,1000,100)
    comps = ['CMB','kSZ','tSZ','mu','rSZ','CIB','CIB_Jysr']
    dirname = os.path.dirname(os.path.abspath(__file__))
    bp_list = glob.glob(dirname+"/../data/*.txt") + [None]


    fdict = {}
    fdict['mix0'] = {}
    fdict['mix1'] = {}

    for comp in comps:
        mixes = get_mix(nus, comp)
        fdict['mix0'][comp] = mixes.copy()
        if comp!='CIB_Jysr':
            fdict['mix1'][comp] = {}
            mixes_bp = get_mix_bandpassed(bp_list, comp)
            for i in range(len(bp_list)):
                fdict['mix1'][comp][os.path.basename(str(bp_list[i]))] = mixes_bp[i]
    return fdict


