import numpy as np
import matplotlib.pyplot as plt
from fg import dBnudT_KCMB,get_mix
"""
compute various conversion factors for LFI bandpasses
"""
TCMB = 2.726  # Kelvin
TCMB_uK = 2.726e6  # micro-Kelvin
hplanck = 6.626068e-34  # MKS
kboltz = 1.3806503e-23  # MKS
clight = 299792458.0  # MKS
clight_cmpersec = 2.99792458*1.e10 #speed of light in cm/s
N_freqs = 3
LFI_freqs = []
LFI_freqs.append('030')
LFI_freqs.append('044')
LFI_freqs.append('070')
LFI_freqs_float = np.array([30.0e9, 44.0e9, 70.0e9])
LFI_files = []
for i in xrange(N_freqs):
    print "----------"
    print LFI_freqs[i]
    LFI_files.append('../data/LFI_BANDPASS_F'+LFI_freqs[i]+'_reformat.txt')
    LFI_loc = np.loadtxt(LFI_files[i])
    # check norm
    LFI_loc_Hz = LFI_loc[:,0]
    LFI_loc_trans = LFI_loc[:,1]
    print "norm = ", np.trapz(LFI_loc_trans, LFI_loc_Hz)
    # compute K_CMB -> y_SZ conversion
    print "K_CMB -> y_SZ conversion: ", np.trapz(LFI_loc_trans*dBnudT_KCMB(LFI_loc_Hz/1.e9), LFI_loc_Hz) / np.trapz(LFI_loc_trans*dBnudT_KCMB(LFI_loc_Hz/1.e9)*get_mix(LFI_loc_Hz/1.e9,'tSZ')/TCMB_uK, LFI_loc_Hz) / TCMB
    # compute K_CMB -> MJy/sr conversion [IRAS convention, alpha=-1 power-law SED]
    print "K_CMB -> MJy/sr conversion [IRAS convention, alpha=-1 power-law SED]: ", np.trapz(LFI_loc_trans*dBnudT_KCMB(LFI_loc_Hz/1.e9), LFI_loc_Hz) / np.trapz(LFI_loc_trans*(LFI_freqs_float[i]/LFI_loc_Hz), LFI_loc_Hz) * 1.e20
    # compute color correction from IRAS to "dust" (power-law with alpha=4)
    print "MJy/sr color correction (power-law, alpha=-1 to alpha=4): ", np.trapz(LFI_loc_trans*(LFI_freqs_float[i]/LFI_loc_Hz), LFI_loc_Hz) / np.trapz(LFI_loc_trans*(LFI_loc_Hz/LFI_freqs_float[i])**4.0, LFI_loc_Hz)
    # compute color correction from IRAS to modified blackbody with T=13.6 K, beta=1.4 (to compare to results at https://wiki.cosmos.esa.int/planckpla2015/index.php/UC_CC_Tables )
    print "MJy/sr color correction (power-law alpha=-1 to MBB T=13.6 K/beta=1.4): ", np.trapz(LFI_loc_trans*(LFI_freqs_float[i]/LFI_loc_Hz), LFI_loc_Hz) / np.trapz(LFI_loc_trans*(LFI_loc_Hz/LFI_freqs_float[i])**(1.4+3.) * (np.exp(hplanck*LFI_freqs_float[i]/(kboltz*13.6))-1.)/(np.exp(hplanck*LFI_loc_Hz/(kboltz*13.6))-1.), LFI_loc_Hz)
    print "----------"
