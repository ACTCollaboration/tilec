from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from fg import dBnudT,get_mix
"""
reproduce various conversion factors for HFI bandpasses from Table 6 of https://arxiv.org/pdf/1303.5070.pdf
"""
TCMB = 2.726  # Kelvin
TCMB_uK = 2.726e6  # micro-Kelvin
hplanck = 6.626068e-34  # MKS
kboltz = 1.3806503e-23  # MKS
clight = 299792458.0  # MKS
clight_cmpersec = 2.99792458*1.e10 #speed of light in cm/s
N_freqs = 6
HFI_freqs = []
HFI_freqs.append('100')
HFI_freqs.append('143')
HFI_freqs.append('217')
HFI_freqs.append('353')
HFI_freqs.append('545')
HFI_freqs.append('857')
HFI_freqs_GHz = np.array([100.0, 143.0, 217.0, 353.0, 545.0, 857.0])
HFI_files = []
for i in xrange(N_freqs):
    print("----------")
    print(HFI_freqs[i])
    HFI_files.append('../data/HFI_BANDPASS_F'+HFI_freqs[i]+'_reformat.txt')
    HFI_loc = np.loadtxt(HFI_files[i])
    # check norm, i.e., make sure response is unity for CMB
    HFI_loc_GHz = HFI_loc[:,0]
    HFI_loc_trans = HFI_loc[:,1]
    print("CMB norm = ", np.trapz(HFI_loc_trans, HFI_loc_GHz))
    # check various conversions against Table 6 of https://arxiv.org/pdf/1303.5070.pdf
    # compute K_CMB -> y_SZ conversion
    print("K_CMB -> y_SZ conversion: ", np.trapz(HFI_loc_trans*dBnudT(HFI_loc_GHz)*1.e6, HFI_loc_GHz) / np.trapz(HFI_loc_trans*dBnudT(HFI_loc_GHz)*1.e6*get_mix(HFI_loc_GHz,'tSZ')/TCMB_uK, HFI_loc_GHz) / TCMB)
    # compute K_CMB -> MJy/sr conversion [IRAS convention, alpha=-1 power-law SED]
    print("K_CMB -> MJy/sr conversion [IRAS convention, alpha=-1 power-law SED]: ", np.trapz(HFI_loc_trans*dBnudT(HFI_loc_GHz)*1.e6, HFI_loc_GHz) / np.trapz(HFI_loc_trans*(HFI_freqs_GHz[i]/HFI_loc_GHz), HFI_loc_GHz) * 1.e20)
    # compute color correction from IRAS to "dust" (power-law with alpha=4)
    print("MJy/sr color correction (power-law, alpha=-1 to alpha=4): ", np.trapz(HFI_loc_trans*(HFI_freqs_GHz[i]/HFI_loc_GHz), HFI_loc_GHz) / np.trapz(HFI_loc_trans*(HFI_loc_GHz/HFI_freqs_GHz[i])**4.0, HFI_loc_GHz))
    # compute color correction from IRAS to modified blackbody with T=13.6 K, beta=1.4 (to compare to results at https://wiki.cosmos.esa.int/planckpla2015/index.php/UC_CC_Tables )
    print("MJy/sr color correction (power-law alpha=-1 to MBB T=13.6 K/beta=1.4): ", np.trapz(HFI_loc_trans*(HFI_freqs_GHz[i]/HFI_loc_GHz), HFI_loc_GHz) / np.trapz(HFI_loc_trans*(HFI_loc_GHz/HFI_freqs_GHz[i])**(1.4+3.) * (np.exp(hplanck*HFI_freqs_GHz[i]*1.e9/(kboltz*13.6))-1.)/(np.exp(hplanck*HFI_loc_GHz*1.e9/(kboltz*13.6))-1.), HFI_loc_GHz))
    print("----------")
