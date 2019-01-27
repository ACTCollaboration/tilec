from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from tilec.fg import dBnudT,get_mix
"""
reproduce various ``effective frequencies'' for ACT bandpasses from https://phy-wiki.princeton.edu/polwiki/pmwiki.php?n=Calibration.DetectorPassbands?action=download&upname=ACTPol_eff_cen_freq.pdf
note that we will consider two assumptions for the scaling of the beam width with frequency within a given bandpass, either alpha=0 or alpha=-1 where FWHM \propto nu^alpha
we will then take the average of the two results to obtain the final effective frequency -- see the pdf linked above for further details.
Update: actually we just want alpha=-1 for diffuse sources
Also: PA4/5/6 are currently unusable.  Leave out for now.  For PA1/2/3, use files with upper and lower truncations determined in my updated/corrected version of Rahul's notebook.
"""
TCMB = 2.726  # Kelvin
TCMB_uK = 2.726e6  # micro-Kelvin
hplanck = 6.626068e-34  # MKS
kboltz = 1.3806503e-23  # MKS
clight = 299792458.0  # MKS
clight_cmpersec = 2.99792458*1.e10 #speed of light in cm/s
ACT_file_dir = '../data/'
#N_freqs = 10 #PA1/2/3/4/5/6
#ACT_freqs_GHz = np.array([150.0, 150.0, 90.0, 150.0, 150.0, 220.0, 90.0, 150.0, 90.0, 150.0])
#ACT_files = ['PA1_avg_passband_wErr.txt','PA2_avg_passband_wErr.txt','PA3_avg_passband_90_wErr.txt','PA3_avg_passband_150_wErr.txt','PA4_avg_passband_150_wErr.txt','PA4_avg_passband_220_wErr.txt','PA5_avg_passband_90_wErr.txt','PA5_avg_passband_150_wErr.txt','PA6_avg_passband_90_wErr.txt','PA6_avg_passband_150_wErr.txt']
N_freqs = 4 #PA1/2/3 only
ACT_freqs_GHz = np.array([150.0, 150.0, 90.0, 150.0])
ACT_file_dir = '../data/'
ACT_files = ['PA1_avg_passband_wErr_trunc.txt','PA2_avg_passband_wErr_trunc.txt','PA3_avg_passband_90_wErr_trunc.txt','PA3_avg_passband_150_wErr_trunc.txt']
al_1 = 0.0
al_2 = -1.0 #diffuse sources
for i in xrange(N_freqs):
    print("----------")
    print(i,ACT_freqs_GHz[i],ACT_files[i])
    ACT_loc = np.loadtxt(ACT_file_dir+ACT_files[i])
    ACT_loc_GHz = ACT_loc[:,0]
    ACT_loc_trans = ACT_loc[:,1]
    ACT_loc_trans_err = ACT_loc[:,2]
    nu = ACT_loc_GHz * 1.e9 #Hz
    dnu = nu[1]-nu[0] #all of the ACT bandpass files have equally spaced samples in frequency (though the spacing is different for different bandpasses)
    fnu = ACT_loc_trans
    # effective frequency for CMB -- following Rahul's notebook
    nu_be_CMB = dBnudT(ACT_loc_GHz)*TCMB_uK
    nu_eff_CMB1 = np.sum(nu * fnu * nu**(-2.*(1.+al_1)) * nu_be_CMB * dnu)/np.sum(fnu * nu**(-2.*(1.+al_1)) * nu_be_CMB * dnu)
    nu_eff_CMB2 = np.sum(nu * fnu * nu**(-2.*(1.+al_2)) * nu_be_CMB * dnu)/np.sum(fnu * nu**(-2.*(1.+al_2)) * nu_be_CMB * dnu)
    nu_eff_CMB = 0.5*(nu_eff_CMB1+nu_eff_CMB2)
    print('nu_eff_CMB1 =', "{0:.1f}".format(nu_eff_CMB1/1.e9))
    print('nu_eff_CMB2 =', "{0:.1f}".format(nu_eff_CMB2/1.e9))
    print('nu_eff_CMB =', "{0:.1f}".format(nu_eff_CMB/1.e9), '+/-', 2.4, 'GHz') #uncertainty is due to systematic uncertainty in FTS measurements
    # effective frequency for tSZ -- following Rahul's notebook
    nu_be_SZ = nu_be_CMB * get_mix(ACT_loc_GHz, 'tSZ')/TCMB_uK
    nu_eff_SZ1 = np.sum(nu * fnu * nu**(-2.*(1.+al_1)) * nu_be_SZ * dnu)/np.sum(fnu * nu**(-2.*(1.+al_1)) * nu_be_SZ * dnu)
    nu_eff_SZ2 = np.sum(nu * fnu * nu**(-2.*(1.+al_2)) * nu_be_SZ * dnu)/np.sum(fnu * nu**(-2.*(1.+al_2)) * nu_be_SZ * dnu)
    nu_eff_SZ = 0.5*(nu_eff_SZ1+nu_eff_SZ2)
    print('nu_eff_SZ1 =', "{0:.1f}".format(nu_eff_SZ1/1.e9))
    print('nu_eff_SZ2 =', "{0:.1f}".format(nu_eff_SZ2/1.e9))
    print('nu_eff_SZ =', "{0:.1f}".format(nu_eff_SZ/1.e9), '+/-', 2.4, 'GHz')
    # compute K_CMB -> y_SZ conversion
    print("K_CMB -> y_SZ conversion: ", np.trapz(ACT_loc_trans*dBnudT(ACT_loc_GHz)*1.e6, ACT_loc_GHz) / np.trapz(ACT_loc_trans*dBnudT(ACT_loc_GHz)*1.e6*get_mix(ACT_loc_GHz,'tSZ')/TCMB_uK, ACT_loc_GHz) / TCMB)
    # compute K_CMB -> MJy/sr conversion [IRAS convention, alpha=-1 power-law SED]
    print("K_CMB -> MJy/sr conversion [IRAS convention, alpha=-1 power-law SED]: ", np.trapz(ACT_loc_trans*dBnudT(ACT_loc_GHz)*1.e6, ACT_loc_GHz) / np.trapz(ACT_loc_trans*(ACT_freqs_GHz[i]/ACT_loc_GHz), ACT_loc_GHz) * 1.e20)
    # compute color correction from IRAS to "dust" (power-law with alpha=4)
    print("MJy/sr color correction (power-law, alpha=-1 to alpha=4): ", np.trapz(ACT_loc_trans*(ACT_freqs_GHz[i]/ACT_loc_GHz), ACT_loc_GHz) / np.trapz(ACT_loc_trans*(ACT_loc_GHz/ACT_freqs_GHz[i])**4.0, ACT_loc_GHz))
    # compute color correction from IRAS to modified blackbody with T=13.6 K, beta=1.4 (to compare to results at https://wiki.cosmos.esa.int/planckpla2015/index.php/UC_CC_Tables )
    print("MJy/sr color correction (power-law alpha=-1 to MBB T=13.6 K/beta=1.4): ", np.trapz(ACT_loc_trans*(ACT_freqs_GHz[i]/ACT_loc_GHz), ACT_loc_GHz) / np.trapz(ACT_loc_trans*(ACT_loc_GHz/ACT_freqs_GHz[i])**(1.4+3.) * (np.exp(hplanck*ACT_freqs_GHz[i]*1.e9/(kboltz*13.6))-1.)/(np.exp(hplanck*ACT_loc_GHz*1.e9/(kboltz*13.6))-1.), ACT_loc_GHz))
    print("----------")
