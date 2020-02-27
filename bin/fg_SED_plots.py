from __future__ import print_function
import numpy as np
import os,sys
from tilec.fg import get_mix,ItoDeltaT
import matplotlib
from cycler import cycler
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])
matplotlib.use('pdf')
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
fontProperties = {'family':'sans-serif',
                  'weight' : 'normal', 'size' : 16}
import matplotlib.pyplot as plt

"""
Script that produces example plots of foreground (or signal) SEDs
"""

# assume 1 uK signal for each component at fiducial frequency of 148 GHz
nu_fid_ghz = [148.]
T_fid_uK = [1.]

# evaluate components over a range of frequencies
nu_min_ghz = 10.
nu_max_ghz = 1000.
nu_step = 1.
nu_ghz = np.arange(nu_min_ghz, nu_max_ghz+nu_step/10., nu_step)

# plot examples
# CMB temperature units
plt.clf()
plt.xlim(nu_min_ghz, nu_max_ghz)
plt.ylim(1.e-2, 1.e3)
plt.loglog(nu_ghz, T_fid_uK * get_mix(nu_ghz, 'CMB') / get_mix(nu_fid_ghz, 'CMB'), label='CMB', lw=2., ls='-')
plt.loglog(nu_ghz, np.absolute(T_fid_uK * get_mix(nu_ghz, 'tSZ') / get_mix(nu_fid_ghz, 'tSZ')), label='tSZ', lw=2., ls='--') # need absolute value due to negative values at nu<217 GHz
plt.loglog(nu_ghz, np.absolute(T_fid_uK * get_mix(nu_ghz, 'rSZ') / get_mix(nu_fid_ghz, 'rSZ')), label=r'rSZ ($kT_e = 5$ keV)', lw=2., ls='-.') # need absolute value due to negative values; this assumes fiducial kT_e value in ../input/fg_SEDs_default_params.yml
plt.loglog(nu_ghz, T_fid_uK * get_mix(nu_ghz, 'CIB') / get_mix(nu_fid_ghz, 'CIB'), label='CIB [fid.]', lw=2., ls=':') #this assumes fiducial CIB SED parameters in ../input/fg_SEDs_default_params.yml
plt.loglog(nu_ghz, T_fid_uK * get_mix(nu_ghz, 'radio') / get_mix(nu_fid_ghz, 'radio'), label='radio [fid.]', lw=2., ls=':') #this assumes fiducial radio SED parameters in ../input/fg_SEDs_default_params.yml
plt.xlabel(r'$\nu \, [{\rm GHz}]$', fontsize=17)
plt.ylabel(r'$|\Delta T_{\nu} / \Delta T_{148 \, {\rm GHz}}|$', fontsize=17)
plt.grid(alpha=0.5)
plt.legend(loc='upper left')
plt.title('CMB Temperature Units', fontsize=17)
plt.savefig('fg_SED_plot_DeltaT.pdf')

# Jy/sr units
plt.clf()
plt.xlim(nu_min_ghz, nu_max_ghz)
#plt.ylim(1.e-2, 1.e3)
plt.loglog(nu_ghz, T_fid_uK * get_mix(nu_ghz, 'CMB') * ItoDeltaT(nu_fid_ghz) / get_mix(nu_fid_ghz, 'CMB') / ItoDeltaT(nu_ghz), label='CMB', lw=2., ls='-')
plt.loglog(nu_ghz, np.absolute(T_fid_uK * get_mix(nu_ghz, 'tSZ') * ItoDeltaT(nu_fid_ghz) / get_mix(nu_fid_ghz, 'tSZ') / ItoDeltaT(nu_ghz)), label='tSZ', lw=2., ls='--') # need absolute value due to negative values at nu<217 GHz
plt.loglog(nu_ghz, np.absolute(T_fid_uK * get_mix(nu_ghz, 'rSZ') * ItoDeltaT(nu_fid_ghz) / get_mix(nu_fid_ghz, 'rSZ') / ItoDeltaT(nu_ghz)), label=r'rSZ ($kT_e = 5$ keV)', lw=2., ls='-.') # need absolute value due to negative values; this assumes fiducial kT_e value in ../input/fg_SEDs_default_params.yml
#plt.loglog(nu_ghz, T_fid_uK * get_mix(nu_ghz, 'CIB') * ItoDeltaT(nu_fid_ghz) / get_mix(nu_fid_ghz, 'CIB') / ItoDeltaT(nu_ghz), label='CIB [fid.]', lw=2., ls='-') #this assumes fiducial CIB SED parameters in ../input/fg_SEDs_default_params.yml
plt.loglog(nu_ghz, T_fid_uK * get_mix(nu_ghz, 'CIB_Jysr') / get_mix(nu_fid_ghz, 'CIB_Jysr'), label='CIB [fid.]', lw=2., ls=':') #this assumes fiducial CIB SED parameters in ../input/fg_SEDs_default_params.yml
plt.loglog(nu_ghz, T_fid_uK * get_mix(nu_ghz, 'radio_Jysr') / get_mix(nu_fid_ghz, 'radio_Jysr'), label='radio [fid.]', lw=2., ls=':') #this assumes fiducial radio SED parameters in ../input/fg_SEDs_default_params.yml
plt.xlabel(r'$\nu \, [{\rm GHz}]$', fontsize=17)
plt.ylabel(r'$|I_{\nu} / I_{148 \, {\rm GHz}}|$', fontsize=17)
plt.grid(alpha=0.5)
plt.legend(loc='upper left')
plt.title('Specific Intensity Units', fontsize=17)
plt.savefig('fg_SED_plot_Jysr.pdf')
