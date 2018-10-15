import numpy as np

# Set up SZ frequency dependence
def gnu(nu_ghz,tcmb=2.7255):
    nu = 1e9*np.asarray(nu_ghz)
    hplanck = 6.62607e-34
    kboltzmann = 1.38065e-23 
    x = hplanck*nu/kboltzmann/tcmb
    coth = np.cosh(x/2.)/np.sinh(x/2.)
    return x*coth-4.

