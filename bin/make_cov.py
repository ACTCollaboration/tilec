from tilec import utils as tutils,covtool,fg
import argparse, yaml
import numpy as np
from pixell import enmap,fft,utils
from orphics import maps,io,stats


"""
Issues:
1. SOLVED by only using ells<lmax: 12 arrays form a 22 GB covmat for deep56. Can reduce resolution by 4x if lmax=5000, fit it in 5.5 GB.
2. Cross-covariances are noisy.
3. Beam is gaussian now.
4. Maps will be band-limited. Maps will require inpainting.
5. Don't have bandpass corrected tSZ yet.
6. Use Planck half mission

Do super simple sims first

"""

# Parse command line
parser = argparse.ArgumentParser(description='Make ILC maps.')
parser.add_argument("arrays", type=str,help='List of arrays named in arrays.yml.')
args = parser.parse_args()

# Load dictionary of array specs from yaml file
config = tutils.Config(arrays = args.arrays.split(','))
    
narrays = len(arrays)
freqs = []
for i in range(narrays):
    f = config.darrays[arrays[i]]['freq']
    freqs.append(f)
    

    
# Get the common map geometry from the coadd map of the first array
shape,wcs = enmap.read_fits_geometry(coaddfname(0))
Ny,Nx = shape[-2:]
# Set up a fourier space calculator (for power spectra)
fc = maps.FourierCalc(shape[-2:],wcs)

modlmap = enmap.modlmap(shape,wcs)
lmax = 5000
ells = modlmap[modlmap<lmax].reshape(-1) # unraveled disk
nells = ells.size

Scov = np.zeros((narrays,narrays,nells))
Ncov = np.zeros((narrays,narrays,nells))

for aindex1 in range(narrays):
    for aindex2 in range(aindex1,narrays) :

        print("Noise calc...")
        scov,ncov,autos = ncalc(aindex1,aindex2) # raw power spectra in full 2d space before any downsampling

        if isplanck(aindex1) and isplanck(aindex2):
            """
            Ugh Planck is so low noise at low ells that the cross spectrum
            is too close to the auto spectrum. So the inferred noise can be
            negative. So we fill in below some lmin with the mean noise between
            lmin and lmin+ldelta.
            """
            lmin = 400
            ldelta = 200
            nmean = ncov[np.logical_and(modlmap>lmin,modlmap<(lmin+ldelta))].mean()
            ncov[modlmap<lmin] = nmean
            """
            OR we can just take the absolute value below some lmin
            """
            # lmin = 600
            # ncov[modlmap<lmin] = np.abs(ncov[modlmap<lmin])

        print("Signal avg...")
        dscov = tilec.signal_average(scov,bin_width=40)

        print("Noise avg...")

        if isplanck(aindex1) or isplanck(aindex2):
            do_radial = False
        elif aindex1==aindex2:
            do_radial = True
        else:
            do_radial = False
        # dncov,_ = tilec.noise_average(ncov,radial_fit=do_radial) if ((aindex1==aindex2) or is90150(aindex1,aindex2)) else (0.,None)
        dncov,_ = tilec.noise_average(ncov,radial_fit=do_radial) if (aindex1==aindex2)  else (0.,None) # ignore 90 / 150

        if isplanck(aindex1) and isplanck(aindex2):
            lmin = maps.minimum_ell(shape,wcs)
        else:
            lmin = 300

        sel = np.logical_or(modlmap<lmin,modlmap>lmax)
        ifwhm,jfwhm = get_beams(aindex1,aindex2)
        dncov /= np.nan_to_num(maps.gauss_beam(modlmap,ifwhm)*maps.gauss_beam(modlmap,jfwhm))
        dscov /= np.nan_to_num(maps.gauss_beam(modlmap,ifwhm)*maps.gauss_beam(modlmap,jfwhm))
        dncov[sel] = 1e90 #np.inf # inf gives invertible matrix but nonsensical output, 1e90 gives noninvertible, but with eigpow sensible
        dscov[sel] = 1e90 #np.inf

        # io.plot_img((np.fft.fftshift(ncov)),"tuncov%d%d.png"%(aindex1,aindex2),aspect='auto')
        # io.plot_img(np.log10(np.fft.fftshift(scov+ncov)),"udsncov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])
        io.plot_img(np.log10(np.fft.fftshift(ncov)),"uncov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])
        io.plot_img(np.log10(np.fft.fftshift(scov)),"usncov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])

        io.plot_img(np.log10(np.fft.fftshift(dscov+dncov)),"dsncov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])
        io.plot_img(np.log10(np.fft.fftshift(dncov)),"dncov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])
        io.plot_img(np.log10(np.fft.fftshift(dscov)),"dsncov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])
        io.plot_img(np.log10(np.fft.fftshift(ncov/dncov)),"rcov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])


        Scov[aindex1,aindex2] = dscov[modlmap<lmax].reshape(-1)
        Ncov[aindex1,aindex2] = dncov[modlmap<lmax].reshape(-1)


        if aindex1!=aindex2: Scov[aindex2,aindex1] = Scov[aindex1,aindex2].copy()
        if aindex1!=aindex2: Ncov[aindex2,aindex1] = Ncov[aindex1,aindex2].copy()




Cov = Scov + Ncov
np.save("cov.npy",Cov)

#Cov = np.load("cov.npy")
Cov = np.rollaxis(Cov,2)
icinv = utils.eigpow(Cov,-1)
np.save("icinv.npy",icinv)

