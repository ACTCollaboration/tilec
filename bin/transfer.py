from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import covtools

bin_edges = np.arange(100,8000,40)
pfunc = lambda x,y: np.real(x*y.conj())
ffunc = lambda x: enmap.fft(x,normalize='phys')
Ncrop = 400

navg = lambda x,delta : covtools.noise_block_average(x,nsplits=1,delta_ell=delta,
                                                         radial_fit=False,lmax=None,
                                                         wnoise_annulus=None,
                                                         lmin = 40,
                                                         bin_annulus=None,fill_lmax=None,
                                                         log=False)

loc = "/home/r/rbond/sigurdkn/project/actpol/maps/mr3f_20190328/transfun/release/"

mask = sints.get_act_mr3_crosslinked_mask('deep56')
binner = stats.bin2D(mask.modlmap(),bin_edges)
bin = lambda x: binner.bin(x)

isim1 = enmap.extract(enmap.read_map(loc+'../sims/deep56_00.fits'),mask.shape,mask.wcs) * mask
isim2 = enmap.extract(enmap.read_map(loc+'../sims/deep56_01.fits'),mask.shape,mask.wcs) * mask
tmap1 = enmap.extract(enmap.read_map(loc+'s15_deep56_pa2_f150_nohwp_night_sim00_3pass_4way_coadd_transmap.fits'),mask.shape,mask.wcs) * mask
tmap2 = enmap.extract(enmap.read_map(loc+'s15_deep56_pa2_f150_nohwp_night_sim01_3pass_4way_coadd_transmap.fits'),mask.shape,mask.wcs) * mask


lmap = mask.lmap()
lymap,lxmap = lmap
# def model(x,width,amplitude,sigma):
#     mmap = 1-amplitude * np.exp(-lymap**2./2./sigma**2.)
#     mmap[lxmap>width/2.] = 1
#     mmap[lxmap<-width/2.] = 1
#     return mmap

def model(x,width,amplitude,sigma):
    mmap = (1-amplitude * np.exp(-lymap**2./2./sigma**2.))* (1-np.exp(-lxmap**2./2./width**2.))
    return mmap

m = model(0,100,1,3000)
io.power_crop(np.fft.fftshift(m),Ncrop,'model.png',ftrans=False)
# sys.exit()

# io.hplot(enmap.downgrade(isim,4),'isim')
# io.hplot(enmap.downgrade(tmap,4),'tmap')


kisim1 = ffunc(isim1[0])
kisim2 = ffunc(isim2[0])
ktmap1 = ffunc(tmap1[0])
ktmap2 = ffunc(tmap2[0])

pcross2d1 = pfunc(kisim1,ktmap1)
pcross2d2 = pfunc(kisim2,ktmap2)
psim2d1 = pfunc(kisim1,kisim1)
psim2d2 = pfunc(kisim2,kisim2)
ptmap2d1 = pfunc(ktmap1,ktmap1)
ptmap2d2 = pfunc(ktmap2,ktmap2)

cents, pcross1 = bin(pcross2d1)
cents, psim1 = bin(psim2d1)
cents, ptmap1 = bin(ptmap2d1)

cents, pcross2 = bin(pcross2d2)
cents, psim2 = bin(psim2d2)
cents, ptmap2 = bin(ptmap2d2)

r = (pcross2d1/np.sqrt(psim2d1*ptmap2d1) + pcross2d2/np.sqrt(psim2d2*ptmap2d2))/2
r[~np.isfinite(r)] = 0

fitcrop = 500
data = maps.crop_center(np.fft.fftshift(r),fitcrop)
fitfunc = lambda a,x,y,z : maps.crop_center(np.fft.fftshift(model(a,x,y,z)),fitcrop).ravel()
X = maps.crop_center(np.fft.fftshift(lxmap),fitcrop)
Y = maps.crop_center(np.fft.fftshift(lymap),fitcrop)
xdata = np.vstack((X.ravel(), Y.ravel()))
ydata = data.ravel()
from scipy.optimize import curve_fit
popt,pcov = curve_fit(fitfunc,xdata,ydata,p0=[20,0.5,3000],bounds=([1,0,100],[100,1,8000]))
print(popt)
x,y,z = popt
fit = model(0,x,y,z)



cents,r1d1 = bin(r)
# r1d2 = pcross/np.sqrt(psim*ptmap)

Nsmooth = 100

rsmooth,_,_ = navg(r,Nsmooth)

io.power_crop(np.fft.fftshift(fit),Ncrop,'fit.png',ftrans=False)
io.power_crop(np.fft.fftshift(r-fit),Ncrop,'residual.png',ftrans=False)
io.power_crop(np.fft.fftshift(r),Ncrop,'ptrans.png',ftrans=False)
io.power_crop(np.fft.fftshift(rsmooth),Ncrop,'ptrans_smooth_after.png',ftrans=False)

pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')
pl.add(cents,pcross1,label='cross')
pl.add(cents,psim1,label='sim')
pl.add(cents,ptmap1,label='tmap')
pl.done("ptrans1d1.png")

pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')
pl.add(cents,pcross2,label='cross')
pl.add(cents,psim2,label='sim')
pl.add(cents,ptmap2,label='tmap')
pl.done("ptrans1d2.png")

pl = io.Plotter(xyscale='loglin',xlabel='l',ylabel='T')
pl.add(cents,r1d1,label='bin after ratio')
# pl.add(cents,r1d2,label='ratio after bin')
pl.hline(y=1)
pl.vline(x=500)
#pl._ax.set_ylim(0.975,1.01)
pl.done("ptrans1ddiff.png")




