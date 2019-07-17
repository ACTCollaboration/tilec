from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
import healpy as hp

nside = 1024
beam = 13.315

ztt = np.load("tt.npy")*1e12
zls = np.arange(0,len(ztt),1)

froot = "/scratch/r/rbond/msyriac/data/planck/data/pr2/"
mask = hp.ud_grade(hp.read_map(froot + "COM_Mask_Lensing_2048_R2.00.fits"),nside)
mask = hp.smoothing(mask,fwhm=np.deg2rad(2))
io.mollview(mask,os.environ['WORK']+"/tiling/planck_mask.png")
fsky = mask.sum()/mask.size
print(fsky)

full = hp.read_map(froot + "LFI_SkyMap_070_1024_R2.01_full.fits")*mask*1e6
ring1 = hp.read_map(froot + "LFI_SkyMap_070_1024_R2.01_full-ringhalf-1.fits")*mask*1e6
ring2 = hp.read_map(froot + "LFI_SkyMap_070_1024_R2.01_full-ringhalf-2.fits")*mask*1e6

# full = hp.read_map(froot + "HFI_SkyMap_143_2048_R2.02_full.fits")*mask*1e6
# ring1 = hp.read_map(froot + "HFI_SkyMap_143_2048_R2.02_full-ringhalf-1.fits")*mask*1e6
# ring2 = hp.read_map(froot + "HFI_SkyMap_143_2048_R2.02_full-ringhalf-2.fits")*mask*1e6
# hm1 = hp.read_map(froot + "HFI_SkyMap_143_2048_R2.02_halfmission-1.fits")*mask*1e6
# hm2 = hp.read_map(froot + "HFI_SkyMap_143_2048_R2.02_halfmission-2.fits")*mask*1e6


# print(ring1.shape,hm1.shape)

afull = hp.map2alm(full)

aring1 = hp.map2alm(ring1)
aring2 = hp.map2alm(ring2)

# ahm1 = hp.map2alm(hm1)
# ahm2 = hp.map2alm(hm2)

bin_edges = np.append(np.arange(2,2500,80),np.arange(2500+400,6000,400))
binner = stats.bin1D(bin_edges)

cls0 = hp.alm2cl(afull,afull) / fsky 
cls1 = hp.alm2cl(aring1,aring2) / fsky 
# cls2 = hp.alm2cl(ahm1,ahm2) / fsky

pwin = hp.pixwin(hp.npix2nside(ring1.size))
print(pwin.shape,cls1.shape)

ls = np.arange(0,len(cls1),1)

cls0 = cls0 / maps.gauss_beam(ls,beam)**2. / pwin**2
cls1 = cls1 / maps.gauss_beam(ls,beam)**2. / pwin**2
# cls2 = cls2 / maps.gauss_beam(ls,beam)**2. / pwin**2

lss,cls0 = binner.binned(ls,cls0)
lss,cls1 = binner.binned(ls,cls1)
# lss,cls2 = binner.binned(ls,cls2)
# zls,clsz = binner.binned(zls,ztt)

ells,cls = np.loadtxt('theory_150.txt',unpack=True)

pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')
pl.add(ells,cls,color='green',ls='--')
# pl.add(zls,clsz,ls=':',alpha=0.4,label='zack')
pl.add(lss,cls0,ls=':',alpha=0.4,label='full')
pl.add(lss,cls1,alpha=0.4,label='ring')
# pl.add(lss,cls2,alpha=0.4,label='hm')
pl._ax.set_ylim(1e-7,1e1)
pl.done(os.environ['WORK']+"/tiling/planck_healpix_split_lfi.png")
#pl.done(os.environ['WORK']+"/tiling/planck_healpix_split.png")


sys.exit()



array = '143'
region = 'deep56'
mask = sints.get_act_mr3_crosslinked_mask(region)    
dm = sints.PlanckHybrid(region=mask)
imap0 = dm.get_split(array,0,srcfree=True,ncomp=1)
imap1 = dm.get_split(array,1,srcfree=True,ncomp=1)
diff = imap0-imap1
io.plot_img(diff,os.environ['WORK']+"/tiling/planck_diff.png")
shape,wcs = imap0.shape[-2:],imap0.wcs
modlmap = enmap.modlmap(shape,wcs)


w2 = np.mean(mask**2.)
k1 = enmap.fft(imap0*mask,normalize='phys')
k2 = enmap.fft(imap1*mask,normalize='phys')

ells = np.arange(2,8000,1)
lbeam = dm.get_beam(ells,array)
pl = io.Plotter(xlabel='l',ylabel='B')
pl.add(ells,lbeam)
pl.add(ells,maps.gauss_beam(ells,7.0))
pl.done(os.environ['WORK']+"/tiling/planck_beam.png")

ells,cls = np.loadtxt('theory_150.txt',unpack=True)

kbeam = dm.get_beam(modlmap,array)
p2d = np.real(k1*k2.conj())/w2/kbeam**2.
p2d1 = np.real(k1*k1.conj())/w2/kbeam**2.
p2d2 = np.real(k2*k2.conj())/w2/kbeam**2.
print(p2d.shape)

bin_edges = np.arange(80,8000,80)
binner = stats.bin2D(modlmap,bin_edges)
cents,p1d = binner.bin(p2d)
cents,p1d1 = binner.bin(p2d1)
cents,p1d2 = binner.bin(p2d2)

pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')
pl.add(ells,cls,color='green',ls='--')
pl.add(cents,p1d,alpha=0.3)
pl.add(cents,p1d1,alpha=0.3)
pl.add(cents,p1d2,alpha=0.3)
pl._ax.set_ylim(1e-7,1e1)
pl.done(os.environ['WORK']+"/tiling/planck_split.png")
