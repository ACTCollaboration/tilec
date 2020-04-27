from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
from orphics import maps,io,cosmology,catalogs,stats # msyriac/orphics ; pip install -e . --user
from pixell import enmap,reproject
import numpy as np
import os,sys
from soapack import interfaces as sints # simonsobs/soapack ; README
import tilec.fg as tfg # ACTCollaboration/tilec ; pip install -e . --user
import tilec.utils as tutils

"""
Fig 8 and Fig 10 of arxiv 1911.05717
"""

# Whether to stack on random locations
random = False

# ACT+Planck? or ACT only? or Planck only?
cversion = 'joint'

# deep56 vs boss-n
region = 'deep56'

# array to compare to fig 8
array = "pa3_f090"

# remove modes below this scale from stacks
lmin = 2000

# make stack images
do_images = True

# Load the ACT cluster catalog
cols = catalogs.load_fits("AdvACT_S18dn_confirmed_clusters.fits",['RAdeg','DECdeg','SNR'])
ras = cols['RAdeg']
decs = cols['DECdeg']
sns = cols['SNR']

# Rough central frequencies
freqs = {"pa3_f090":97, "pa3_f150":149}

# Get a region mask
mask = sints.get_act_mr3_crosslinked_mask(region)
bmask = mask.copy()
bmask[bmask<0.99] = 0

# Map loader
dm = sints.ACTmr3(region=mask,calibrated=True)
modlmap = mask.modlmap()
ells = np.arange(0,modlmap.max())
# 2D beam
kbeam = dm.get_beam(modlmap, "s15",region,array,sanitize=True)
# 1D beam
lbeam = dm.get_beam(ells, "s15",region,array,sanitize=True)

# beam and map files
# Y-maps
bfile = os.environ["WORK"] + "/data/depot/tilec/v1.2.0_20200324/map_v1.2.0_%s_%s/tilec_single_tile_%s_comptony_map_v1.2.0_%s_beam.txt" % (cversion,region,region,cversion)
yfile = os.environ["WORK"] + "/data/depot/tilec/v1.2.0_20200324/map_v1.2.0_%s_%s/tilec_single_tile_%s_comptony_map_v1.2.0_%s.fits" % (cversion,region,region,cversion)
# CMB maps
bfile2 = os.environ["WORK"] + "/data/depot/tilec/v1.2.0_20200324/map_v1.2.0_%s_%s/tilec_single_tile_%s_cmb_map_v1.2.0_%s_beam.txt" % (cversion,region,region,cversion)
yfile2 = os.environ["WORK"] + "/data/depot/tilec/v1.2.0_20200324/map_v1.2.0_%s_%s/tilec_single_tile_%s_cmb_map_v1.2.0_%s.fits" % (cversion,region,region,cversion)
# CMB deproj-tSZ maps
bfile3 = os.environ["WORK"] + "/data/depot/tilec/v1.2.0_20200324/map_v1.2.0_%s_%s/tilec_single_tile_%s_cmb_deprojects_comptony_map_v1.2.0_%s_beam.txt" % (cversion,region,region,cversion)
yfile3 = os.environ["WORK"] + "/data/depot/tilec/v1.2.0_20200324/map_v1.2.0_%s_%s/tilec_single_tile_%s_cmb_deprojects_comptony_map_v1.2.0_%s.fits" % (cversion,region,region,cversion)

# Reconvolve to the same beam and filter out modes < ellmin
ls,obells = np.loadtxt(bfile,unpack=True)
obeam = maps.interp(ls,obells)(modlmap)
beam_ratio = kbeam/obeam
beam_ratio[~np.isfinite(beam_ratio)] = 0
kmask = maps.mask_kspace(mask.shape,mask.wcs,lmin=lmin,lmax=30000)
# Get coadd map of reference array
kmap = enmap.fft(dm.get_coadd("s15",region,array,srcfree=True,ncomp=1)[0]*mask)
# Get pixel window correction
pwin = tutils.get_pixwin(mask.shape[-2:])
bpfile = "data/"+dm.get_bandpass_file_name(array)
# Apply a filter that converts array map to Compton Y units (Eq 8)
f = tfg.get_mix_bandpassed([bpfile], 'tSZ',ccor_cen_nus=[freqs[array]], ccor_beams=[lbeam])[0]
f2d = maps.interp(ells,f)(modlmap)
filt = kmask/pwin/f2d
filt[~np.isfinite(filt)] = 0
imap = enmap.ifft(kmap*filt).real

# Reconvolve Y map
ymap = maps.filter_map(enmap.read_map(yfile),beam_ratio*kmask)

# Get CMB maps
smap = enmap.read_map(yfile2)
cmap = enmap.read_map(yfile3)

# Width in pixels of stamp cutout
pix = 60

# Initialize stacks
i = 0
istack = 0
ystack = 0
sstack = 0
cstack = 0

# 1d statistics collector
s = stats.Stats()

for ra,dec,sn in zip(ras,decs,sns):

    # For random stacking
    if random:
        box = mask.box()
        ramin = min(box[0,1],box[1,1])
        ramax = max(box[0,1],box[1,1])
        decmin = min(box[0,0],box[1,0])
        decmax = max(box[0,0],box[1,0])
        ra = np.rad2deg(np.random.uniform(ramin,ramax))
        dec = np.rad2deg(np.random.uniform(decmin,decmax))

    # SNR cut
    if sn<5:
        continue
        
    # Reject based on mask cutout
    mcut = reproject.cutout(bmask, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    if mcut is None: 
        continue
    if np.any(mcut)<=0: 
        continue

    
    # Stamp cutouts
    icut = reproject.cutout(imap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    ycut = reproject.cutout(ymap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    scut = reproject.cutout(smap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    ccut = reproject.cutout(cmap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)

    # Some binning tools
    if i==0:
        modrmap = np.rad2deg(icut.modrmap())*60. # Map of distance from center
        bin_edges = np.arange(0.,10.,1.0)
        binner = stats.bin2D(modrmap,bin_edges)

    # Bin profiles
    cents,i1d = binner.bin(icut)
    cents,y1d = binner.bin(ycut)
    s.add_to_stats("i1d",i1d*1e6)
    s.add_to_stats("y1d",y1d*1e6)

    # Stack
    istack = istack + icut
    ystack = ystack + ycut
    sstack = sstack + scut
    cstack = cstack + ccut
    i += 1
    if i>5000: break

s.get_stats()
istack = istack / i * 1e6
ystack = ystack / i * 1e6
sstack = sstack / i
cstack = cstack / i

# Approximate geometry of stack
_,nwcs = enmap.geometry(pos=(0,0),shape=istack.shape,res=np.deg2rad(0.5/60.))

# Plot images
if do_images:
    io.hplot(enmap.enmap(istack,nwcs),"istack",ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,min=-5,max=25)
    io.hplot(enmap.enmap(ystack,nwcs),"ystack",ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,min=-5,max=25)

    if region=='deep56':
        io.hplot(enmap.enmap(sstack,nwcs),"fig_sstack_%s" % region,ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,min=-10,max=30)
        io.hplot(enmap.enmap(cstack,nwcs),"fig_cstack_%s" % region,ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,min=-10,max=30)
    elif region=='boss':
        io.hplot(enmap.enmap(sstack,nwcs),"fig_sstack_%s" % region,ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,min=-50,max=17)
        io.hplot(enmap.enmap(cstack,nwcs),"fig_cstack_%s" % region,ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,min=-50,max=17)


# Make profile plot
i1d = s.stats['i1d']['mean']
ei1d = s.stats['i1d']['errmean']

y1d = s.stats['y1d']['mean']
ey1d = s.stats['y1d']['errmean']

pl = io.Plotter(xlabel='$\\theta$ (arcmin)',ylabel='Filtered $y~(\\times 10^6)$',ftsize=16,labsize=14)
pl.add_err(cents,i1d,yerr=ei1d,label="Single-frequency D56_5_098 map",marker="_",markersize=8,elinewidth=2,mew=2,ls='none')#-')
pl.add_err(cents,y1d,yerr=ey1d,label='Compton-$y$ map',ls="none",marker="x",markersize=8,elinewidth=2,mew=2,addx=0.1)
pl.hline(y=0)

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

pl._ax.yaxis.set_minor_locator(AutoMinorLocator())
pl._ax.xaxis.set_minor_locator(AutoMinorLocator())
pl._ax.tick_params(axis='x',which='both', width=1)
pl._ax.tick_params(axis='y',which='both', width=1)
pl._ax.xaxis.grid(True, which='both',alpha=0.3)
pl._ax.yaxis.grid(True, which='both',alpha=0.3)

pl.done('fig_yprofile_%s.pdf' % region)
