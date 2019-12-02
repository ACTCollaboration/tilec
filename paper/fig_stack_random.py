from __future__ import print_function
from orphics import maps,io,cosmology,catalogs,stats
from pixell import enmap,reproject
import numpy as np
import os,sys
from soapack import interfaces as sints
import tilec.fg as tfg
import tilec.utils as tutils

random = True
cversion = 'joint'
region = 'deep56'


fname = os.environ['WORK'] + "/data/boss/eboss_dr14/data_DR14_QSO_S.fits"
cols = catalogs.load_fits(fname,['RA','DEC'])
ras = cols['RA']
decs = cols['DEC']
sns = np.array(ras)*0 + 6

tdir = '/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919'

mask = sints.get_act_mr3_crosslinked_mask(region)
bmask = mask.copy()
bmask[bmask<0.99] = 0
io.hplot(enmap.downgrade(bmask,4),"fig_qso_bmask")

solution = 'tsz'

bfile = tutils.get_generic_fname(tdir,region,'cmb',None,'act',beam=True)
yfile = tutils.get_generic_fname(tdir,region,'cmb',None,'act')

bfile2 = tutils.get_generic_fname(tdir,region,'cmb',None,'planck',beam=True)
yfile2 = tutils.get_generic_fname(tdir,region,'cmb',None,'planck')

cmap = enmap.read_map(yfile2)
modlmap = cmap.modlmap()
ymap = enmap.read_map(yfile)



pix = 80

s = stats.Stats()

i = 0
for ra,dec,sn in zip(ras,decs,sns):

    if random:
        box = mask.box()
        ramin = min(box[0,1],box[1,1])
        ramax = max(box[0,1],box[1,1])
        decmin = min(box[0,0],box[1,0])
        decmax = max(box[0,0],box[1,0])
        ra = np.rad2deg(np.random.uniform(ramin,ramax))
        dec = np.rad2deg(np.random.uniform(decmin,decmax))

    if sn<5:
        continue
        
    mcut = reproject.cutout(bmask, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    if mcut is None: 
        continue
    if np.any(mcut)<=0: 
        continue

    

    ycut = reproject.cutout(ymap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    ccut = reproject.cutout(cmap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)


    if i==0:
        modrmap = np.rad2deg(ycut.modrmap())*60.
        bin_edges = np.arange(0.,15.,1.0)
        binner = stats.bin2D(modrmap,bin_edges)

    cents,y1d = binner.bin(ycut)
    cents,c1d = binner.bin(ccut)
    s.add_to_stats("c1d",c1d*1e6)
    s.add_to_stats("y1d",y1d*1e6)
    s.add_to_stack("cstack",ccut*1e6)
    s.add_to_stack("ystack",ycut*1e6)
    i = i + 1
    #if i>100: break

print(i)



s.get_stats()
s.get_stacks()
ystack = s.stacks['ystack']
cstack = s.stacks['cstack']
_,nwcs = enmap.geometry(pos=(0,0),shape=ystack.shape,res=np.deg2rad(0.5/60.))
io.hplot(enmap.enmap(ystack,nwcs),"fig_random_ystack_%s"  % cversion,ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4)
io.hplot(enmap.enmap(cstack,nwcs),"fig_random_cstack_%s"  % cversion,ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4)


c1d = s.stats['c1d']['mean']
ec1d = s.stats['c1d']['errmean']

y1d = s.stats['y1d']['mean']
ey1d = s.stats['y1d']['errmean']

pl = io.Plotter(xlabel='$\\theta$ (arcmin)',ylabel='Filtered $Y  (\\times 10^6)$')
pl.add_err(cents,c1d,yerr=ec1d,label="CIB deproj",ls="none",marker="_",markersize=8,elinewidth=2,mew=2)
pl.add_err(cents+0.1,y1d,yerr=ey1d,label='Compton-Y map',ls="none",marker="x",markersize=8,elinewidth=2,mew=2)
pl.hline(y=0)
pl.done('fig_random_yprofile_%s.png' % cversion)

print((c1d-y1d)/y1d*100.)

    


