from __future__ import print_function
from orphics import maps,io,cosmology,catalogs,stats
from pixell import enmap,reproject
import numpy as np
import os,sys
from soapack import interfaces as sints
import tilec.fg as tfg
import tilec.utils as tutils

random = False
cversion = 'joint'
#region = 'boss'
region = 'deep56'

mask = sints.get_act_mr3_crosslinked_mask(region)
dm = sints.ACTmr3(region=mask,calibrated=True)
modlmap = mask.modlmap()
lmin = 2000
kmask = maps.mask_kspace(mask.shape,mask.wcs,lmin=lmin,lmax=30000)
kbeam150 = dm.get_beam(modlmap, "s15",region,"pa3_f150",sanitize=True)
kbeam090 = dm.get_beam(modlmap, "s15",region,"pa3_f090",sanitize=True)
m150 = maps.filter_map(dm.get_coadd("s15",region,"pa3_f150",srcfree=True,ncomp=1)[0]*mask,kmask*kbeam150)
m090 = maps.filter_map(dm.get_coadd("s15",region,"pa3_f090",srcfree=True,ncomp=1)[0]*mask,kmask*kbeam090)
m1502 = maps.filter_map(dm.get_coadd("s15",region,"pa2_f150",srcfree=True,ncomp=1)[0]*mask,kmask*kbeam150)
m1503 = maps.filter_map(dm.get_coadd("s15",region,"pa1_f150",srcfree=True,ncomp=1)[0]*mask,kmask*kbeam150)
m1504 = maps.filter_map(dm.get_coadd("s14",region,"pa2_f150",srcfree=True,ncomp=1)[0]*mask,kmask*kbeam150)
m1505 = maps.filter_map(dm.get_coadd("s15",region,"pa1_f150",srcfree=True,ncomp=1)[0]*mask,kmask*kbeam150)

fname = os.environ['WORK'] + "/data/boss/eboss_dr14/data_DR14_QSO_S.fits"
#fname = os.environ['WORK'] + "/data/boss/boss_dr12/galaxy_DR12v5_CMASS_North.fits"
#fname = os.environ['WORK'] + "/data/boss/boss_dr12/galaxy_DR12v5_CMASS_South.fits"
cols = catalogs.load_fits(fname,['RA','DEC'])
ras = cols['RA']
decs = cols['DEC']
sns = np.array(ras)*0 + 6

tdir = '/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919'

bmask = mask.copy()
bmask[bmask<0.99] = 0
#io.hplot(enmap.downgrade(bmask,4),"fig_qso_bmask")

solution = 'tsz'

bfile = tutils.get_generic_fname(tdir,region,solution,None,cversion,beam=True)
yfile = tutils.get_generic_fname(tdir,region,solution,None,cversion)

bfile2 = tutils.get_generic_fname(tdir,region,solution,'cib',cversion,beam=True)
yfile2 = tutils.get_generic_fname(tdir,region,solution,'cib',cversion)

cmap = enmap.read_map(yfile2)
modlmap = cmap.modlmap()
ls1,b1 = np.loadtxt(bfile,unpack=True)
ls2,b2 = np.loadtxt(bfile2,unpack=True)
obeam1 = maps.interp(ls1,b1)(modlmap)
obeam2 = maps.interp(ls2,b2)(modlmap)
beam_ratio = obeam2/obeam1
#beam_ratio = obeam1/obeam2
beam_ratio[~np.isfinite(beam_ratio)] = 0

ymap = enmap.read_map(yfile)

#ymap = maps.filter_map(ymap,beam_ratio)
#cmap = maps.filter_map(cmap,beam_ratio)



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

    # if ra>15: continue
        
    mcut = reproject.cutout(bmask, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    if mcut is None: 
        continue
    if np.any(mcut)<=0: 
        continue

    

    ycut = reproject.cutout(ymap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    ccut = reproject.cutout(cmap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    cut150 = reproject.cutout(m150, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    cut090 = reproject.cutout(m090, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    cut1502 = reproject.cutout(m1502, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    cut1503 = reproject.cutout(m1503, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    cut1504 = reproject.cutout(m1504, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
    cut1505 = reproject.cutout(m1505, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)


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
    s.add_to_stack("c150",cut150)
    s.add_to_stack("c090",cut090)
    s.add_to_stack("c1502",cut1502)
    s.add_to_stack("c1503",cut1503)
    s.add_to_stack("c1504",cut1504)
    s.add_to_stack("c1505",cut1505)
    i = i + 1
    # if i>430 00: break

print(i)



s.get_stats()
s.get_stacks()
ystack = s.stacks['ystack']
cstack = s.stacks['cstack']
c150 = s.stacks['c150']
c1502 = s.stacks['c1502']
c1503 = s.stacks['c1503']
c1504 = s.stacks['c1504']
c1505 = s.stacks['c1505']
c090 = s.stacks['c090']
_,nwcs = enmap.geometry(pos=(0,0),shape=ystack.shape,res=np.deg2rad(0.5/60.))
io.hplot(enmap.enmap(ystack,nwcs),"fig_qso_ystack_%s_%s"  % (cversion,region),ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,quantile=1e-3)
io.hplot(enmap.enmap(cstack,nwcs),"fig_qso_cstack_%s_%s"  % (cversion,region),ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,quantile=1e-3)
io.hplot(enmap.enmap(c150,nwcs),"fig_qso_c150_%s_%s"  % (cversion,region),ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,quantile=1e-3)
io.hplot(enmap.enmap(c1502,nwcs),"fig_qso_c1502_%s_%s"  % (cversion,region),ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,quantile=1e-3)
io.hplot(enmap.enmap(c1503,nwcs),"fig_qso_c1503_%s_%s"  % (cversion,region),ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,quantile=1e-3)
io.hplot(enmap.enmap(c1504,nwcs),"fig_qso_c1504_%s_%s"  % (cversion,region),ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,quantile=1e-3)
io.hplot(enmap.enmap(c1505,nwcs),"fig_qso_c1505_%s_%s"  % (cversion,region),ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,quantile=1e-3)
io.hplot(enmap.enmap(c090,nwcs),"fig_qso_c090_%s_%s"  % (cversion,region),ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,quantile=1e-3)
io.hplot(enmap.enmap(cstack-ystack,nwcs),"fig_qso_dust_%s_%s"  % (cversion,region),ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,quantile=1e-3)


c1d = s.stats['c1d']['mean']
ec1d = s.stats['c1d']['errmean']

y1d = s.stats['y1d']['mean']
ey1d = s.stats['y1d']['errmean']

pl = io.Plotter(xlabel='$\\theta$ (arcmin)',ylabel='Filtered $Y  (\\times 10^6)$')
pl.add_err(cents,c1d,yerr=ec1d,label="CIB deproj",ls="none",marker="_",markersize=8,elinewidth=2,mew=2)
pl.add_err(cents+0.1,y1d,yerr=ey1d,label='Compton-Y map',ls="none",marker="x",markersize=8,elinewidth=2,mew=2)
pl.hline(y=0)
pl._ax.set_ylim(-0.05,0.35)
pl.done('fig_qso_yprofile_%s_%s.png' % (cversion,region))

print((c1d-y1d)/y1d*100.)

    


