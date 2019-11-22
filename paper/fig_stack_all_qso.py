from __future__ import print_function
from orphics import maps,io,cosmology,catalogs,stats,mpi
from pixell import enmap,reproject
import numpy as np
import os,sys
from soapack import interfaces as sints
import tilec.fg as tfg
import tilec.utils as tutils
import random


cversion = 'joint'

region1 = 'deep56'
region2 = 'boss'


fname = os.environ['WORK'] + "/data/boss/sdss_dr14/DR14Q_v4_4.fits"
#fname = os.environ['WORK'] + "/data/boss/eboss_dr14/data_DR14_QSO_S.fits"
cols = catalogs.load_fits(fname,['RA','DEC'])
iras = cols['RA']
idecs = cols['DEC']

mask1 = sints.get_act_mr3_crosslinked_mask(region1)
mask1[mask1<0.99] = 0
mask2 = sints.get_act_mr3_crosslinked_mask(region2)
mask2[mask2<0.99] = 0

dm1 = sints.ACTmr3(region=mask1,calibrated=True)
dm2 = sints.ACTmr3(region=mask2,calibrated=True)
wt1 = dm1.get_coadd_ivar("s15",region1,"pa2_f150")
wt2 = dm2.get_coadd_ivar("s15",region2,"pa2_f150")


ras1,decs1 = catalogs.select_based_on_mask(iras,idecs,mask1)
ras2,decs2 = catalogs.select_based_on_mask(iras,idecs,mask2)
# ras1 = iras
# decs1 = idecs

tdir = '/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919'


solution = 'tsz'


yfile1 = tutils.get_generic_fname(tdir,region1,solution,None,cversion)
cfile1 = tutils.get_generic_fname(tdir,region1,solution,'cib',cversion)

yfile2 = tutils.get_generic_fname(tdir,region2,solution,None,cversion)
cfile2 = tutils.get_generic_fname(tdir,region2,solution,'cib',cversion)

ymap1 = enmap.read_map(yfile1)
cmap1 = enmap.read_map(cfile1)

ymap2 = enmap.read_map(yfile2)
cmap2 = enmap.read_map(cfile2)


arcmin = 40.
pix = 0.5


def do(ymap,cmap,mask,ras,decs,wt):


    combined = list(zip(ras, decs))
    random.shuffle(combined)
    ras[:], decs[:] = zip(*combined)


    njobs = len(ras)
    comm,rank,my_tasks = mpi.distribute(njobs)
    print("Rank %d starting" % rank)
    s = stats.Stats(comm)

    i = 0
    for task in my_tasks:

        ra = ras[task]
        dec = decs[task]

        # mcut = reproject.cutout(mask, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=int(arcmin/pix))
        mcut = reproject.postage_stamp(mask, np.deg2rad(ra), np.deg2rad(dec),arcmin,pix)
        if mcut is None: 
            continue
        if np.any(mcut)<=0: 
            continue

        # ycut = reproject.cutout(ymap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=int(arcmin/pix))
        # ccut = reproject.cutout(cmap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=int(arcmin/pix))
        # wcut = reproject.cutout(wt, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=int(arcmin/pix))

        ycut = reproject.postage_stamp(ymap, np.deg2rad(ra), np.deg2rad(dec),arcmin,pix)
        ccut = reproject.postage_stamp(cmap, np.deg2rad(ra), np.deg2rad(dec),arcmin,pix)
        wcut = reproject.postage_stamp(wt, np.deg2rad(ra), np.deg2rad(dec),arcmin,pix)
        weight = wcut.mean()

        # if i==0:
        #     modrmap = np.rad2deg(ycut.modrmap())*60.
        #     bin_edges = np.arange(0.,15.,1.0)
        #     binner = stats.bin2D(modrmap,bin_edges)

        # cents,y1d = binner.bin(ycut)
        # cents,c1d = binner.bin(ccut)
        # s.add_to_stats("c1d",c1d*1e6)
        # s.add_to_stats("y1d",y1d*1e6)
        s.add_to_stack("cstack",ccut*1e6*weight)
        s.add_to_stack("ystack",ycut*1e6*weight)
        s.add_to_stats("sum",(weight,))
        i = i + 1
        if i%10==0 and rank==0: print(i)
    print("Rank %d done " % rank)
    s.get_stats()
    s.get_stacks()
    if rank==0:
        N = s.vectors['sum'].sum()
        ystack = s.stacks['ystack'] * N
        cstack = s.stacks['cstack'] * N

        _,nwcs = enmap.geometry(pos=(0,0),shape=ystack.shape,res=np.deg2rad(0.5/60.))

        return rank,enmap.enmap(ystack,nwcs),enmap.enmap(cstack,nwcs),N
    else:
        return rank,None,None,None


hplot = lambda x,y: io.hplot(x,os.environ['WORK']+"/"+y,ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,quantile=1e-3)


print("Starting deep56")
rank,ystack1,cstack1,i1 = do(ymap1,cmap1,mask1,ras1,decs1,wt1)
if rank == 0:
    print(i1)
    hplot(ystack1,"fig_all_qso_ystack_%s_%s"  % (cversion,'deep56'))
    hplot(cstack1,"fig_all_qso_cstack_%s_%s"  % (cversion,'deep56'))


print("Starting boss")
rank,ystack2,cstack2,i2 = do(ymap2,cmap2,mask2,ras2,decs2,wt2)
if rank == 0:
    print(i2)
    hplot(ystack2,"fig_all_qso_ystack_%s_%s"  % (cversion,'boss'))
    hplot(cstack2,"fig_all_qso_cstack_%s_%s"  % (cversion,'boss'))


    ystack = (ystack1+ystack2)/(i1+i2)
    cstack = (cstack1+cstack2)/(i1+i2)

    hplot(enmap.smooth_gauss( ystack,np.deg2rad(2/60.)),"fig_all_qso_ystack_%s_%s"  % (cversion,'both'))
    hplot(enmap.smooth_gauss( cstack,np.deg2rad(2/60.)),"fig_all_qso_cstack_%s_%s"  % (cversion,'both'))

