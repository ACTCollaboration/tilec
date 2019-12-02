from __future__ import print_function
from orphics import maps,io,cosmology,catalogs,stats,mpi
from pixell import enmap,reproject
import numpy as np
import os,sys
from soapack import interfaces as sints
import tilec.fg as tfg
import tilec.utils as tutils
import random

np.random.seed(100)
cversion = 'joint'

region1 = 'deep56'
region2 = 'boss'


# cols = catalogs.load_fits("AdvACT.fits",['RAdeg','DECdeg','SNR'])
# ras = cols['RAdeg']
# decs = cols['DECdeg']
# sns = cols['SNR']
# iras = ras[sns>5]
# idecs = decs[sns>5]

# fname = os.environ['WORK'] + "/data/boss/boss_dr12/galaxy_DR12v5_CMASS_South.fits"
# cols = catalogs.load_fits(fname,['RA','DEC'])
# iras1 = cols['RA']
# idecs1 = cols['DEC']

# fname = os.environ['WORK'] + "/data/boss/boss_dr12/galaxy_DR12v5_CMASS_North.fits"
# cols = catalogs.load_fits(fname,['RA','DEC'])
# iras2 = cols['RA']
# idecs2 = cols['DEC']

# iras = np.append(iras1,iras2)
# idecs = np.append(idecs1,idecs2)



fname = os.environ['WORK'] + "/data/boss/sdss_dr8/redmapper_dr8_public_v6.3_catalog.fits"
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

tdir = '/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919'

solution = 'tsz'


yfile1 = tutils.get_generic_fname(tdir,region1,solution,None,cversion)
cfile1 = tutils.get_generic_fname(tdir,region1,solution,'cib',cversion)
dfile1 = tutils.get_generic_fname(tdir,region1,solution,'cmb',cversion)
ybfile1 = tutils.get_generic_fname(tdir,region1,solution,None,cversion,beam=True)
cbfile1 = tutils.get_generic_fname(tdir,region1,solution,'cib',cversion,beam=True)


yfile2 = tutils.get_generic_fname(tdir,region2,solution,None,cversion)
cfile2 = tutils.get_generic_fname(tdir,region2,solution,'cib',cversion)
dfile2 = tutils.get_generic_fname(tdir,region2,solution,'cmb',cversion)
ybfile2 = tutils.get_generic_fname(tdir,region2,solution,None,cversion,beam=True)
cbfile2 = tutils.get_generic_fname(tdir,region2,solution,'cib',cversion,beam=True)

cmap1 = enmap.read_map(cfile1)
cmap2 = enmap.read_map(cfile2)
dmap1 = enmap.read_map(dfile1)
dmap2 = enmap.read_map(dfile2)
modlmap1 = cmap1.modlmap()
modlmap2 = cmap2.modlmap()


lsy1,by1 = np.loadtxt(ybfile1,unpack=True)
by12d = maps.interp(lsy1,by1)(modlmap1)
lsc1,bc1 = np.loadtxt(cbfile1,unpack=True)
bc12d = maps.interp(lsc1,bc1)(modlmap1)
beam_ratio1 = bc12d/by12d
beam_ratio1[~np.isfinite(beam_ratio1)] = 0


lsy2,by2 = np.loadtxt(ybfile2,unpack=True)
by22d = maps.interp(lsy2,by2)(modlmap2)
# ls,bc2 = np.loadtxt(cbfile2,unpack=True)
bc22d = maps.interp(lsc1,bc1)(modlmap2)
beam_ratio2 = bc22d/by22d
beam_ratio2[~np.isfinite(beam_ratio2)] = 0


ymap1 = maps.filter_map(enmap.read_map(yfile1),beam_ratio1)
ymap2 = maps.filter_map(enmap.read_map(yfile2),beam_ratio2)


arcmin = 40.
pix = 0.5


def get_cuts(mask,ymap,cmap,dmap,wtmap,ra,dec,arcmin,pix):
    mcut = reproject.cutout(mask, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=int(arcmin/pix))
    if mcut is None: 
        return None,None,None,None,None
    if np.any(mcut)<=0: 
        return None,None,None,None,None

    ycut = reproject.cutout(ymap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=int(arcmin/pix))
    ccut = reproject.cutout(cmap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=int(arcmin/pix))
    dcut = reproject.cutout(dmap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=int(arcmin/pix))
    wcut = reproject.cutout(wtmap, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=int(arcmin/pix))
    weight = wcut.mean()
    return mcut,ycut,ccut,dcut,weight

def do(ymap,cmap,dmap,mask,ras,decs,wt):


    combined = list(zip(ras, decs))
    random.shuffle(combined)
    ras[:], decs[:] = zip(*combined)

    Nrand = 400

    njobs = len(ras)
    comm,rank,my_tasks = mpi.distribute(njobs)
    print("Rank %d starting" % rank)
    s = stats.Stats(comm)

    i = 0
    for task in my_tasks:

        ra = ras[task]
        dec = decs[task]

        mcut,ycut,ccut,dcut,weight = get_cuts(mask,ymap,cmap,dmap,wt,ra,dec,arcmin,pix)
        if mcut is None: continue

        if i==0:
            modrmap = np.rad2deg(ycut.modrmap())*60.
            bin_edges = np.arange(0.,15.,1.0)
            binner = stats.bin2D(modrmap,bin_edges)

        rras,rdecs = catalogs.random_catalog(ymap.shape,ymap.wcs,Nrand,edge_avoid_deg=4.)
        nrej = 0
        for rra,rdec in zip(rras,rdecs):
            rmcut,rycut,rccut,rdcut,rweight = get_cuts(mask,ymap,cmap,dmap,wt,rra,rdec,arcmin,pix)
            if rmcut is None: 
                nrej = nrej + 1
                continue
            cents,ry1d = binner.bin(rycut)
            cents,rc1d = binner.bin(rccut)
            cents,rd1d = binner.bin(rdcut)
            s.add_to_stats("rc1d",rc1d*1e6)
            s.add_to_stats("ry1d",ry1d*1e6)
            s.add_to_stats("rd1d",rd1d*1e6)
        if rank==0: print(Nrand-nrej, " accepted")
            


        cents,y1d = binner.bin(ycut)
        cents,c1d = binner.bin(ccut)
        cents,d1d = binner.bin(dcut)
        s.add_to_stats("c1d",c1d*1e6)
        s.add_to_stats("y1d",y1d*1e6)
        s.add_to_stats("d1d",d1d*1e6)
        s.add_to_stack("cstack",ccut*1e6*weight)
        s.add_to_stack("dstack",dcut*1e6*weight)
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
        dstack = s.stacks['dstack'] * N
        y1ds = s.vectors['y1d']
        c1ds = s.vectors['c1d']
        d1ds = s.vectors['d1d']
        ry1d = s.stats['ry1d']['mean']
        rc1d = s.stats['rc1d']['mean']
        rd1d = s.stats['rd1d']['mean']

        _,nwcs = enmap.geometry(pos=(0,0),shape=ystack.shape,res=np.deg2rad(0.5/60.))

        return rank,enmap.enmap(ystack,nwcs),enmap.enmap(cstack,nwcs),enmap.enmap(dstack,nwcs),N,cents,y1ds,c1ds,d1ds,ry1d,rc1d,rd1d
    else:
        return rank,None,None,None,None,None,None,None,None,None,None,None


hplot = lambda x,y: io.hplot(x,os.environ['WORK']+"/"+y,ticks=5,tick_unit='arcmin',grid=True,colorbar=True,color='gray',upgrade=4,quantile=1e-3)


print("Starting deep56")
rank,ystack1,cstack1,dstack1,i1,cents,y1ds1,c1ds1,d1ds1,ry1d1,rc1d1,rd1d1 = do(ymap1,cmap1,dmap1,mask1,ras1,decs1,wt1)
if rank == 0:
    print(i1)
    hplot(ystack1,"fig_all_cmass_ystack_%s_%s"  % (cversion,'deep56'))
    hplot(cstack1,"fig_all_cmass_cstack_%s_%s"  % (cversion,'deep56'))
    hplot(dstack1,"fig_all_cmass_dstack_%s_%s"  % (cversion,'deep56'))


print("Starting boss")
rank,ystack2,cstack2,dstack2,i2,cents,y1ds2,c1ds2,d1ds2,ry1d2,rc1d2,rd1d2 = do(ymap2,cmap2,dmap2,mask2,ras2,decs2,wt2)
if rank == 0:
    print(i2)
    hplot(ystack2,"fig_all_cmass_ystack_%s_%s"  % (cversion,'boss'))
    hplot(cstack2,"fig_all_cmass_cstack_%s_%s"  % (cversion,'boss'))
    hplot(dstack2,"fig_all_cmass_dstack_%s_%s"  % (cversion,'boss'))


    ystack = (ystack1+ystack2)/(i1+i2)
    cstack = (cstack1+cstack2)/(i1+i2)
    dstack = (dstack1+dstack2)/(i1+i2)

    hplot(ystack,"fig_all_cmass_ystack_%s_%s"  % (cversion,'both'))
    hplot(cstack,"fig_all_cmass_cstack_%s_%s"  % (cversion,'both'))
    hplot(dstack,"fig_all_cmass_dstack_%s_%s"  % (cversion,'both'))



    sy1 = stats.get_stats(y1ds1)
    sc1 = stats.get_stats(c1ds1)
    sd1 = stats.get_stats(d1ds1)

    sy2 = stats.get_stats(y1ds2)
    sc2 = stats.get_stats(c1ds2)
    sd2 = stats.get_stats(d1ds2)

    y1 = sy1['mean']
    ey1 = sy1['errmean']

    c1 = sc1['mean']
    ec1 = sc1['errmean']

    d1 = sd1['mean']
    ed1 = sd1['errmean']

    y2 = sy2['mean']
    ey2 = sy2['errmean']

    c2 = sc2['mean']
    ec2 = sc2['errmean']

    d2 = sd2['mean']
    ed2 = sd2['errmean']

    pl = io.Plotter(xlabel='$\\theta$ (arcmin)',ylabel='Filtered $Y  (\\times 10^6)$')
    pl.add_err(cents,y1-ry1d1,yerr=ey1,label="deep56",ls="none",marker="x",markersize=8,elinewidth=2,mew=2,color='C0')
    pl.add_err(cents,c1-rc1d1,yerr=ec1,label="deep56 no dust",ls="none",marker="o",markersize=8,elinewidth=2,mew=2,color='C0')
    pl.add_err(cents,d1-rd1d1,yerr=ed1,label="deep56 no cmb",ls="none",marker="_",markersize=8,elinewidth=2,mew=2,color='C0')
    pl.add_err(cents,y2-ry1d2,yerr=ey2,label="boss",ls="none",marker="x",markersize=8,elinewidth=2,mew=2,color='C1')
    pl.add_err(cents,c2-rc1d2,yerr=ec2,label="boss no dust",ls="none",marker="o",markersize=8,elinewidth=2,mew=2,color='C1')
    pl.add_err(cents,d2-rd1d2,yerr=ed2,label="boss no cmb",ls="none",marker="_",markersize=8,elinewidth=2,mew=2,color='C1')

    pl.add_err(cents+0.1,y1,yerr=ey1,ls="none",marker="x",markersize=8,elinewidth=2,mew=2,color='C0',alpha=0.2)
    pl.add_err(cents+0.1,c1,yerr=ec1,ls="none",marker="o",markersize=8,elinewidth=2,mew=2,color='C0',alpha=0.2)
    pl.add_err(cents+0.1,d1,yerr=ed1,ls="none",marker="_",markersize=8,elinewidth=2,mew=2,color='C0',alpha=0.2)
    pl.add_err(cents+0.1,y2,yerr=ey2,ls="none",marker="x",markersize=8,elinewidth=2,mew=2,color='C1',alpha=0.2)
    pl.add_err(cents+0.1,c2,yerr=ec2,ls="none",marker="_",markersize=8,elinewidth=2,mew=2,color='C1',alpha=0.2)
    pl.add_err(cents+0.1,d2,yerr=ed2,ls="none",marker="o",markersize=8,elinewidth=2,mew=2,color='C1',alpha=0.2)

    pl.hline(y=0)
    pl.done(os.environ['WORK']+"/"+'fig_boss_yprofile.png')

