from __future__ import print_function
from orphics import maps,io,cosmology,catalogs
from pixell import enmap,reproject
import numpy as np
import os,sys,shutil
from soapack import interfaces as sints
import healpy as hp

version = "map_v1.1.1_joint"
cversion = "v1.1.1"

down = 6
    
# nilc = hp.read_alm("/scratch/r/rbond/msyriac/data/planck/data/pr2/COM_CompMap_Compton-SZMap-nilc-ymaps_2048_R2.00_alm.fits")

# annot = 'paper/all_planck_act.csv'
#annot = 'paper/all_planck_clusters.csv'
#annot = 'paper/public_clusters.csv'
#annot = None

t = {'deep56': 2, 'boss':4}
sels = {'deep56':np.s_[...,220:-220,300:-300] , 'boss':np.s_[...,450:-450,500:-500]}

for region in ['boss','deep56']:
    yname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_single_tile_%s_comptony_%s.fits" % (version,region,region,version)
    ybname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_single_tile_%s_comptony_%s_beam.txt" % (version,region,region,version)

    cname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_single_tile_%s_cmb_deprojects_comptony_%s.fits" % (version,region,region,version)
    cbname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_single_tile_%s_cmb_deprojects_comptony_%s_beam.txt" % (version,region,region,version)

    sname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_single_tile_%s_cmb_%s.fits" % (version,region,region,version)
    sbname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_single_tile_%s_cmb_%s_beam.txt" % (version,region,region,version)

    mname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_mask.fits" % (version,region)

    # shutil.copy(yname,"/scratch/r/rbond/msyriac/data/for_sigurd/")
    # shutil.copy(sname,"/scratch/r/rbond/msyriac/data/for_sigurd/")
    # shutil.copy(mname,"/scratch/r/rbond/msyriac/data/for_sigurd/")
    # continue

    mask = maps.binary_mask(enmap.read_map(mname))


    # Planck
    cols = catalogs.load_fits("/scratch/r/rbond/msyriac/data/planck/data/J_A+A_594_A27.fits",['RAdeg','DEdeg'])
    ras = cols['RAdeg']
    decs = cols['DEdeg']

    # ACT
    cols = catalogs.load_fits("paper/E-D56Clusters.fits",['RAdeg','DECdeg'])
    ras = np.append(ras,cols['RAdeg'])
    decs = np.append(decs,cols['DECdeg'])

    if region=='boss':
        radius = 10
        width = 2
        fontsize = 28
    elif region=='deep56':
        radius = 6
        width = 1
        fontsize = 16

        

    #annot = 'paper/temp_all_clusters.csv'
    annot = None
    # catalogs.convert_catalog_to_enplot_annotate_file(annot,ras,
    #                                                  decs,radius=radius,width=width,
    #                                                  color='red',mask=mask,threshold=0.99)



    # dm = sints.PlanckHybrid(region=mask)
    # pmap = dm.get_splits(season=None,patch=None,arrays=['545'],ncomp=1,srcfree=False)[0,0,0]

    ymap = enmap.read_map(yname)*mask
    smap = enmap.read_map(sname)*mask

    # nmap = reproject.enmap_from_healpix(nilc, mask.shape, mask.wcs, ncomp=1, unit=1, lmax=0,
    #                    rot="gal,equ", first=0, is_alm=True, return_alm=False, f_ell=None)

    # io.hplot(nmap[sels[region]],'fig_full_nmap_%s' % region,color='gray',grid=True,colorbar=True,
    #          annotate=annot,min=-1.25e-5,max=3.0e-5,ticks=t[region],mask=0,downgrade=down,mask_tol=1e-14)


    # io.hplot(pmap[sels[region]],'fig_full_pmap_%s' % region,color='planck',grid=True,colorbar=True,
    #          ticks=t[region],downgrade=down)
    # io.hplot(ymap[sels[region]],'fig_full_ymap_%s' % region,color='gray',grid=True,colorbar=True,
    #          annotate=annot,min=-1.25e-5,max=3.0e-5,ticks=t[region],mask=0,downgrade=down,mask_tol=1e-14,font_size=fontsize)
    io.hplot(ymap[sels[region]],'fig_full_ymap_%s' % region,color='gray',grid=True,colorbar=True,
             annotate=annot,min=-0.7e-5,max=2.0e-5,ticks=t[region],mask=0,downgrade=down,mask_tol=1e-14,font_size=fontsize)
    io.hplot(smap[sels[region]],'fig_full_smap_%s' % region,color='planck',grid=True,colorbar=True,
             range=300,ticks=t[region],mask=0,downgrade=down,mask_tol=1e-14,font_size=fontsize)

