from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils,covtools

region = "boss"



opath = "/scratch/r/rbond/msyriac/data/act/omar/"
tpath = os.environ['WORK'] + "/data/depot/tilec/map_v1.0.0_rc_joint_%s/" % region

"""
We compare masks
and make sure they are identical
"""

omask = enmap.read_map("%smask_s14&15_%s.fits" % (opath,region))
tmask = enmap.read_map("%stilec_mask.fits" % tpath)
assert np.all(np.isclose(omask,tmask))

modlmap = omask.modlmap()
bin_edges = np.arange(400,8000,200)
binner = stats.bin2D(modlmap,bin_edges)
Nplot = 300
kbeam = tutils.get_kbeam("d56_05",modlmap,sanitize=False)
w2 = np.mean(omask**2.)
ls,bells = np.loadtxt("%stilec_single_tile_%s_cmb_map_v1.0.0_rc_joint_beam.txt" % (tpath,region),unpack=True)
tkbeam = maps.interp(ls,bells)(modlmap)
ls,bells = np.loadtxt("%stilec_single_tile_%s_cmb_deprojects_comptony_map_v1.0.0_rc_joint_beam.txt" % (tpath,region),unpack=True)
tkbeam_nosz = maps.interp(ls,bells)(modlmap)

def pow(x):
    k = enmap.fft(x,normalize='phys')
    return (k*k.conj()).real/w2/kbeam**2

def pow2(x,y,xbeam,ybeam):
    k = enmap.fft(x,normalize='phys')
    k2 = enmap.fft(y,normalize='phys')
    return (k*k2.conj()).real/w2/xbeam/ybeam


"""
We plot 2D Q/U power

We compare PS
(1) k-space I total power vs ILC power
(2) k-space Q/U total power vs Q/U total power of deep6
"""

# ti_noise2d = enmap.read_map("%stilec_single_tile_%s_cmb_map_v1.0.0_rc_joint_noise.fits" % (tpath,region))
# ti_nosz_noise2d = enmap.read_map("%stilec_single_tile_%s_cmb_deprojects_comptony_map_v1.0.0_rc_joint_noise.fits" % (tpath,region))
# ti_cross_noise2d = enmap.read_map("%stilec_single_tile_%s_cmb_deprojects_comptony_map_v1.0.0_rc_joint_cross_noise.fits" % (tpath,region))

ti_map = enmap.read_map("%stilec_single_tile_%s_cmb_map_v1.0.0_rc_joint.fits" % (tpath,region))
ti_nosz_map = enmap.read_map("%stilec_single_tile_%s_cmb_deprojects_comptony_map_v1.0.0_rc_joint.fits" % (tpath,region))
ti_noise2d = pow2(ti_map,ti_map,tkbeam,tkbeam)
ti_nosz_noise2d = pow2(ti_nosz_map,ti_nosz_map,tkbeam_nosz,tkbeam_nosz)
ti_cross_noise2d = pow2(ti_map,ti_nosz_map,tkbeam,tkbeam_nosz)


io.power_crop(ti_noise2d,Nplot,"ti_noise2d.png",ftrans=True)
io.power_crop(ti_nosz_noise2d,Nplot,"ti_nosz_noise2d.png",ftrans=True)
io.power_crop(ti_cross_noise2d,Nplot,"ti_cross_noise2d.png",ftrans=True)


tmap = enmap.read_map("%sdataCoadd_combined_I_s14&15_%s.fits" % (opath,region))
oi_noise2d = pow(tmap)

io.power_crop(oi_noise2d,Nplot,"oi_noise2d.png",ftrans=True)


cents,ti_noise1d = binner.bin(ti_noise2d)
cents,ti_nosz_noise1d = binner.bin(ti_nosz_noise2d)
cents,ti_cross_noise1d = binner.bin(ti_cross_noise2d)
cents,oi_noise1d = binner.bin(oi_noise2d)


pl = io.Plotter(xlabel='l',ylabel='C',xyscale='linlog')
pl.add(cents,ti_noise1d,label='tilec')
pl.add(cents,oi_noise1d,label='kspace')
pl.add(cents,ti_nosz_noise1d,label='tilec deproj',ls="--")
pl.add(cents,ti_cross_noise1d,label='tilec deproj cross',ls=":")
pl.done("powcomp.png")

pl = io.Plotter(xlabel='l',ylabel='R',xyscale='linlin')
pl.add(cents,oi_noise1d/ti_noise1d)
pl.hline(y=1)
pl.done("powcomp_rel.png")


"""
Q/U

"""

def get_pol_powers(tmap,qmap,umap,kbeam,mask_w2):
    f = lambda x: enmap.fft(x,normalize='phys')
    p = lambda x,y: (x*y.conj()).real/mask_w2/kbeam**2
    kt = f(tmap)
    kq = -f(qmap)
    ku = f(umap)
    tt = p(kt,kt)
    tu = p(kt,ku)
    tq = p(kt,kq)
    qq = p(kq,kq)
    uu = p(ku,ku)
    qu = p(kq,ku)
    p2d = np.array([[tt,tq,tu],
                    [tq,qq,qu],
                    [tu,qu,uu]])
    p2d[~np.isfinite(p2d)] = 0

    rp2d = enmap.enmap(maps.rotate_pol_power(tmap.shape,tmap.wcs,p2d,iau=False),tmap.wcs)
    return rp2d[0,0],rp2d[1,1],rp2d[2,2]

qmap = enmap.read_map("%sdataCoadd_combined_Q_s14&15_%s.fits" % (opath,region))
umap = enmap.read_map("%sdataCoadd_combined_U_s14&15_%s.fits" % (opath,region))

#oq_noise2d = pow(qmap)
#ou_noise2d = pow(umap)

_,oe_p2d,ob_p2d = get_pol_powers(tmap,qmap,umap,kbeam,w2)

io.power_crop(oe_p2d,Nplot,"oe_noise2d.png",ftrans=True)
    

#io.power_crop(ob_p2d,Nplot,"ob_noise2d.png",ftrans=True)


dmask = sints.get_act_mr3_crosslinked_mask(region='deep6')
dw2 = np.mean(dmask**2)
dm = sints.ACTmr3(calibrated=True,region=dmask)
imap = dm.get_coadd(season='s13',array='pa1_f150',patch='deep6',srcfree=True)
tmap = imap[0]*dmask
qmap = imap[1]*dmask
umap = imap[2]*dmask
dbeam = tutils.get_kbeam("d6",dmask.modlmap(),sanitize=False)
_,de_p2d,db_p2d = get_pol_powers(tmap,qmap,umap,dbeam,dw2)

io.power_crop(de_p2d,Nplot,"de_noise2d.png",ftrans=True)


dbinner = stats.bin2D(dmask.modlmap(),bin_edges)
cents,de_noise1d = dbinner.bin(de_p2d)
cents,db_noise1d = dbinner.bin(db_p2d)
cents,oe_noise1d = binner.bin(oe_p2d)
cents,ob_noise1d = binner.bin(ob_p2d)

#cents,smoothed_oe_noise1d = binner.bin(smoothed_oe_p2d)

pl = io.Plotter(xlabel='l',ylabel='C',xyscale='linlog')
pl.add(cents,de_noise1d,label='d6 E')
#pl.add(cents,db_noise1d,label='d6 B')
pl.add(cents,oe_noise1d,label='kspace E')
#pl.add(cents,smoothed_oe_noise1d,label='smoothed kspace E')
#pl.add(cents,ob_noise1d,label='kspace B')
pl.done("powcomp_pol.png")





"""
We make lensing noise curves from 1d noise curves of :
(a) ILC A+P tSZ deproj symmetrized combination
(b) k-space Q/U
"""


txnoise = ti_noise1d
tynoise = ti_nosz_noise1d
tcrossnoise = ti_cross_noise1d

enoise = oe_noise1d
bnoise = ob_noise1d

io.save_cols("lnoises.txt",(cents,txnoise,tynoise,tcrossnoise,enoise,bnoise))
