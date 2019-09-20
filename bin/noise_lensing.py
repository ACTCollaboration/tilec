from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,utils
import numpy as np
import os,sys
from enlib import bench

def get_ilc_noise(ells):
    cents,sn1d,cn1d,scn1d,_,_ = np.loadtxt("lnoises.txt",unpack=True)
    return maps.interp(cents,sn1d)(ells),maps.interp(cents,cn1d)(ells),maps.interp(cents,scn1d)(ells)

def get_pol_noise(ells):
    cents,sn1d,cn1d,scn1d,enoise,bnoise = np.loadtxt("lnoises.txt",unpack=True)
    return maps.interp(cents,enoise)(ells),maps.interp(cents,bnoise)(ells)

def get_cmb(ells):
    theory = cosmology.default_theory()
    lcltt = theory.lCl('TT',ells)
    lclee = theory.lCl('EE',ells)
    lclte = theory.lCl('TE',ells)
    lclbb = theory.lCl('BB',ells)
    clkk = theory.gCl('kk',ells)
    return lcltt,lclee,lclte,lclbb,clkk


tellmin = 500
tellmax = 3000
pellmin = 500
pellmax = 3000

"""
1. Planck TT only
2. ACT TT only
3. Planck-ACT TT grad clean
4. Planck-ACT TT ILC grad clean
1. Planck TT+pol
2. ACT TT+pol
3. Planck-ACT TT grad clean + pol
4. Planck-ACT TT ILC grad clean + pol
"""


deg = 5.
px = 2.0
shape,wcs = maps.rect_geometry(width_deg=deg,px_res_arcmin=px)
modlmap = enmap.modlmap(shape,wcs)
tmask = maps.mask_kspace(shape,wcs,lmin=tellmin,lmax=tellmax,lxcut=20,lycut=20)
pmask = maps.mask_kspace(shape,wcs,lmin=pellmin,lmax=pellmax,lxcut=20,lycut=20)

tsnoise,tcnoise,tscnoise = get_ilc_noise(modlmap)
tenoise,tbnoise = get_pol_noise(modlmap)
lcltt,lclee,lclte,lclbb,clkk = get_cmb(modlmap)


def get_mv(nls):
    ninv = utils.eigpow(nls, -1.,axes=[0,1])
    ncoadd = 1./ninv.sum(axis=(0,1))
    return ncoadd
    

def get_feed(lcltt,lclte,lclee,lclbb,tnoise,enoise,bnoise):
    return {
        'uC_T_T': lcltt,
        'uC_T_E': lclte,
        'uC_E_E': lclee,
        'uC_B_B': lclbb,
        'tC_T_T': lcltt + tnoise,
        'tC_T_E': lclte,
        'tC_E_E': lclee + enoise,
        'tC_B_B': lclbb + bnoise,
        'tC_T_B': lclbb*0.,
        'tC_E_B': lclbb*0.,
        'tC_T_B': lclbb*0.,
        'tC_B_E': lclbb*0.,
        'tC_B_T': lclbb*0.,
        }

def get_feed_cross(lcltt,lclte,lclee,lclbb,stnoise,ctnoise,xnoise,enoise,bnoise):
    return {
        'uC_T_T': lcltt,
        'uC_T_E': lclte,
        'uC_E_E': lclee,
        'uC_B_B': lclbb,
        'tC_S_T_S_T': lcltt + stnoise,
        'tC_C_T_C_T': lcltt + ctnoise,
        'tC_S_T_C_T': lcltt + xnoise,
        'tC_C_T_S_T': lcltt + xnoise,
        'tC_S_T_S_E': lclte,
        'tC_C_T_S_E': lclte,
        'tC_S_E_S_E': lclee + enoise,
        'tC_S_B_S_B': lclbb + bnoise,
        'tC_S_T_S_B': lclbb*0.,
        'tC_S_E_S_B': lclbb*0.,
        'tC_S_T_S_B': lclbb*0.,
        'tC_S_B_S_E': lclbb*0.,
        'tC_S_B_S_T': lclbb*0.,
        'tC_C_T_S_B': lclbb*0.,
        'tC_C_T_S_B': lclbb*0.,
        'tC_S_B_C_T': lclbb*0.,
        }

def get_nlkk_single(estimator,modlmap,tnoise,enoise,bnoise,tmask,pmask,pols=['TT','TE','EE','EB','TB']):
    import symlens as s
    feed_dict = get_feed(lcltt,lclte,lclee,lclbb,tnoise,enoise,bnoise)
    alpha_estimator = estimator
    beta_estimator = estimator
    npols = len(pols)
    masks = {'T': tmask,'E':pmask,'B':pmask}
    bin_edges = np.arange(40,2500,40)
    binner = stats.bin2D(modlmap,bin_edges)
    cents = binner.centers
    nls = np.zeros((npols,npols,cents.size))
    Als = []
    for i in range(npols):
        a,b = pols[i]
        Als.append(s.A_l(shape,wcs,feed_dict,alpha_estimator,pols[i],xmask=masks[a],ymask=masks[b]))
    for i in range(npols):
        for j in range(i,npols):
            print(pols[i],'x',pols[j])
            alpha_XY = pols[i]
            beta_XY = pols[j]
            a,b = alpha_XY
            c,d = beta_XY
            if i==j:
                xmask = masks[a]
                ymask = masks[b]
            else:
                xmask = masks['T']
                ymask = masks['T']
            nl = s.N_l_cross(shape, wcs, feed_dict, alpha_estimator, alpha_XY, beta_estimator, beta_XY, xmask=xmask, ymask=ymask, Aalpha=Als[i], Abeta=Als[j], field_names_alpha=None, field_names_beta=None)
            cents,nl1d = binner.bin(nl)
            nls[i,j] = nl1d.copy()
            if i!=j: nls[j,i] = nl1d.copy()
    ncoadd = get_mv(nls)
    return cents,nls,ncoadd

def get_nlkk_mixed(modlmap,stnoise,ctnoise,xnoise,enoise,bnoise,tmask,pmask,ipols):
    estimator = "hdv"
    import symlens as s
    feed_dict = get_feed_cross(lcltt,lclte,lclee,lclbb,stnoise,ctnoise,xnoise,enoise,bnoise)
    alpha_estimator = estimator
    beta_estimator = estimator

    pols = []
    for pol in ipols:
        if pol=='TT':
            pols.append('S_T_C_T')
            pols.append('C_T_S_T')
        else:
            x,y = pol
            pols.append('S_%s_S_%s' % (x,y))

    def get_xy(ipol):
        sp = ipol.split('_')
        return ''.join([sp[1],sp[3]])
        
    npols = len(pols)
    masks = {'T': tmask,'E':pmask,'B':pmask}
    bin_edges = np.arange(40,2500,40)
    binner = stats.bin2D(modlmap,bin_edges)
    cents = binner.centers
    nls = np.zeros((npols,npols,cents.size))
    for i in range(npols):
        for j in range(i,npols):
            print(pols[i],'x',pols[j])
            alpha_XY = get_xy(pols[i])
            beta_XY = get_xy(pols[j])
            fa,a,fb,b = pols[i].split('_')
            fc,c,fd,d = pols[j].split('_')
            if i==j:
                xmask = masks[a]
                ymask = masks[b]
            else:
                xmask = masks['T']
                ymask = masks['T']
            fnames1 = [fa,fb]
            fnames2 = [fc,fd]
            nl = s.N_l_cross(shape, wcs, feed_dict, estimator, alpha_XY, estimator, beta_XY, xmask=xmask, ymask=ymask, field_names_alpha=fnames1, field_names_beta=fnames2)
            cents,nl1d = binner.bin(nl)
            nls[i,j] = nl1d.copy()
            if i!=j: nls[j,i] = nl1d.copy()
    ncoadd = get_mv(nls)
    return cents,nls,ncoadd,pols

def plot(cents,nls,ncoadd,pols,tag="default"):
    npols = len(pols)
    pl = io.Plotter(xyscale='linlog',ylabel='$C_L$',xlabel='$L$')
    for i in range(npols):
        for j in range(i,npols):
            nl1d = nls[i,j]
            if i!=j:
                pl.add(cents,np.abs(nl1d),ls="--",alpha=0.2)#,label=pols[i]+'x'+pols[j]
            else:
                pl.add(cents,nl1d,alpha=0.6)#,label=pols[i])
    
    ells = np.arange(0,2500,1)
    theory = cosmology.default_theory()
    clkk = theory.gCl('kk',ells)
    pl.add(ells,clkk,color='k',lw=3)
    pl.legend(loc='upper right')
    pl._ax.set_ylim(1e-10,1e-4)
    pl.add(cents,ncoadd,color='red',lw=3)#,label='MV')
    pl.done("nlkk_%s.png"%tag)
    # pl = io.Plotter(xyscale='linlin')
    # pl.add(cents,nls[0,3],label="TT x EB")
    # pl.hline(y=0)
    # pl.done("nltteb_%s.png"%tag)
    io.save_cols("lensing_noise_%s.txt" % tag,(cents,ncoadd))




# pols = ['TT','TE','EE','EB','TB']
# with bench.show("huok"):
#     cents,nls,ncoadd = get_nlkk_single("hu_ok",modlmap,asnoise-lcltt,apnoise,apnoise,tmask,pmask,pols=pols)
# plot(cents,nls,ncoadd,pols,tag="act")

# pols = ['TT','TE','ET','EE','EB','TB']
# with bench.show("hdv"):
#     cents,nls,ncoadd = get_nlkk_single("hdv",modlmap,onoise,apnoise,apnoise,tmask,pmask,pols=pols)
# plot(cents,nls,ncoadd,pols,tag="act_only_all")

# pols = ['TT','TE','ET','EE','EB','TB']
# with bench.show("hdv cross-check"):
#     cents,nls,ncoadd,pols = get_nlkk_mixed(modlmap,pnoise,onoise,asnoise*0.,apnoise,apnoise,tmask,pmask,pols)
# plot(cents,nls,ncoadd,pols,tag="act_smica_noilc_all")

pols = ['TT','TE','ET','EE','EB','TB']
with bench.show("hdv cross-check"):
    cents,nls,ncoadd,pols = get_nlkk_mixed(modlmap,tcnoise-lcltt,tsnoise-lcltt,tscnoise-lcltt,tenoise-lclee,tbnoise-lclbb,tmask,pmask,pols)
plot(cents,nls,ncoadd,pols,tag="act_planck_ilc_all")


# pols = ['TT']
# with bench.show("hdv"):
#     cents,nls,ncoadd = get_nlkk_single("hdv",modlmap,onoise,apnoise,apnoise,tmask,pmask,pols=pols)
# plot(cents,nls,ncoadd,pols,tag="act_only_tt")

# pols = ['TT']
# with bench.show("hdv cross-check"):
#     cents,nls,ncoadd,pols = get_nlkk_mixed(modlmap,pnoise,onoise,asnoise*0.,apnoise,apnoise,tmask,pmask,pols)
# plot(cents,nls,ncoadd,pols,tag="act_smica_noilc_tt")

# pols = ['TT']
# with bench.show("hdv cross-check"):
#     cents,nls,ncoadd,pols = get_nlkk_mixed(modlmap,acnoise-lcltt,asnoise-lcltt,ascnoise-lcltt,apnoise,apnoise,tmask,pmask,pols)
# plot(cents,nls,ncoadd,pols,tag="act_planck_ilc_tt")


# pols = ['TT','TE','ET','EE','EB','TB']a
# cents,nls,ncoadd = get_nlkk_single("hdv",modlmap,pnoise,pnoise*2.,pnoise*2.,ptmask,ppmask,pols=pols)
# plot(cents,nls,ncoadd,pols,tag="planck")
