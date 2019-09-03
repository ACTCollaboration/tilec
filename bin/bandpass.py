from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import fg as tfg, utils as tutils
from enlib import bench
from scipy.interpolate import interp2d

qids = ['d56_0%d' % x for x in range(1,7)]
#qids = qids + ['p0%d' % x for x in range(1,9)]

aspecs = tutils.ASpecs().get_specs

# fs = {'p01': [x-2 for x in [26,28,30,32,34]] ,
#       'p02': [40,42,44,46,48] ,
#       'p03': [70+x*1.2 for x in range(-10,12,4)] ,
#       'p04': [100+x*2 for x in range(-4,6,2)]  ,
#       'p05': [143+x*5 for x in range(-4,6,2)] ,
#       'p06': [217+x*5 for x in range(-4,6,2)] ,
#       'p07': [353+x for x in range(-40,60,20)] ,
#       'p08': [545+x*1.5 for x in range(-40,60,20)] ,
#       'p09': [857+x*2 for x in range(-40,60,20)] ,
#       'd56_01': [148+x*1.5 for x in range(-10,12,4)]  ,
#       'd56_02': [148+x*1.5 for x in range(-10,12,4)],
#       'd56_05': [93+x for x in range(-10,12,4)],
#       'd56_06': [148+x for x in range(-10,12,4)]}


# for qid in fs.keys():
#     dm = sints.models[sints.arrays(qid,'data_model')]()
#     if dm.name=='act_mr3':
#         season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
#         array = '_'.join([array1,array2])
#     elif dm.name=='planck_hybrid':
#         season,patch,array = None,None,sints.arrays(qid,'freq')
#     lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)
#     bpf = "data/"+dm.get_bandpass_file_name(array)
#     nu_ghz, trans = np.loadtxt(bpf, usecols=(0,1), unpack=True)

#     trans = trans/trans.max()
#     sel = trans>1e-3
#     nu_ghz = nu_ghz[sel]
#     trans = trans[sel]


#     pl = io.Plotter(xlabel='nu',ylabel='t',xyscale='loglin')
#     pl.add(nu_ghz,trans)
#     for f in fs[qid]:
#         pl.vline(x=f)
#     pl.done("bpass_%s.png" % qid)
    
# sys.exit()


comp = 'tSZ'

ells = np.arange(0,30000,1)

gauss_beam = False
gauss_band = False #False

for qid in qids:
    dm = sints.models[sints.arrays(qid,'data_model')]()
    if dm.name=='act_mr3':
        season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
        array = '_'.join([array1,array2])
    elif dm.name=='planck_hybrid':
        season,patch,array = None,None,sints.arrays(qid,'freq')
    lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)
    bpf = "data/"+dm.get_bandpass_file_name(array)
    
    nu_ghz, trans = np.loadtxt(bpf, usecols=(0,1), unpack=True)

    trans = trans/trans.max()
    sel = trans>1e-4
    nu_ghz = nu_ghz[sel]
    trans = trans[sel]


    if gauss_band:
        pl = io.Plotter(xlabel='nu',ylabel='band')
        pl.add(nu_ghz,trans/trans.max())
        bcent = cfreq
        bsig = 0.15 * bcent
        trans = np.exp(-(nu_ghz-bcent)**2./2./bsig**2.)
        pl.add(nu_ghz,trans/trans.max())
        pl.done("band_%s.png" % qid)
    

    

    if gauss_beam:
        if 'p' in qid:
            fwhm = dm.fwhms[array]
        else:
            fwhm = 1.4 * (150./cfreq)
        lbeam = maps.gauss_beam(ells,fwhm)
    else:
        lbeam = tutils.get_kbeam(qid,ells,sanitize=False,planck_pixwin=False) 

    with bench.show("beamint"):
        fbnus = maps.interp(ells,lbeam[None,:],fill_value=(lbeam[0],lbeam[-1]))
        bnus = fbnus((cfreq/nu_ghz)*ells[:,None])[0].swapaxes(0,1)
        bnus = bnus / bnus[:,:1]

        pl = io.Plotter(xlabel='l',ylabel='b')
        for i in range(bnus.shape[0]):

            if trans[i]>1e-1: pl.add(ells,bnus[i]/lbeam)
        pl.hline(y=1)
        pl._ax.set_ylim(0.8,1.2)
        pl.done("abeams_%s.png" % qid)
            
        


    

    mixc = tfg.get_mix(cfreq, comp)
    mix = tfg.get_mix(nu_ghz, comp)
    mix2 = np.trapz(trans * tfg.dBnudT(nu_ghz) * mix, nu_ghz) / np.trapz(trans * tfg.dBnudT(nu_ghz), nu_ghz)
    mix3 = np.trapz(trans * tfg.dBnudT(nu_ghz) * bnus.swapaxes(0,1) * mix, nu_ghz) / np.trapz(trans * tfg.dBnudT(nu_ghz), nu_ghz) / lbeam
    print(qid,mixc,mix2,mix3[300])

    if qid in ['p01','p02']:
        lmax = 300
    elif 'p' in qid:
        lmax = 3000
    else:
        lmax = 10000
    
    pl = io.Plotter(xyscale='linlin',xlabel='l',ylabel='r')
    pl.add(ells[ells<lmax],(mixc/mix2+0*ells)[ells<lmax],label='cfreq approx')
    pl.add(ells[ells<lmax],(mix3/mix2)[ells<lmax],label='color corrected')
    pl.hline(y=1)
    pl.done("modbeam_%s.png" % qid)
