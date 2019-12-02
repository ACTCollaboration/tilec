from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from tilec import fg as tfg,utils as tutils
from soapack import interfaces as sints

qids = ['d56_0%d' % x for x in range(1,7)] + ['boss_0%d' % x for x in range(1,5)] 
print(qids)

bps = []
lbeams = []
cfreqs = []
lexps1 = []
lexps2 = []
exp1 = {148: -1, 93: -1}
exp2 = {148: -0.8, 93: -0.9}
aspecs = tutils.ASpecs().get_specs
ells = np.arange(0,8000,1)

for qid in qids:
    dm = sints.models[sints.arrays(qid,'data_model')]()
    lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)
    cfreqs.append(cfreq)
    season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
    array = '_'.join([array1,array2])
    bps.append("data/"+dm.get_bandpass_file_name(array))

    lbeams.append(tutils.get_kbeam(qid,ells,sanitize=False))
    lexps1.append(exp1[cfreq])
    lexps2.append(exp2[cfreq])

r1 = tfg.get_mix_bandpassed(bps, 'tSZ', 
                            ccor_cen_nus=cfreqs, ccor_beams=lbeams, ccor_exps = lexps1)


r2 = tfg.get_mix_bandpassed(bps, 'tSZ', 
                            ccor_cen_nus=cfreqs, ccor_beams=lbeams, ccor_exps = lexps2)


pl = io.Plotter(xlabel='$\\ell$',ylabel='$\\Delta R/R$')
for i,qid in enumerate(qids):
    season,array1,array2,region = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq'),sints.arrays(qid,'region')
    array = '_'.join([array1,array2])
    pl.add(ells,(r2[i]-r1[i])/r1[i],label="%s %s %s" % (region,season,array))
pl.hline(y=0)
pl.done("rdiff.png")
