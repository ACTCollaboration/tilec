import healpy as hp
from pixell import enmap, curvedsky
from orphics import maps,io
from soapack import interfaces as sints
from tilec.pipeline import JointSim 


qids = ['d56_03','p05']
jsim = JointSim(qids,None)
jsim.update_signal_index(0)
mask = sints.get_act_mr3_crosslinked_mask('deep6')
imap = jsim.compute_map(mask.shape,mask.wcs,'p05',include_cmb=True,include_tsz=True,include_fgres=True)
io.hplot(imap,'planck')
imap = jsim.compute_map(mask.shape,mask.wcs,'d56_03',include_cmb=True,include_tsz=True,include_fgres=True)
io.hplot(imap,'act')

