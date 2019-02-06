from orphics import maps,io,cosmology,stats
from pixell import enmap,wcsutils
import numpy as np
import os,sys

# Get the global config
gconfig = io.config_from_yaml("input/data.yml")

# Get the path config for this system
paths = io.config_from_yaml("input/paths.yml")

# Get the region name -> mask file names mapping that defines analysis footprint
masks = gconfig['masks']

# Prepend the mask path to the mask file names
for key in masks:
    masks[key] = paths[gconfig['mask_path_name']] + "/" + masks[key]

class DataModel(object):
    def __init__(self,region,config):
        """
        config should be a dictionary with following specified:
        1. region: string containing name of region that maps the `masks` dictionary to a filename
        """
        self.c = config
        self.shape,self.wcs = enmap.read_map_geometry(masks[region])
        self.maproot = paths[self.c['map_path_name']]

    def _read_map(self,fname,ncomp=None,**kwargs):
        ishape,iwcs = enmap.read_map_geometry(self.maproot+"/"+fname)
        assert wcsutils.is_compatible(self.wcs, iwcs)
        if wcsutils.equal(self.wcs, iwcs):
            pixbox = None
        else:
            pixbox = enmap.get_pixbox(iwcs,self.shape,self.wcs)
        if ncomp is None: sel = None
        else: sel = np.s_[:ncomp,:,:]
        return enmap.read_map(self.maproot+"/"+fname,pixbox=pixbox,sel=sel,**kwargs)


class ACTmr3(DataModel):
    def __init__(self,region,config):
        DataModel.__init__(self,region,config)
        season,array,freq = self.c['id'].split('_')
        self.saf = '_'.join([season,array,freq])
        self.patch = self.c['region']
        pref = '_'.join([season,self.c['region'],array,freq])
        self._ftag = "%s_nohwp_night_3pass_4way" % (pref)
        self.beamroot = paths[self.c['beam_path_name']]
        self.nsplits = 4
        
    def get_coadd(self,ncomp=3,srcfree=True):
        return self._read_map(self._ftag+"_coadd_map%s.fits" % ("_srcfree" if srcfree else ""),ncomp=ncomp)
    
    def get_split(self,k,ncomp=3,srcfree=True):
        return self._read_map(self._ftag+"_set%d_map%s.fits" % (k,"_srcfree" if srcfree else ""),ncomp=ncomp)

    def get_coadd_ivar(self,ncomp=3,srcfree=True):
        return self._read_map(self._ftag+"_coadd_ivar.fits" ,ncomp=ncomp)
    
    def get_split_ivar(self,k,ncomp=3,srcfree=True):
        return self._read_map(self._ftag+"_set%d_ivar.fits" % (k),ncomp=ncomp)

    def get_beam(self,ells):
        fname = self.beamroot+"/181220/mr3c_%s_nohwp_night_beam_tform_jitter_%s_181220.txt" % (self.saf,self.patch)
        ls,bells = np.loadtxt(fname,unpack=True,usecols=[0,1])
        assert np.isclose(ls[0],0)
        bnorm = bells[0]
        bells = bells/bnorm
        return maps.interp(ls,bells)(ells)
            


class PlanckHybrid(DataModel):
    def __init__(self,region,config):
        DataModel.__init__(self,region,config)
        self._ftag = lambda k,s: "planck_hybrid_%s_2way_%d_map%s.fits" % (self.c['id'],k,"_srcfree" if s else "")
        self._iftag = lambda k: "planck_hybrid_%s_2way_%divar.fits" % (self.c['id'],k)
        self.nsplits = 2
        self.beamroot = paths[self.c['beam_path_name']]
        
    def get_coadd(self,ncomp=3,srcfree=True):
        coadd = 0.
        ivars = 0.
        for i in range(self.nsplits):
            split = self.get_split(i,ncomp=ncomp,srcfree=srcfree)
            ivar = self.get_split_ivar(i,ncomp=ncomp,srcfree=srcfree)
            coadd += (split*ivar)
            ivars += ivar
        return np.nan_to_num(coadd / ivar) #FIXME: nan to num
    
    def get_split(self,k,ncomp=3,srcfree=True):
        return self._read_map(self._ftag(k,srcfree),ncomp=ncomp)

    def get_coadd_ivar(self,ncomp=3,srcfree=True):
        return self._read_map(self._iftag(k),ncomp=ncomp)
    
    def get_split_ivar(self,k,ncomp=3,srcfree=True):
        return self._read_map(self._ftag+"_set%d_ivar.fits" % (k),ncomp=ncomp)

    def get_beam(self,ells):
        bconfig = io.config_from_yaml(self.beamroot+"/planck.yml")
        return maps.gauss_beam(ells,bconfig[self.c['id']]['fwhm'])
    
        
    
adm = ACTmr3(gconfig['a1'])
a = adm.get_coadd(ncomp=1)
print(a.shape,a.wcs)
sys.exit()

datamodels = {
    'act_mr3': ACTmr3,
    'planck_hybrid': PlanckHybrid,
}

array_datamodel = config[array]['data_model']
dm = datamodels[array_datamodel](config[array])
dm.get_splits
