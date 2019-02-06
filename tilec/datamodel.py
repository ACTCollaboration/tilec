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
        self.region = region
        self.mask = enmap.read_map(masks[region])
        self.shape,self.wcs = self.mask.shape,self.mask.wcs
        self.maproot = paths[self.c['map_path_name']]
        self.fc = maps.FourierCalc(self.shape,self.wcs)
        

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


    def process(self,ncomp=1,srcfree=True):
        """
        ai is index of array (in the "arrays" list that you specified as an argument)
        Return (nsplits,Ny,Nx) fourier transform
        Return (Ny,Nx) fourier transform of coadd
        """
        if ncomp!=1: raise NotImplementedError
        
        nsplits = self.nsplits
        ksplits = []
        imaps = []
        wins = []
        for i in range(nsplits):
            iwin = self.get_split_ivar(i,srcfree)
            wins.append(iwin.copy())
            imap = self.get_split(i,ncomp,srcfree) * iwin
            imaps.append(imap.copy())
            _,_,ksplit = self.fc.power2d(imap*self.mask)
            ksplits.append(ksplit.copy()/ np.sqrt(np.mean((iwin*self.mask)**2.)))
        ksplits = enmap.enmap(np.stack(ksplits),self.wcs)
        wins = enmap.enmap(np.stack(wins),self.wcs)
        imaps = enmap.enmap(np.stack(imaps),self.wcs)
        coadd = np.nan_to_num(imaps.sum(axis=0)/wins.sum(axis=0))
        kcoadd = enmap.enmap(self.fc.fft(coadd*self.mask) / np.sqrt(np.mean(self.mask**2.)),self.wcs)
        return ksplits,kcoadd
    
    
class ACTmr3(DataModel):
    def __init__(self,region,config):
        DataModel.__init__(self,region,config)
        season,array,freq = self.c['id'].split('_')
        self.saf = '_'.join([season,array,freq])
        self.patch = self.region
        pref = '_'.join([season,self.region,array,freq])
        self._ftag = "%s_nohwp_night_3pass_4way" % (pref)
        self.beamroot = paths[self.c['beam_path_name']]
        self.nsplits = 4
        
    def get_coadd(self,ncomp=3,srcfree=True):
        return self._read_map(self._ftag+"_coadd_map%s.fits" % ("_srcfree" if srcfree else ""),ncomp=ncomp)
    
    def get_split(self,k,ncomp=3,srcfree=True):
        return self._read_map(self._ftag+"_set%d_map%s.fits" % (k,"_srcfree" if srcfree else ""),ncomp=ncomp)

    def get_coadd_ivar(self,srcfree=True):
        return self._read_map(self._ftag+"_coadd_ivar.fits" ,ncomp=None)
    
    def get_split_ivar(self,k,srcfree=True):
        return self._read_map(self._ftag+"_set%d_ivar.fits" % (k),ncomp=None)

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
        self._iftag = lambda k: "planck_hybrid_%s_2way_%d_ivar.fits" % (self.c['id'],k)
        self.nsplits = 2
        self.beamroot = paths[self.c['beam_path_name']]
        
    def get_coadd(self,ncomp=3,srcfree=True):
        coadd = 0.
        ivars = 0.
        for i in range(self.nsplits):
            split = self.get_split(i,ncomp=ncomp,srcfree=srcfree)
            ivar = self.get_split_ivar(i,srcfree=srcfree)
            coadd += (split*ivar)
            ivars += ivar
        return np.nan_to_num(coadd / ivar) #FIXME: nan to num
    
    def get_split(self,k,ncomp=3,srcfree=True):
        return self._read_map(self._ftag(k,srcfree),ncomp=ncomp)

    def get_split_ivar(self,k,ncomp=3,srcfree=True):
        return self._read_map(self._iftag(k),ncomp=ncomp)

    def get_beam(self,ells):
        bconfig = io.config_from_yaml(self.beamroot+"/planck.yml")
        return maps.gauss_beam(ells,bconfig[self.c['id']]['fwhm'])
    
        
    
# Data model names specified in config file
datamodels = {
    'act_mr3': ACTmr3,
    'planck_hybrid': PlanckHybrid,
}
