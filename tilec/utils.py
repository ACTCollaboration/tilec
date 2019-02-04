import yaml, numpy, six
from pixell import enmap
import numpy as np
from orphics import maps,io

"""
saf = season_array_freq
"""

def get_beam(ells,patch,saf,directory='/home/msyriac/data/act/beams/181220/'):
    fname = "mr3c_%s_nohwp_night_beam_tform_jitter_%s_181220.txt" % (saf,patch)
    ls,bells = np.loadtxt(directory+fname,unpack=True,usecols=[0,1])
    assert np.isclose(ls[0],0)
    bnorm = bells[0]
    bells = bells/bnorm
    return maps.interp(ls,bells)(ells)
    


def estimate_separable_pixwin_from_normalized_ps(ps2d):
    """
    Tool from Sigurd to account for MBAC 220 CEA->CAR pixelization

    Sigurd: Then the spectrum is whitened with: ps2d /= ywin[:,None]**2; ps2d /= xwin[None,:]**2
    And the fourier map is whitened by the same, but without the squares

    Mat M [6:11 PM]
    great, and what does normalized ps mean?

    Sigurd [6:12 PM]
    ah. It's a power spectrum that's been normalized so that the white noise level should be 1

    Mat M [6:12 PM]
    mmm, how did you do that exactly

    Sigurd [6:12 PM]
    it's based on div
    
    Mat M [6:12 PM]
    i see
    """
    
    mask = ps2d < 2
    res  = []
    for i in range(2):
        profile  = np.sum(ps2d*mask,1-i)/np.sum(mask,1-i)
        profile /= np.percentile(profile,90)
        profile  = np.fft.fftshift(profile)
        edge     = np.where(profile >= 1)[0]
        if len(edge) == 0:
            res.append(np.full(len(profile),1.0))
            continue
        edge = edge[[0,-1]]
        profile[edge[0]:edge[1]] = 1
        profile  = np.fft.ifftshift(profile)
        # Pixel window is in signal, not power
        profile **= 0.5
        res.append(profile)
    return res


class Config(object):
    def __init__(self,arrays=None,yaml_file="input/arrays.yml"):
        with open(yaml_file) as f:
            self.config = yaml.safe_load(f)
        self.darrays = {}
        for d in self.config['arrays']:
            self.darrays[d['name']] = d.copy()
        self.arrays = arrays
        if self.arrays is None: self.arrays = list(self.darrays.keys())
        self.narrays = len(self.arrays)

        self.mask = enmap.read_map(self.xmaskfname()) # steve's mask
        self.shape,self.wcs = self.mask.shape,self.mask.wcs
        Ny,Nx = self.shape[-2:]
        self.Ny,self.Nx = Ny,Nx
        self.fc = maps.FourierCalc(self.shape[-2:],self.wcs)
        self.modlmap = enmap.modlmap(self.shape,self.wcs)

            
    def get_beam(self,ells,array): 
        ibeam = self.darrays[array]['beam']
        if isinstance(ibeam, six.string_types):
            # return get_beam(ells,"deep56",array,'/home/msyriac/data/act/beams/181220/')
            ls,bells = np.loadtxt(ibeam,unpack=True,usecols=[0,1])
            assert np.isclose(ls[0],0)
            bnorm = bells[0]
            bells = bells/bnorm
            return maps.interp(ls,bells)(ells)
        else:
            return maps.gauss_beam(ells,ibeam)
    def isplanck(self,aindex): # is array index ai a planck array?
        name = self.darrays[self.arrays[aindex]]['name'].lower()
        return True if ("hfi" in name) or ("lfi" in name) else False
    def is90150(self,ai,aj): # is array index ai,aj an act 150/90 combination?
        iname = self.darrays[self.arrays[ai]]['name'].lower()
        jname = self.darrays[self.arrays[aj]]['name'].lower()
        return True if (("90" in iname) and ("150" in jname)) or (("90" in jname) and ("150" in iname)) else False
    # Functions for filenames corresponding to an array index
    def coaddfname(self,aindex): return self.darrays[self.arrays[aindex]]['froot'] + self.darrays[self.arrays[aindex]]['coadd']
    def cinvvarfname(self,aindex): return self.darrays[self.arrays[aindex]]['froot'] + self.darrays[self.arrays[aindex]]['cinvvar']
    def splitfname(self,aindex,split):
        try:
            splitstart = self.darrays[self.arrays[aindex]]['splitstart']
            raise # always have splitstart = 0 hack
        except: splitstart = 0
        return self.darrays[self.arrays[aindex]]['froot'] + self.darrays[self.arrays[aindex]]['splits'] % (split + splitstart)
    def sinvvarfname(self,aindex,split):
        try: splitstart = self.darrays[self.arrays[aindex]]['splitstart']
        except: splitstart = 0
        return self.darrays[self.arrays[aindex]]['froot'] + self.darrays[self.arrays[aindex]]['sinvvars'] % (split + splitstart)
    def get_nsplits(self,aindex): return self.darrays[self.arrays[aindex]]['nsplits']
    def xmaskfname(self): return self.config['xmask']

    def _read_map(self,name,**kwargs):
        if "planck_hybrid" in name:
            ishape,iwcs = enmap.read_fits_geometry(name)
            pixbox = enmap.get_pixbox(iwcs,self.shape,self.wcs)
            omap = enmap.read_map(name,pixbox=pixbox,**kwargs)
            # if omap.ndim>2: omap = omap[0]
            return omap
        else:
            return enmap.read_map(name,**kwargs)
        
    # def load_coadd_real(self,ai): return self._read_map(self.coaddfname(ai),sel=np.s_[0,:,:])
    def load(self,ai,skip_splits=False):
        """
        ai is index of array (in the "arrays" list that you specified as an argument)
        Return (nsplits,Ny,Nx) fourier transform
        Return (Ny,Nx) fourier transform of coadd
        """
        nsplits = self.get_nsplits(ai)
        ksplits = []
        imaps = []
        wins = []
        for i in range(nsplits):
            #print(i,nsplits)
            iwin = self._read_map(self.sinvvarfname(ai,i)).reshape((self.Ny,self.Nx))
            wins.append(iwin.copy())
            imap = self._read_map(self.splitfname(ai,i),sel=np.s_[0,:,:]) * iwin # document sel usage
            imaps.append(imap.copy())
            if not(skip_splits):
                _,_,ksplit = self.fc.power2d(imap*self.mask)
                ksplits.append(ksplit.copy()/ np.sqrt(np.mean((iwin*self.mask)**2.)))
        if not(skip_splits): ksplits = enmap.enmap(np.stack(ksplits),self.wcs)
        wins = enmap.enmap(np.stack(wins),self.wcs)
        imaps = enmap.enmap(np.stack(imaps),self.wcs)
        coadd = np.nan_to_num(imaps.sum(axis=0)/wins.sum(axis=0))
        kcoadd = enmap.enmap(self.fc.fft(coadd*self.mask) / np.sqrt(np.mean(self.mask**2.)),self.wcs)
        return ksplits,kcoadd

