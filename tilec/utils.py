import yaml, numpy
from pixell import enmap
import numpy as np
from orphics import maps,io


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

        self.shape,self.wcs = enmap.read_fits_geometry(self.coaddfname(0))
        Ny,Nx = self.shape[-2:]
        self.Ny,self.Nx = Ny,Nx
        self.fc = maps.FourierCalc(self.shape[-2:],self.wcs)
        self.modlmap = enmap.modlmap(self.shape,self.wcs)
        self.mask = enmap.read_map(self.xmaskfname()) # steve's mask

            
    def get_beams(self,ai,aj): # get beam fwhm for array indices ai and aj
        return self.darrays[self.arrays[ai]]['beam'],self.darrays[self.arrays[aj]]['beam']
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
        try: splitstart = self.darrays[self.arrays[aindex]]['splitstart']
        except: splitstart = 0
        return self.darrays[self.arrays[aindex]]['froot'] + self.darrays[self.arrays[aindex]]['splits'] % (split + splitstart)
    def sinvvarfname(self,aindex,split):
        try: splitstart = self.darrays[self.arrays[aindex]]['splitstart']
        except: splitstart = 0
        return self.darrays[self.arrays[aindex]]['froot'] + self.darrays[self.arrays[aindex]]['sinvvars'] % (split + splitstart)
    def get_nsplits(self,aindex): return self.darrays[self.arrays[aindex]]['nsplits']
    def xmaskfname(self): return self.config['xmask']

    def load_coadd_real(self,ai): return enmap.read_map(self.coaddfname(ai),sel=np.s_[0,:,:])
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
            iwin = enmap.read_map(self.sinvvarfname(ai,i)).reshape((self.Ny,self.Nx))
            wins.append(iwin.copy())
            imap = enmap.read_map(self.splitfname(ai,i),sel=np.s_[0,:,:]) * iwin # document sel usage
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

