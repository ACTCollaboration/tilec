import yaml, numpy
from pixell import enmap
import numpy as np
from orphics import maps

def tpower(p2d): return np.fft.fftshift(np.log10(p2d))

class Config(object):
    def __init__(self,arrays,yaml_file="input/arrays.yml"):
        self.arrays = arrays
        self.narrays = len(arrays)
        with open(yaml_file) as f:
            self.config = yaml.safe_load(f)
        self.darrays = {}
        for d in self.config['arrays']:
            self.darrays[d['name']] = d.copy()


        self.shape,self.wcs = enmap.read_fits_geometry(self.coaddfname(0))
        Ny,Nx = self.shape[-2:]
        self.fc = maps.FourierCalc(self.shape[-2:],self.wcs)
        self.modlmap = enmap.modlmap(self.shape,self.wcs)
            
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

    def get_coadds(self):
        """
        Should also beam deconvolve and low pass here
        (narray,Ny,Nx)
        """
        kcoadds = []
        wins = []
        mask = enmap.read_map(self.xmaskfname())
        ais = range(self.narrays)
        for ai in ais:
            # iwin = enmap.read_map(cinvvarfname(ai))[0]
            iwin = 1.
            window = mask*iwin
            wins.append(window)
            imap = enmap.read_map(self.coaddfname(ai),sel=np.s_[0,:,:])
            _,_,kcoadd = self.fc.power2d(window*imap)
            ifwhm,_ = self.get_beams(ai,ai)
            kcoadd = np.nan_to_num(kcoadd/maps.gauss_beam(self.modlmap,ifwhm))
            ncoadd = np.nan_to_num(kcoadd/np.sqrt(np.mean(window**2.))) # NEED TO RETHINK THE SPATIAL WEIGHTING
            kcoadds.append(ncoadd) 
        kcoadds = enmap.enmap(np.stack(kcoadds),self.wcs)
        wins = enmap.enmap(np.stack(wins),self.wcs)
        return kcoadds,wins

    def get_splits(self,ai):
        """
        Should also beam deconvolve and low pass here
        ai is index of array (in the "arrays" list that you specified as an argument)
        Return (nsplits,Ny,Nx) ndmap
        """
        nsplits = self.get_nsplits(ai)
        ksplits = []
        wins = []
        mask = enmap.read_map(self.xmaskfname()) # steve's mask
        for i in range(nsplits):
            # iwin = enmap.read_map(sinvvarfname(ai,i))
            iwin = 1. # window function is 1
            window = mask*iwin
            wins.append(window)
            imap = enmap.read_map(self.splitfname(ai,i),sel=np.s_[0,:,:]) # document sel usage
            _,_,ksplit = self.fc.power2d(window*imap)
            ksplits.append(ksplit)
        ksplits = enmap.enmap(np.stack(ksplits),self.wcs)
        wins = enmap.enmap(np.stack(wins),self.wcs)
        return ksplits,wins

    def get_single_coadd(self,ai):
        """
        ai is index of array (in the "arrays" list that you specified as an argument)
        Return (Ny,Nx) ndmap
        """
        mask = enmap.read_map(self.xmaskfname()) # steve's mask
        iwin = 1. # window function is 1
        window = mask*iwin
        imap = enmap.read_map(self.coaddfname(ai),sel=np.s_[0,:,:]) # document sel usage
        _,_,ksplit = self.fc.power2d(window*imap)
        ksplit = enmap.enmap(ksplit,self.wcs)
        win = enmap.enmap(window,self.wcs)
        return ksplit,win
    

    def ncalc(self,ai,aj,do_coadd_noise=True):
        """
        Cross spectrum and noise power calculator
        For i x j element of Cov
        ai and aj are array indices
        """
        iksplits,iwins = self.get_splits(ai) # each ksplit multiplied by mask and inv var map, returning also mask*inv var map
        if aj!=ai:
            jksplits,jwins = self.get_splits(aj) # each ksplit multiplied by mask and inv var map, returning also mask*inv var map
        else:
            jksplits = iksplits.copy()
            jwins = iwins.copy()
        nisplits = iksplits.shape[0]
        njsplits = jksplits.shape[0]
        autos = 0. ; crosses = 0.
        nautos = 0 ; ncrosses = 0
        for p in range(nisplits):
            for q in range(p,njsplits):
                if p==q:
                    nautos += 1
                    autos += self.fc.f2power(iksplits[p],jksplits[q]) / np.mean(iwins[p]*jwins[q])
                else:
                    ncrosses += 1
                    crosses += self.fc.f2power(iksplits[p],jksplits[q]) / np.mean(iwins[p]*jwins[q])
        autos /= nautos
        crosses /= ncrosses
        scov = crosses
        ncov = autos-crosses
        if do_coadd_noise:
            assert nisplits==njsplits
            ikcoadd,iwin = self.get_single_coadd(ai)
            if aj!=ai:
                jkcoadd,jwin = self.get_single_coadd(aj)
            else:
                jkcoadd = ikcoadd.copy()
                jwin = iwins.copy()
            npower = 0.
            for i in range(nisplits):
                diff1 = iksplits[i] - ikcoadd
                diff2 = jksplits[i] - jkcoadd
                npower += (self.fc.f2power(diff1,diff2) / np.mean(iwin*jwin))
            npower *= 1./((1.-1./nisplits)*nisplits)
            return enmap.enmap(scov,self.wcs),enmap.enmap(ncov,self.wcs),enmap.enmap(autos,self.wcs),enmap.enmap(npower,self.wcs)
        else:
            return enmap.enmap(scov,self.wcs),enmap.enmap(ncov,self.wcs),enmap.enmap(autos,self.wcs)
            
