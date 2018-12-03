from tilec import utils as tutils,covtools
import numpy as np
from orphics import io,maps,stats,cosmology
from pixell import enmap,enplot
from szar import foregrounds as fgs

class FlatSim(object):
    def __init__(self,shape,wcs,beams,rmss,lknees,alphas,aniss,inhoms,nsplits,plancks,response_dict,ps_dict,ellmin=100):
        """
        TODO: inhomogenity
        noise cross covariance
        """
        self.fc = maps.FourierCalc(shape,wcs)
        self.narrays = len(beams)
        self.modlmap = enmap.modlmap(shape,wcs)
        self.beams = beams
        self.inhoms = inhoms
        self.nsplits = nsplits
        self.plancks = plancks.astype(np.bool)
        self.ngens = []
        antemplate = covtools.get_anisotropic_noise_template(shape,wcs)
        for rms,lknee,alpha,anis in zip(rmss,lknees,alphas,aniss):
            if anis:
                template = antemplate.copy()
            else:
                template = 1
            p2d = covtools.rednoise(enmap.modlmap(shape,wcs),rms,lknee,alpha) * template
            p2d[self.modlmap<ellmin] = 0
            self.ngens.append(maps.MapGen(shape,wcs,p2d[None,None,...]))
        self.fgens = {}
        assert "cmb" in ps_dict.keys()
        self.components = ps_dict.keys()
        for key in ps_dict.keys():
            self.fgens[key] = maps.MapGen(shape,wcs,ps_dict[key][None,None,...])
        self.shape = shape
        self.wcs = wcs
        self.rdict = response_dict
        self.ellmin = ellmin

    def get_maps(self,seed=None):
        # SKY
        fgs = []
        for i,comp in enumerate(self.components):
            fgs.append(self.fgens[comp].get_map(seed=(1,i,seed)))
        arrays = []
        for k,(beam,inhom,nsplit) in enumerate(zip(self.beams,self.inhoms,self.nsplits)):
            # SKY
            sky = 0.
            for i,comp in enumerate(self.components):
                sky += (fgs[i] * self.rdict[comp][i])
            kbeam = maps.gauss_beam(self.modlmap,beam)
            kbeam[self.modlmap<self.ellmin] = 0
            sky_convolved = maps.filter_map(sky,kbeam)
            # NOISE
            nmaps = []
            for j in range(nsplit):
                nmap = self.ngens[k].get_map(seed=(2,j,seed)) * nsplit
                nmaps.append(nmap.copy())
            nmaps = np.stack(nmaps)
            observed = enmap.enmap(sky_convolved+nmaps,self.wcs)
            arrays.append(observed.copy())
        return arrays
        

