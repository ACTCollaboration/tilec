from tilec import utils as tutils,covtools
import numpy as np
from orphics import io,maps,stats,cosmology
from pixell import enmap,enplot
from szar import foregrounds as fgs

class FlatSim(object):
    def __init__(self,shape,wcs,beams,rmss,lknees,alphas,aniss,inhoms,nsplits,response_dict,ps_dict,ellmin=100):
        """
        TODO: inhomogenity
        noise cross covariance
        """

        self.modlmap = enmap.modlmap(shape,wcs)
        self.beams = beams
        self.inhoms = inhoms
        self.nsplits = nsplits
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
        

shape,wcs = maps.rect_geometry(width_deg=10.,px_res_arcmin=0.5)

params = np.loadtxt("input/simple_sim.txt")
beams = params[:,0]
freqs = params[:,1]
rmss = params[:,2]
lknees = params[:,3]
alphas = params[:,4]
inhoms = params[:,5]
aniss = params[:,6]
nsplits = params[:,7].astype(np.int)

tcmb = 2.7255e6
response_dict = {}
response_dict['cmb'] = np.ones(shape=(beams.size,))
response_dict['y'] = fgs.ffunc(freqs)*tcmb
response_dict['dust'] = fgs.cib_nu(freqs)
theory = cosmology.default_theory()
ells = np.arange(0,8000,1)
cltt = theory.lCl('TT',ells)
ps_dict = {}
ps_dict['cmb'] = cltt
ps_dict['y'] = fgs.power_y(ells)
ps_dict['dust'] = fgs.power_cibp_raw(ells)+fgs.power_cibc_raw(ells)

sobj = FlatSim(shape,wcs,beams,rmss,lknees,alphas,aniss,inhoms,nsplits,response_dict,ps_dict)
imaps = sobj.get_maps(seed=100)
for k,imap in enumerate(imaps):
    for j in range(imap.shape[0]):
        io.plot_img(imap[j],"test_%d_%d.png" % (k,j))



# cobj = HILC(shape,wcs,response_dict)
# cov = HILC.learn(arrays)
# kmap = HILC.solve("cmb",deproject="y")
