from __future__ import print_function
from orphics import maps,io,cosmology,stats,lensing
from pixell import enmap,lensing as enlensing,powspec,utils
import numpy as np
import os,sys
from enlib import bench
from soapack import interfaces as sints
from orphics import io,maps
from orphics import mpi


class TiledAnalysis(object):
    """MPI-enabled tiled analysis on rectangular pixel maps following
    Sigurd Naess' scheme.
    This class currently does not handle sky wrapping.

    You initialize by specifying the geometry of the full map, the MPI
    object and the pixel dimensions of the tiling scheme, with defaults
    set for a 0.5 arcmin pixel resolution following:
    http://folk.uio.no/sigurdkn/actpol/coadd_article/

    >>> ta = TiledAnalysis(shape,wcs,comm)

    The initializer calculates the relevant pixel boxes.
    You might want to prepare output maps for the final results.
    
    >>> ta.initalize_output(name="processed")

    A typical analysis then involves looping over the generator tiles()
    which returns an extracter function and an inserter function.
    The extracter function can be applied to a map of geometry (shape,wcs)
    such that the corresponding tile (for each MPI job and for the 
    current iteration) is extracted and prepared with apodization.

    >>> for extracter,inserter in ta.tiles():
    >>>     emap = extracter(imap)

    You can then proceed to analyze the extracted tile, possibly along with 
    corresponding tiles extracted from other maps. When you are ready to
    insert a processed result back in to a pre-initialized output map, you can
    do:
    >>> for extracter,inserter in ta.tiles():
    >>>     ...
    >>>     ta.update_output("processed",pmap,inserter)

    This updates the output map and its normalization (a normalization might be
    needed depending on choice of tile overlap and cross-fade scheme).

    When you are done with processing, you can get the final normalized output map
    (through MPI collection) outside the loop using:

    >>> outmap = ta.get_final_output("processed")

    Currently, sky-wrapping and partial tiles are not supported, so the final
    output geometry will be a cropped version of the input maps. To crop
    any map of the original geometry to the cropped geometry of the outputs,
    you can use:

    >>> cmap = ta.native_footprint(imap)

    which facilitates comparison of tiled outputs with untiled outputs.

    """
    def __init__(self,shape,wcs,comm=None,pix_width=480,pix_pad=480,pix_apod=120,pix_cross=240):
        self.ishape,self.iwcs = shape,wcs
        iNy,iNx = shape[-2:]
        self.numy = iNy // pix_width
        self.numx = iNx // pix_width
        Ny = self.numy * pix_width
        Nx = self.numx * pix_width
        dny = (iNy - Ny)//2
        dnx = (iNx - Nx)//2
        self.fpixbox = [[dny,dnx],[Ny+dny,Nx+dnx]]
        self.pboxes = []
        self.ipboxes = []
        sy = 0
        for i in range(self.numy):
            sx = 0
            for j in range(self.numx):
                self.pboxes.append( [[sy-pix_pad//2,sx-pix_pad//2],[sy+pix_width+pix_pad//2,sx+pix_width+pix_pad//2]] )
                self.ipboxes.append( [[sy-pix_pad//2+pix_apod,sx-pix_pad//2+pix_apod],
                                      [sy+pix_width+pix_pad//2-pix_apod,sx+pix_width+pix_pad//2-pix_apod]] )
                sx += pix_width
            sy += pix_width
        if comm is None:
            from orphics import mpi
            comm = mpi.MPI.COMM_WORLD
        self.comm = comm
        N = pix_width + pix_pad
        self.apod = enmap.apod(np.ones((N,N)), pix_apod, profile="cos", fill="zero")
        self.pix_apod = pix_apod
        self.N = N
        self.cN = self.N-self.pix_apod*2
        self.crossfade = self._linear_crossfade(pix_cross)
        self.outputs = {}

    def _prepare(self,imap):
        return imap*self.apod

    def _finalize(self,imap):
        return maps.crop_center(imap,self.cN)*self.crossfade

    def tiles(self):
        comm = self.comm
        for i in range(comm.rank, len(self.pboxes), comm.size):
            extracter = lambda x: self._prepare(enmap.extract_pixbox(self.native_footprint(x),self.pboxes[i]))
            inserter = lambda inp,out: enmap.insert_at(out,self.ipboxes[i],self._finalize(inp),op=np.ndarray.__iadd__)
            yield extracter,inserter
            
    def _linear_crossfade(self,npix):
        init = np.ones((self.cN,self.cN))
        cN = self.cN
        fys = np.ones((cN,))
        fxs = np.ones((cN,))
        fys[:npix] = np.linspace(0.,1.,npix)
        fys[cN-npix:] = np.linspace(0.,1.,npix)[::-1]
        fxs[:npix] = np.linspace(0.,1.,npix)
        fxs[cN-npix:] = np.linspace(0.,1.,npix)[::-1]
        return fys[:,None] * fxs[None,:]

    def initialize_output(self,name):
        omap = self.get_empty_map()
        self.outputs[name] = (omap,omap.copy())

    def get_empty_map(self):
        return self.native_footprint(enmap.zeros(self.ishape,self.iwcs))

    def native_footprint(self,imap):
        return enmap.extract_pixbox(imap,self.fpixbox)

    def update_output(self,name,emap,inserter):
        inserter(emap,self.outputs[name][0])
        inserter(emap*0+1,self.outputs[name][1])

    def get_final_output(self,name):
        return utils.allreduce(self.outputs[name][0],self.comm)/utils.allreduce(self.outputs[name][1],self.comm)
        
