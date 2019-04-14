# This program converts planck maps to car, outputting:
#  map:  interpolated using harmonic interpolation
#  ivar: the inverse variance in T,Q,U of each pixel using nearest neighbor. All correlations ignored.
#  map0: like map, but using nearest neighbor. Optional
# I didn't expect this to become so involved, though much of the verbosity is status printing.
# It goes to some length to avoid wasting memory.
# WARNING: All provided files must have the same nside

import argparse, os, sys, time
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("template")
parser.add_argument("-l", "--lmax",    type=int, default=0)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-O", "--outputs", type=str, default="map,ivar")
args = parser.parse_args()
import numpy as np, healpy
from pixell import enmap, utils, curvedsky, coordinates, mpi, memory

# These factors when multiplied by the Planck maps convert from MJy/sr to K_CMB
factor_545 = 0.01723080316
factor_857 = 0.44089766765

unit  = 1e6
euler = np.array([57.06793215,  62.87115487, -167.14056929])*utils.degree
dtype = np.float32
ctype = np.result_type(dtype,0j)
rstep = 100
nside = 0
verbose = args.verbose
outputs = args.outputs.split(",")
if len(outputs) == 0:
	print("No outputs selected - nothing to do")
	sys.exit(0)
t0 = time.time()

def progress(msg):
	if verbose:
		print("%6.2f %6.2f %6.2f %s" % ((time.time()-t0)/60, memory.current()/1024.**3, memory.max()/1024.**3, msg))

comm = mpi.COMM_WORLD

shape, wcs = enmap.read_map_geometry(args.template)
shape = shape[-2:]
progress("Allocating output map %s %s" % (str((3,)+shape), str(dtype)))
if "map" in outputs:
	for ifile in args.ifiles[comm.rank::comm.size]:
		name = os.path.basename(ifile)
		runit = unit
		if "545" in name: 
			runit *= factor_545
			npol = 1
		elif "857" in name: 
			runit *= factor_857
			npol = 1
		elif ("smica" in name) or ("WPR2" in name):
			npol = 1
		else:
			npol = 3
		fields = range(npol)
		omap = enmap.zeros((npol,)+shape, wcs, dtype)
		progress("%s TQU read" % name)
		imap  = np.array(healpy.read_map(ifile, fields)).astype(dtype)
		nside = healpy.npix2nside(imap.shape[-1])
		progress("%s TQU mask" % name)
		imap[healpy.mask_bad(imap)] = 0
		progress("%s TQU scale" % name)
		imap *= runit
		nside = healpy.npix2nside(imap.shape[-1])
		lmax  = args.lmax or 3*nside
		progress("%s TQU alm2map" % name)
		alm   = curvedsky.map2alm_healpix(imap, lmax=lmax)
		del imap
		# work around healpix bug
		progress("%s TQU rotate_alm" % name)
		alm   = alm.astype(np.complex128,copy=False)
		healpy.rotate_alm(alm, euler[0], euler[1], euler[2])
		alm   = alm.astype(ctype,copy=False)
		progress("%s TQU map2alm" % name)
		curvedsky.alm2map_cyl(alm, omap)
		del alm
		ofile = ifile[:-5] + "_map.fits"
		progress("%s TQU write %s" % (name, ofile))
		enmap.write_map(ofile, omap)

def get_pixsize_rect(shape, wcs):
	"""Return the exact pixel size in steradians for the rectangular cylindrical
	projection given by shape, wcs. Returns area[ny], where ny = shape[-2] is the
	number of rows in the image. All pixels on the same row have the same area."""
	ymin  = enmap.sky2pix(shape, wcs, [-np.pi/2,0])[0]
	ymax  = enmap.sky2pix(shape, wcs, [ np.pi/2,0])[0]
	y     = np.arange(shape[-2])
	x     = y*0
	dec1  = enmap.pix2sky(shape, wcs, [np.maximum(ymin, y-0.5),x])[0]
	dec2  = enmap.pix2sky(shape, wcs, [np.minimum(ymax, y+0.5),x])[0]
	area  = np.abs((np.sin(dec2)-np.sin(dec1))*wcs.wcs.cdelt[0]*np.pi/180)
	return area

if "ivar" in outputs or "map0" in outputs:
	if not nside:
		progress("Determining nside")
		nside = healpy.npix2nside(healpy.read_map(args.ifiles[0]).size)
	progress("Computing pixel position map")
	dec, ra = enmap.posaxes(shape, wcs)
	pix  = np.zeros(shape, np.int32)
	psi  = np.zeros(shape, dtype)
	# Get the pixel area. We assume a rectangular pixelization, so this is just
	# a function of y
	ipixsize = 4*np.pi/(12*nside**2)
	opixsize = get_pixsize_rect(shape, wcs)
	progress("Computing inverse rotation")
	nblock = (shape[-2]+rstep-1)//rstep
	for bi in range(comm.rank, nblock, comm.size):
		if bi % comm.size != comm.rank: continue
		i = bi*rstep
		rdec = dec[i:i+rstep]
		opos = np.zeros((2,len(rdec),len(ra)))
		opos[0] = rdec[:,None]
		opos[1] = ra  [None,:]
		# This is unreasonably slow
		ipos = coordinates.transform("equ", "gal", opos[::-1], pol=True)
		pix[i:i+rstep,:] = healpy.ang2pix(nside, np.pi/2-ipos[1], ipos[0])
		psi[i:i+rstep,:] = -ipos[2]
		del ipos, opos
	for i in range(0, shape[-2], rstep):
		pix[i:i+rstep] = utils.allreduce(pix[i:i+rstep], comm)
		psi[i:i+rstep] = utils.allreduce(psi[i:i+rstep], comm)
	for output in outputs:
		if output not in ["ivar","map0"]: continue

		for ifile in args.ifiles[comm.rank::comm.size]:
			name = os.path.basename(ifile)
			runit = unit
			if "545" in name: 
				runit *= factor_545
				npol = 1
				fields = {"ivar":(2),"map0":(0)}[output]
			elif "857" in name: 
				runit *= factor_857
				npol = 1
				fields = {"ivar":(2),"map0":(0)}[output]
			else:
				fields = {"ivar":(4,7,9),"map0":(0,1,2)}[output]
				npol = 3
			omap = enmap.zeros((npol,)+shape, wcs, dtype)
			progress("%s %s read" % (name, output))
			imap  = np.array(healpy.read_map(ifile, fields)).astype(dtype)
			if imap.ndim==1:
				assert ("545" in name) or ("857" in name)
				imap = imap[None]
			progress("%s %s mask" % (name, output))
			bad = healpy.mask_bad(imap)
			if output == "ivar": bad |= imap <= 0
			imap[bad] = 0
			progress("%s %s scale" % (name, output))
			runit = unit
			if "545" in name: runit *= factor_545
			if "857" in name: runit *= factor_857
			if output == "ivar":
				imap[~bad] = 1/imap[~bad] / runit**2
			else:
				imap *= runit
			del bad
			# Read off the nearest neighbor values
			progress("%s %s interpol" % (name, output))
			omap[:] = imap[:,pix]
			if output == "ivar":
				progress("%s %s area rescale" % (name, output))
				omap *= opixsize[:,None]/ipixsize
				# We ignore QU mixing during rotation for the noise level, so
				# it makes no sense to maintain distinct levels for them
				mask = omap[1:]>0
				omap[1:] = np.mean(omap[1:],0)
				omap[1:]*= mask
				del mask
			if output == "map0":
				progress("%s %s polrot" % (name, output))
				# We need to apply the polarization rotation
				for i in range(0, shape[-2], rstep):
					omap[1:3,i:i+rstep,:] = enmap.rotate_pol(omap[1:3,i:i+rstep,:], psi[i:i+rstep,:])
			ofile = ifile[:-5] + "_" + output + ".fits"
			progress("%s %s write %s" % (name, output, ofile))
			enmap.write_map(ofile, omap)
