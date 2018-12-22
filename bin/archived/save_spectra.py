from tilec import utils as tutils,covtool,fg
import argparse, yaml
import numpy as np
from pixell import enmap,fft,utils
from orphics import maps,io,stats,mpi


"""

Saves unsmoothed cross and auto spectra of all arrays

Issues:
2. Cross-covariances are noisy.
3. Beam is gaussian now.
6. Use Planck half mission

Do super simple sims first

"""

# Parse command line
parser = argparse.ArgumentParser(description='Save spectra.')
parser.add_argument("arrays", type=str,help='List of arrays named in arrays.yml.')
parser.add_argument("version", type=str,help='Version name.')
args = parser.parse_args()

# Load dictionary of array specs from yaml file
arrays = args.arrays.split(',')
config = tutils.Config(arrays = arrays)
    
narrays = len(arrays)
freqs = []
for i in range(narrays):
    f = config.darrays[arrays[i]]['freq']
    freqs.append(f)
    

    
# Get the common map geometry from the coadd map of the first array
shape,wcs = enmap.read_fits_geometry(coaddfname(0))
Ny,Nx = shape[-2:]
# Set up a fourier space calculator (for power spectra)
fc = maps.FourierCalc(shape[-2:],wcs)

modlmap = enmap.modlmap(shape,wcs)


comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
Njobs = int(narrays*(narrays+1.)/2.)
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]

ainds = []
for aindex1 in range(narrays):
    for aindex2 in range(aindex1,narrays) :
        ainds.append((aindex1,aindex2))



io.mkdir("data/spectra/%s" % version,comm)

for task in my_tasks:
    aindex1,aindex2 = ainds[task]
    proot = "data/spectra/%s/%s_%s_" % (version,arrays[aindex1],arrays[aindex2])

    print("Noise calc...")
    scov,_,_,ncov = ncalc(aindex1,aindex2) # raw power spectra in full 2d space before any downsampling
    enmap.write_map(proot + "signal.fits",scov)
    enmap.write_map(proot + "noise.fits",ncov)

    io.plot_img(tutils.tpower(scov),proot+"signal.png")
    io.plot_img(tutils.tpower(ncov),proot+"noise.png")

    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))
