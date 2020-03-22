from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs
import numpy as np
import os,sys
from tilec import utils as tutils
import healpy as hp

#Port of healpix module coord_v_convert.f90 to python by JLS
#Feb 28, 2017

from numpy import sin,cos
from numpy import arctan2 as atan2
from numpy import sqrt

froot = "/scratch/r/rbond/msyriac/data/depot/hpymap/"

DTOR = np.pi/180.0

def angdist(v1,v2):
    sprod=np.dot(v1,v2)
    v3=np.cross(v1,v2)
    vprod=sqrt(np.dot(v3,v3))
    return atan2(vprod,sprod)

def py_coordsys2euler_zyz(iepoch, oepoch, isys, osys):
    #, psi, theta, phi)
    v1=np.asarray([1.0, 0.0, 0.0])
    v2=np.asarray([0.0, 1.0, 0.0])
    v3=np.asarray([0.0, 0.0, 1.0])

    v1p=py_xcc_v_convert(v1,iepoch,oepoch,isys,osys)
    v2p=py_xcc_v_convert(v2,iepoch,oepoch,isys,osys)
    v3p=py_xcc_v_convert(v3,iepoch,oepoch,isys,osys)

    v1p=v1p/sqrt(np.dot(v1p,v1p))
    v2p=v2p/sqrt(np.dot(v1p,v1p))
    v3p=v3p/sqrt(np.dot(v1p,v1p))

    theta=angdist(v3,v3p)
    psi=atan2(v2p[2],-v1p[2])
    phi=atan2(v3p[1],v3p[0])
    return psi,theta,phi




def py_xcc_v_convert(ivector,iepoch,oepoch,isys,osys):
    isys=isys.lower()
    osys=osys.lower()
    isys=isys[0]
    osys=osys[0]
    if (isys=='c'):
        isys='q'
    if (osys=='c'):
        osys='q'

    if (isys=='q'):
        ivector=py_xcc_dp_q_to_e(ivector,iepoch)
    if (isys=='g'):
        ivector=py_xcc_dp_g_to_e(ivector,iepoch)
    if (iepoch!=oepoch):
        ivector=py_xcc_dp_precess(ivector,iepoch,oepoch)
    
    if (osys=='q'):
        ivector=py_xcc_dp_e_to_q(ivector,oepoch)
    if (osys=='g'):
        ivector=py_xcc_dp_e_to_g(ivector,oepoch)
    return ivector

def py_xcc_dp_e_to_q(ivector,epoch):

    T = (epoch - 1900.e0) / 100.e0
    epsilon = 23.452294e0 - 0.0130125e0*T - 1.63889e-6*T**2 + 5.02778e-7*T**3
    
    hvector=np.zeros(ivector.shape)
    dc = cos(DTOR * epsilon)
    ds = sin(DTOR * epsilon)
    hvector[0] = ivector[0]
    hvector[1] = dc*ivector[1] - ds*ivector[2]
    hvector[2] = dc*ivector[2] + ds*ivector[1]
    return hvector

def py_xcc_dp_q_to_e(ivector,epoch):
    hvector=np.zeros(ivector.shape)
    T = (epoch - 1900.e0) / 100.e0
    epsilon = 23.452294e0 - 0.0130125e0*T - 1.63889e-6*T**2  + 5.02778e-7*T**3
    dc = cos(DTOR * epsilon)
    ds = sin(DTOR * epsilon)
    hvector[0] = ivector(1)
    hvector[1] = dc*ivector[1] + ds*ivector[2]
    hvector[2] = dc*ivector[2] - ds*ivector[1]
    return hvector



def py_xcc_dp_e_to_g(ivector,epoch):

    T=np.asarray([-0.054882486e0, -0.993821033e0, -0.096476249e0, 0.494116468e0, -0.110993846e0,  0.862281440e0, -0.867661702e0, -0.000346354e0,  0.497154957e0])
    T=np.reshape(T,[3,3])
    #T=T.transpose()

    if (epoch != 2000.0):
        ivector=py_xcc_dp_precess(ivector,epoch,2000.0)
    return np.dot(T,ivector)

def py_xcc_dp_g_to_e(ivector,epoch):
    T=np.asarray([-0.054882486e0, -0.993821033e0, -0.096476249e0, 0.494116468e0, -0.110993846e0,  0.862281440e0, -0.867661702e0, -0.000346354e0,  0.497154957e0])
    T=np.reshape(T,[3,3])
    T=T.transpose()

    hvector=np.dot(T,ivector)
    if (epoch != 2000.0):
        return py_xcc_dp_precess(hvector,2000.0,epoch)
    else:
        return hvector
    assert(1==0) #never get here
    

def py_xcc_dp_q_to_e(ivector,epoch):
    # Set-up: 
    T = (epoch - 1900.0) / 100.0
    epsilon = 23.452294 - 0.0130125*T - 1.63889e-6*T**2  + 5.02778e-7*T**3


    hvector=np.zeros(ivector.shape)
    # Conversion
    dc = cos(DTOR * epsilon)
    ds = sin(DTOR * epsilon)
    hvector[0] = ivector[0]
    hvector[1] = dc*ivector[1] + ds*ivector[2]
    hvector[2] = dc*ivector[2] - ds*ivector[1]
    return hvector

def py_xcc_dp_precess(ivector,iepoch,oepoch):
    Tm = ((oepoch+iepoch)/2.0 - 1900.0) / 100.0
    gp_long  = (oepoch-iepoch) * (50.2564+0.0222*Tm) / 3600.0
    dE       = (oepoch-iepoch) * (0.4711-0.0007*Tm) / 3600.0
    obl_long = 180.0 - (173.0 + (57.060+54.770*Tm)/60.0)+ gp_long/2.0
    dL       = gp_long - obl_long

    
    tvector=np.zeros(ivector.shape)
    #     Z-axis rotation by OBL_LONG:
    dco = cos(DTOR * obl_long)
    dso = sin(DTOR * obl_long)
    tvector[0] = ivector[0]*dco - ivector[1]*dso
    tvector[1] = ivector[0]*dso + ivector[1]*dco
    tvector[2] = ivector[2]

    #     X-axis rotation by dE:
    dce = cos(DTOR * dE)
    dse = sin(DTOR * dE)
    temp       = tvector[1]*dce - tvector[2]*dse
    tvector[2] = tvector[1]*dse + tvector[2]*dce
    tvector[1] = temp

    # Z-axis rotation by GP_LONG - OBL_LONG:
    dcl = cos(DTOR * dL)
    dsl = sin(DTOR * dL)
    temp       = tvector[0]*dcl - tvector[1]*dsl
    tvector[1] = tvector[0]*dsl + tvector[1]*dcl
    tvector[0] = temp

    return tvector


def rotate_alm(alm,iepoch,oepoch,isys,osys):
    phi,theta,psi=py_coordsys2euler_zyz(iepoch,oepoch,isys,osys)
    hp.rotate_alm(alm,phi,theta,psi)
    return alm

solution = 'comptony'
tdir = "/scratch/r/rbond/msyriac/data/depot/tilec/"
dcomb = 'joint'

nside = 2048
lmax = 3*nside

#for deproject in [None,'cib']:
for deproject in ['cib']:
    hmap = 0
    hmask = 0
    for region in ['deep56','boss']:
        imap = enmap.read_map(tutils.get_generic_fname(tdir,region,solution,deproject=deproject,data_comb=dcomb,version="v1.1.1",sim_index=None))
        mask = enmap.read_map(tutils.get_generic_fname(tdir,region,solution,deproject=deproject,data_comb=dcomb,version="v1.1.1",sim_index=None,mask=True))

        malm = cs.map2alm(mask,lmax=lmax)
        ialm = cs.map2alm(imap,lmax=lmax)

        malm   = malm.astype(np.complex128,copy=False)
        malm = rotate_alm(malm,2000.0,2000.0,'C','G')

        ialm   = ialm.astype(np.complex128,copy=False)
        ialm = rotate_alm(ialm,2000.0,2000.0,'C','G')


        imask = maps.binary_mask(hp.alm2map(malm,nside))
        hmap = hmap + hp.alm2map(ialm,nside)*imask
        hmask = hmask + imask


    io.mollview(hmap,f'{froot}test_rot_{deproject}.png')
    io.mollview(hmask,f'{froot}test_rot_mask_{deproject}.png')

    hp.write_map(f'{froot}test_rot_{deproject}.fits',hmap,overwrite=True)
    hp.write_map(f'{froot}test_rot_mask_{deproject}.fits',hmask,overwrite=True)

