# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:46:19 2021

@author: danpa
"""
import numpy as np
import z2pack
import matplotlib.pyplot as plt
import scipy.linalg as la
import cmath
#import kwant
import math
#from Z2pack_dev import * 
import os
import re
import sys
import time
import haldane_model
import topology_interface_main as tim

fname="bhz3d_output.OUT"
#k_vect=tim.generate_k_3D(nkx=10,nky=10,nkz=10)
nkz=11
nky=11
kx=np.linspace(0,1/2,nky)
ky=np.linspace(0,1,nkz) #[:-1]

kX,kY=np.meshgrid(kx,ky)
kY=np.ravel(kY)
kX=np.ravel(kX)
kZ=np.zeros_like(kX)
k_vect=np.stack((kX,kY,kZ),axis=1)
k_vect=np.vstack((np.zeros(3),k_vect))
print(k_vect)
ham=haldane_model.bhz_3D(0.5, 1, 0., 0, 1)

h2d=haldane_model.bhz(0.5, 1, 0., 0, 1)
print(np.allclose(ham(k_vect[3,:]),h2d(k_vect[3,:2])))
haldane_model.write_yaehmop_output_from_ham(fname,k_vect,ham)

k_vect ,occupation,overlap_k, hamiltonian_data,dim=tim.get_output_arrays(fname)
occ_bands=np.where(occupation[0,:]!=0)
hamiltonian=tim.ham_k(hamiltonian_data,k_vect)
system = z2pack.hm.System(hamiltonian,dim=3,bands=occ_bands,convention=1)
surface=lambda t1,t2: [t1/2,t2,0]
result = z2pack.surface.run(
    system=system,
    surface= surface, 
    pos_tol=None,
    gap_tol=None,
    move_tol=None,
    num_lines=nky, 
    iterator=range(nkz,40) 
    )
print(z2pack.invariant.z2(result))