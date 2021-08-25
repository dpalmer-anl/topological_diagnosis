# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:52:47 2021

@author: danpa
"""


import numpy as np
import topology_interface_main as tim
import haldane_model
import z2pack

def generate_kpoints(ranges,nkx):
    """surfaces: ranges for kx, ky, kz
    default is [(0,1),(0,1),(0,1)]. if surface range is int or same, will automatically
    set nk equal to 1
    ranges dimensions= (number of surfaces, 3, 2)
    """
    nky=nkx
    nkz=nkx
    k_vect=np.array([0,0,0])
    ranges=np.array(ranges)
    if len(ranges.shape)<3:
        it=1
    else:
        it=ranges.shape[0]
    for i in range(it):
        numx=nkx
        numy=nky
        numz=nkz
        if len(ranges.shape)<3:
            s=ranges
        else:
            s=ranges[i,:,:]
        if type(s[0])==float:
            s[0]=(s[0],s[0])
        if type(s[1])==float:
            s[1]=(s[1],s[1])
        if type(s[2])==float:
            s[2]=(s[2],s[2])
        if np.isclose(s[0,0],s[0,1]):
            numx=1
        if np.isclose(s[1,0],s[1,1]):
            numy=1
        if np.isclose(s[2,0],s[2,1]):
            numz=1
        #main kpoint builder

        kx=np.linspace(s[0,0],s[0,1],numx)
        ky=np.linspace(s[1,0],s[1,1],numy)
        kz=np.linspace(s[2,0],s[2,1],numz)

        X,Y,Z=np.meshgrid(kx,ky,kz)
        X=np.ravel(X)
        Y=np.ravel(Y)
        Z=np.ravel(Z)
        
        k_vect=np.vstack((k_vect,np.stack((X,Y,Z),axis=1)))

    return k_vect

def get_k_surfaces(k_vect,dim=3):
    
    return None

def z2_3d(system,surfaces,nk):
    n=nk
    z2_indices=np.zeros(4)
    surfaces=np.array(surfaces)
    if len(surfaces.shape)<3:
        it=1
    else:
        it=surfaces.shape[0]
    z2_=np.zeros(it)  
    for i in range(it):

        if len(surfaces.shape)<3:
            s=surfaces
        else:
            s=surfaces[i,:,:]
            
        if np.isclose(s[0,0],s[0,1]):
            numx=1
            surface=lambda t1,t2: [s[0,1], t1*s[1,1], t2*s[2,1]]
        if np.isclose(s[1,0],s[1,1]):
            numy=1
            surface=lambda t1,t2: [t1*s[0,1], s[1,1], t2*s[2,1]]
        if np.isclose(s[2,0],s[2,1]):
            numz=1
            surface=lambda t1,t2: [t1*s[0,1], t2*s[1,1], s[2,1]]
        
        result = z2pack.surface.run(
            system=system,
            surface= surface, 
            pos_tol=None,
            gap_tol=None,
            move_tol=None,
            num_lines=n, 
            iterator=range(n,40) 
            )
        
        z2_[i]=z2pack.invariant.z2(result, check_kramers_pairs=False)
    
    n4_n1=abs(z2_[3]+z2_[0])%2
    n5_n2=abs(z2_[4]+z2_[1])%2
    n6_n3=abs(z2_[5]+z2_[2])%2
    if n4_n1==1 or n5_n2==1 or n6_n3==1:
        z2_indices[0]=1
    z2_indices[1:]=z2_[0:3]
    return z2_indices

if __name__=="__main__":
    normal_edge_surfaces=np.array([[(1,1),(0,1),(0,1)],
                            [(0,1),(1,1),(0,1)],
                            [(0,1),(0,1),(1,1)],
                            [(0,0),(0,1),(0,1)],
                            [(0,1),(0,0),(0,1)],
                            [(0,1),(0,1),(0,0)],
                            ])
    
    bhz_edge_surfaces=np.array([[(1,1),(0,1/2),(0,1)],
                            [(0,1/2),(1,1),(0,1)],
                            [(0,1/2),(0,1),(1,1)],
                            [(0,0),(0,1/2),(0,1)],
                            [(0,1/2),(0,0),(0,1)],
                            [(0,1/2),(0,1),(0,0)],
                            ])
    nkx=10

    k_vect=generate_kpoints(bhz_edge_surfaces,nkx)
    fname="bhz_output3d.OUT"
    ham=haldane_model.bhz_3D(.5, 1,0,0, 1) #test_bhz_3D(1.0, 0,  m) #
    haldane_model.write_yaehmop_output_from_ham(fname,k_vect,ham)
    
    k_vect ,occupation,overlap_k, hamiltonian_data,dim=tim.get_output_arrays(fname)
    occ_bands=np.where(occupation[0,:]!=0)
    hamiltonian=tim.ham_k(hamiltonian_data,k_vect)

    #define hamiltonian, create z2pack system, run calculation
    system = z2pack.hm.System(hamiltonian,dim=dim,bands=occ_bands,convention=1)
    z2_indices=z2_3d(system,bhz_edge_surfaces,nkx)
    print(z2_indices)

          
    

