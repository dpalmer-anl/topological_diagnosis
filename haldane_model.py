# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:14:14 2021

@author: danpa
"""
import numpy as np
import z2pack
import scipy.linalg as la
import math
import os
import topology_interface_main as tim
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

identity = np.identity(2, dtype=complex)
pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
pauli_vector = [pauli_x, pauli_y, pauli_z]

def ssh_hamiltonian(alpha,t1=-0.75):
    def s_hamiltonian(k):
        try:
            if len(k)!=1:
                k=k[0]
        except:
            l=0
        # if type(k)==np.ndarray or type(k)==list:
        #     kf=k[1]
        
        k*=2*np.pi
        t2=-t1*alpha
        Rx=t1-t2*np.cos(k)
        Ry=-t2*np.sin(k)
        
        h=Rx*pauli_x + Ry*pauli_y
        return h
    return s_hamiltonian

def haldane_hamiltonian(m, t1, t2, phi):
    def h_hamiltonian(k):
        kx, ky = k
        k_a = 2 * np.pi / 3. * np.array([-kx - ky, 2. * kx - ky, -kx + 2. * ky])
        k_b = 2 * np.pi * np.array([kx, -kx + ky, ky])
        H = 2 * t2 * np.cos(phi) * sum([np.cos(-val) for val in k_b]) * identity
        H += t1 * sum([np.cos(-val) for val in k_a]) * pauli_x
        H += t1 * sum([np.sin(-val) for val in k_a]) * pauli_y
        H += m * pauli_z
        H -= 2 * t2 * np.sin(phi) * sum([np.sin(-val) for val in k_b]) * pauli_z
        return H
    return h_hamiltonian

def bhz(A, B, C, D, M):
    def h(k):
        kx, ky = 2* np.pi * np.array(k)
        d = [
            A * np.sin(kx), -A * np.sin(ky),
            -2 * B * (2 - (M / (2 * B)) - np.cos(kx) - np.cos(ky))
        ]
        H = sum(di * pi for di, pi in zip(d, pauli_vector))
        epsilon = C - 2 * D * (2 - np.cos(kx) - np.cos(ky))
        H += epsilon
        return H

    def Hamiltonian(k):
        return np.vstack([
            np.hstack([h(k), np.zeros((2, 2))]),
            np.hstack([np.zeros((2, 2)),
                       h(-np.array(k)).conjugate()])
        ])

    return Hamiltonian

def bhz_3D(A,B,C,D,M):
    
    def h(k):
        kz=np.pi*k[-1]
        ham_func_2D=bhz(A,B,C,D,M)
        partial_ham=ham_func_2D(k[0:2])
        H=np.zeros_like(partial_ham)+partial_ham
        extra_dim=A*kz*np.identity(2)
        H[2:4,0:2]=np.conj(np.transpose(extra_dim))
        H[0:2,2:4]=np.conj(np.transpose(extra_dim))
        return H
    return h

def test_bhz_3D(A,B,M):
    def h(k):
        kx,ky,kz=np.pi*np.array(k)
        M_k=M-B*(np.linalg.norm(k))
        k_plus=kx+1j*ky
        k_minus=kx-1j*ky
        
        h_00=np.array([[M_k,A*k_plus],[A*k_minus,-M_k]])
        h_11=np.array([[M_k,-A*k_plus],[-A*k_minus,-M_k]])
        h_10=np.array([[0,np.conj(A)*kz],[np.conj(A)*kz,0]])
        h_01=np.array([[0,np.conj(A)*kz],[np.conj(A)*kz,0]])
        
        H=np.vstack((np.hstack((h_00,h_01)),
                     np.hstack((h_10,h_11))))
        return H
    return h
    
def plot_bands(h_k,k_vect,axis=0):
    bands=[]
    for k in k_vect:
        # if type(k)==list or type(k)==np.array:
        #     k=k[axis]
        h=h_k(k)
        eigval=la.eigh(h)[0]
        bands.append(eigval)
    bands=np.array(bands)
    for i in range(len(bands[0,:])):
        plt.plot(k_vect[:,axis],bands[:,i])
    plt.title("band structure")
    plt.ylabel("Energy")
    plt.xlabel("Kx")
    
def write_yaehmop_output_from_ham(fname,k_vect,hamiltonian):
    
    dim=len(k_vect[0,:])
    ham_dim=len(hamiltonian(k_vect[0,:])[:,0])
    header=["#BIND_OUTPUT version: 3.0 \n",
        
        "#JOB_TITLE: mp-47 \n",
        
        "; ********* Atoms within the unit cell:  ********* \n",
        "# NUMBER OF ATOMS:  \n",
        "	5 \n",
        "# Crystallographic coordinates of atoms: \n",
        "   1    C 0.333333 0.666667 0.062772 \n",
        "   2    C 0.666667 0.333333 0.562772 \n",
        "  3    C 0.666667 0.333333 0.937228 \n",
        "   4    C 0.333333 0.666667 0.437228 \n",
        "   5    & 0.000000 0.000000 0.000000 \n",
        
        
        "# ******** Extended Hueckel Parameters ******** \n",
        ";  FORMAT  quantum number orbital: Hii, <c1>, exponent1, <c2>, <exponent2> \n",
        
        "ATOM: C   Atomic number: 6  # Valence Electrons: 4 \n",
        "	2S:   -35.7440     2.0083 \n",
        "	2P:   -29.0014     1.9063 \n",
        "ATOM: &   Atomic number: -1  # Valence Electrons: 0 \n",
        "# CRYSTAL SPECIFICATION: \n",
        "#Cell Constants:  \n",
        "2.5131 2.5131 4.1814  \n",
        "# alpha:  90.0000 \n",
        "# beta:  90.0000 \n",
        "# gamma:  120.0000 \n",
        "#Cell Volume: 22.8702 cubic Angstroms \n",
        "# Positions of atoms from crystal coordinates \n",
        "  1    C  -0.0000   1.4509   0.2625 \n",
        "   2    C   1.2565   0.7255   2.3532 \n",
        "   3    C   1.2565   0.7255   3.9189 \n",
        "   4    C  -0.0000   1.4509   1.8282 \n",
        "   5    &   0.0000   0.0000   0.0000 \n",
        
        "; Number of orbitals \n",
        "#Num_Orbitals: "+str(ham_dim)+" \n",
        
        "; ------  Lattice Parameters ------ \n",
        "#Dimensionality: "+str(dim)+" \n",
        "#Lattice Vectors \n",
        "(a) 2.513094 0.000000 0.000000 \n",
        "(b) -1.256547 2.176403 0.000000 \n",
        "(c) 0.000000 0.000000 4.181402 \n",
        
        
        "; RHO = 22.627844 \n"]
    
    f=open(fname,"w")
    f.writelines(header)
    line=[]
    for i,k in enumerate(k_vect):
        kstr=""
        for s in range(dim):
            kstr+=str(k[s])+" "
        line.append(";***& Kpoint: "+str(i+1)+" ("+kstr+") Weight: 0.000000 \n")
        
        temp_h=hamiltonian(k)
        eigval=la.eigh(temp_h)[0]
        line.append(" \n")
        line.append(";		 --- Hamiltonian H(K) --- \n")
        line.append("    C(  1) 2s       C(  1) 2px \n")
        for n in range(len(temp_h[:,0])):
            temp_str=""
            for m in range(len(temp_h[0,:])):
                temp_str+=str(temp_h[n,m])+"         "
            line.append(" C(  1) 2s       "+temp_str+" \n")
        
        line.append("#	******* Energies (in eV)  and Occupation Numbers ******* \n")
        for j,val in enumerate(eigval):
            occ=0
            if (j+1)<=int(len(eigval)/2):
                occ=2
            line.append(str(j+1)+":--->  "+str(val)+"  ["+str(occ)+" Electrons] \n")
            
        line.append("Total_Energy: "+str(sum(eigval))+ " \n")
        line.append(" \n")
        line.append(" \n")
        
    end=[";  The Fermi Level was determined for 217 K points based on \n",
        ";     an ordering of 3472 crystal orbitals occupied by 16.000000 electrons \n",
        ";      in the unit cell (3472.000000 electrons total) \n",
        "#Fermi_Energy:  0 \n"]
    # list(itertools.chain.from_iterable(line))
    
    
    f.writelines(line)
    f.writelines(end)
    f.close()

if __name__=="__main__":
    # k_vect=tim.generate_k(nkx=10,nky=10,nkz=10,max_kx=1,max_ky=1,dim=2) 
    # hamiltonian=haldane_hamiltonian(0.5, 1., 1. / 3., 0.5 * np.pi)
    # write_yaehmop_output_from_ham("haldane_output.OUT",k_vect,hamiltonian)
    
    # k2_vect=tim.generate_k(nkx=10,max_kx=1.0,dim=1)
    # hamiltonian=ssh_hamiltonian(2,t1=-1.0)
    # write_yaehmop_output_from_ham("ssh_output.OUT",k2_vect,hamiltonian)
    
    #k3_vect=tim.generate_k(max_kx=0.5,dim=2)
    #hamiltonian=bhz(0.5, 1., 0., 0., 1.)
    #write_yaehmop_output_from_ham("bhz_output.OUT",k3_vect,hamiltonian)
    
    k_vect=np.zeros((100,3))
    axis=0
    k_vect[:,1]=np.zeros_like(k_vect[:,1])
    k_vect[:,axis]=np.linspace(-1,1,100)
    h_k=test_bhz_3D(2.0, 0, 1.0) #bhz_3D(.5, 1,0,0, -1) #
    # plot_bands(h_k,k_vect,axis=axis)
    # plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x=np.linspace(-1,1,100)
    y=np.linspace(-1,1,100)
    X,Y=np.meshgrid(x,y)
    
    x_=np.ravel(X)
    y_=np.ravel(Y)
    zs=np.zeros((4,X.shape[0],X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            
            ham=h_k([X[i,j],Y[i,j],0])
            eigval=la.eigh(ham)[0]
        
            zs[:,i,j]=eigval
    
    Z = zs.reshape((4,X.shape[0],X.shape[1]))
    
    for n in range(4):
        ax.plot_surface(X, Y, Z[n,:,:])
    
    ax.set_xlabel('Ky')
    ax.set_ylabel('Kx')
    ax.set_zlabel('Energy')
    
    plt.show()
    
    
    
    
    
        
    
    
    