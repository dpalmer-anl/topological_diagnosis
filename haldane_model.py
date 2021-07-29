# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:14:14 2021

@author: danpa
"""
import numpy as np
import z2pack
import matplotlib.pyplot as plt
import scipy.linalg as la
import math
import os
import topology_interface_main as tim
import itertools

# if os.path.exists("haldane_output.OUT"):
#     os.remove("haldane_output.OUT")
identity = np.identity(2, dtype=complex)
pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

def test_hamiltonian(m, t1, t2, phi):
    def haldane_hamiltonian(k):
        kx, ky = k
        k_a = 2 * np.pi / 3. * np.array([-kx - ky, 2. * kx - ky, -kx + 2. * ky])
        k_b = 2 * np.pi * np.array([kx, -kx + ky, ky])
        H = 2 * t2 * np.cos(phi) * sum([np.cos(-val) for val in k_b]) * identity
        H += t1 * sum([np.cos(-val) for val in k_a]) * pauli_x
        H += t1 * sum([np.sin(-val) for val in k_a]) * pauli_y
        H += m * pauli_z
        H -= 2 * t2 * np.sin(phi) * sum([np.sin(-val) for val in k_b]) * pauli_z
        return H
    return haldane_hamiltonian

def generate_k(nkx=10,nky=10,max_kx=1,max_ky=1):
        """generate kpoints for Yaehmop such that they meet requirements of Z2pack.
        Note that the maximum k-value in the z direction must be 1 so that 
        k(t1,t2,0)=k(t1,t2,1)+G , where G is an inverse lattice vector
        
        :param nkx: (int) number of kpoints in x direction
        
        :param nky: (int) number of kpoints in y direction
        
        :param nkz: (int) number of kpoints in z direction
        
        :param max_kx: (float) max k value in x direction. must be from 0 to 1
        
        :param max_ky: (float) max k value in y direction. must be from 0 to 1
        
        :returns: (array) np.array of kpoints, (3 X nkx*nky*nkz)
        """
        kx=np.linspace(0,max_kx,nkx)
        ky=np.linspace(0,max_ky,nky+1)[:-1]
        k_vect=[]
        for i in kx:
            for j in ky:
                k_vect.append([i,j])
                    
        return np.array(k_vect)
def write_yaehmop_output_from_haldane(k_vect,m, t1, t2, phi):
    hamiltonian=test_hamiltonian(m, t1, t2, phi)
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
        "#Num_Orbitals: 2 \n",
        
        "; ------  Lattice Parameters ------ \n",
        "#Dimensionality: 2 \n",
        "#Lattice Vectors \n",
        "(a) 2.513094 0.000000 0.000000 \n",
        "(b) -1.256547 2.176403 0.000000 \n",
        "(c) 0.000000 0.000000 4.181402 \n",
        
        
        "; RHO = 22.627844 \n"]
    
    f=open("haldane_output.OUT","a")
    f.writelines(header)
    line=[]
    for i,k in enumerate(k_vect):
        line.append(";***& Kpoint: "+str(i+1)+" ("+str(k[0])+" "+str(k[1])+") Weight: 0.000000 \n")
        
        line.append("#	******* Energies (in eV)  and Occupation Numbers ******* \n")
        temp_h=hamiltonian(k)
        eigval=la.eigh(temp_h)[0]
        line.append(" \n")
        line.append(";		 --- Hamiltonian H(K) --- \n")
        line.append("    C(  1) 2s       C(  1) 2px \n")
        for n in range(len(temp_h[:,0])):
            line.append(" C(  1) 2s       "+str(temp_h[n,0])+"         "+str(temp_h[n,1])+" \n")
                
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
        "#Fermi_Energy:  -28.376236 \n"]
    list(itertools.chain.from_iterable(line))
    
    
    f.writelines(line)
    f.writelines(end)
    f.close()

if __name__=="__main__":
    k_vect=generate_k() 
    write_yaehmop_output_from_haldane(k_vect,0.5, 1., 1. / 3., 0.5 * np.pi)
        
    
    
    