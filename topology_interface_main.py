# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:56:26 2021

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

###############################################################################
# not needed anymore
###############################################################################
def change_basis(basis_a,basis_b):
    """change array a into basis from array b
    returns change of basis array """
    u=adjoint(basis_a) @ basis_b
    return u

def change_basis_array(array,array_basis,desired_basis):
    u=change_basis(array_basis,desired_basis)
    return adjoint(u) @ array @ u


def adjoint(arr):
    return np.conj(np.transpose(arr))

def extract_floats(array):
    new_array=[]
    for element in array:
        try:
            new_array.append(float(element))
        except:
            l=0 #array.remove(element)
    return new_array
            
def get_overlap(index,read_obj,num_orbitals):
    """extracts overlap matrix for single k value from output file """
    overlap=[]
    first_line=index
    first_line_reached=False
    num_pages=math.ceil(num_orbitals/4)
    n=0
    while first_line_reached==False:
        if "." in read_obj[index+n]:
            first_line_reached=True
            first_line=index+n
        n+=1
        
    for i in range(num_orbitals):
        
        array=[]
        for j in range(num_pages):
            array+=read_obj[first_line+i+j*(num_orbitals+1)].split(" ")
        array=extract_floats(array)
        overlap.append(array)
    return np.array(overlap)

def get_ham(index,read_obj,num_orbitals):
    h=[]
    first_line=index
    first_line_reached=False
    num_pages=math.ceil(num_orbitals/4)
    n=0
    while first_line_reached==False:
        if "." in read_obj[index+n]:
            first_line_reached=True
            first_line=index+n
        n+=1
        
    for i in range(num_orbitals):
        
        array=[]
        for j in range(num_pages):
            array+=read_obj[first_line+i+j*(num_orbitals+1)].split(" ")
        
        array=extract_floats(array)
        h.append(array)
    return np.array(h)

def get_k(line):
    """get an array of k vectors from Yaehmop output file
    each row is a new k vector. number of rows = number of k vectors"""
    ind_left=line.find("(")
    ind_right=line.find(")")
    array=line[ind_left+1:ind_right]
    array=array.split(" ")
    array=np.asarray(array, dtype=np.float64) #,order='C'
    return array

def get_num_orbitals(line):
    ind_left=line.find(":")
    num=line[ind_left+1:]
    return int(num)


def get_k_vect(filename):
    if not os.path.exists("kpoints"):
        from pymatgen.ext.matproj import MPRester
        mpid=(filename.split("/")[-1]).split(".")[0]
        
        with MPRester("vVLA7eR1BMX9IOzN") as M:
            BS = M.get_bandstructure_by_material_id(mpid)
        # catesian kpoints
        kpoints = []
        for val in BS.kpoints:
            kpoints.append(val.cart_coords) 
        kpoints = np.reshape(kpoints,(len(kpoints),len(kpoints[0])))
        np.savetxt("kpoints",kpoints)

    elif os.path.exists("kpoints"):
        kpoints=np.loadtxt("kpoints")

    return kpoints
    
def get_output_arrays(filename):
    
    k_vect = np.array([0,0,0])
    overlap_k=[]
    ham_k=[]
    # Open the file in read only mode
    f = open(filename, "r")
    f_list=f.readlines()
    for i,line in enumerate(f_list):
        # For each line, check if line contains the string
        if "Num_Orbitals" in line:
            num_orbitals=get_num_orbitals(line)
            
        if "Kpoint" in line:
            array=get_k(line)
            if len(array)==3:
                k_vect = np.vstack([k_vect, array])
            
        elif "--- Overlap Matrix S(K) ---" in line:
            temp_overlap=get_overlap(i,f_list,num_orbitals)
            overlap_k.append(temp_overlap)
            
        elif "--- Hamiltonian H(K) ---" in line:
            temp_ham=get_ham(i,f_list,num_orbitals)
            ham_k.append(temp_ham)
    k_vect=np.delete(k_vect,0,0)
    return np.array(k_vect) ,np.array(overlap_k), np.array(ham_k)

def read_HAM_OV(filename,n_kpoints):

    with open(filename,'rb') as f: 
        b = f.read() 
        dat = np.frombuffer(b,dtype=np.csingle)
        del b 
        f.close()
    n_matrix = int(np.sqrt((dat.shape[0]-1)/n_kpoints))
    return(dat[1:].reshape(n_kpoints,n_matrix,n_matrix))



##############################################################################
    
# Yaehmop output file parser
    
##############################################################################


def _parse_eigenvalues(filename):
    
    """
    Parse the bandstructure output of Yaehmop. 
    
    Parameters
    ----------
    bandfile: str
              path of the bandstructure file 
              
     Returns
    -------
    band_dict: dict
               dictionary of the bandstructure information  
               'kpoints': kpoints as numpy matrix
               'kpoints_weight': as list
               'special_point_label': labels for the special points
               'bands': Eigenvalues as numpy matrix 
               'efermi': fermi energy as float
                  
    """ 
       
    
    bandDict = {}
    bandDict['kpoints']= []
    bandDict['kpoints_weight'] = []
    bandDict['bands'] = []
    bandDict['occupation'] = []
    
    datafile = open(filename,"r").readlines()
    
    i=0
    while i < len(datafile):
        line=datafile[i]
        if "Fermi_Energy:" in line:
            bandDict['efermi'] = float(re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",line)[0])
            i+=1
        elif "Kpoint:" in line:
            points = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",line)
            bandDict['kpoints'].append([float(i) for i in points[1:-1]])
            bandDict['kpoints_weight'].append(float(points[-1]))
            
            eigenvalues = []
            occupation = []
            rest_list=datafile[i+1:]
            for line2 in rest_list:
                if "Total_Energy" in line2:
                    break
                else:
                    vals = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",line2)
                    if len(vals) ==3:
                        eigenvalues.append(float(vals[1]))
                        occupation.append(int(float(vals[2])))
                    i+=1
            bandDict['bands'].append(eigenvalues)
            bandDict['occupation'].append(occupation)
        else:
            i+=1
    bandDict['bands'] = np.array(bandDict['bands'])
    bandDict['kpoints'] = np.array(bandDict['kpoints'])
    bandDict['occupation'] = np.array(bandDict['occupation'])
    
    return bandDict
    
##############################################################################
    
##############################################################################
def check_hermitian(arr1):
    """Checks to see if an array is hermitian adjoint(arr)?=arr. print true if
    hermitian, false if not.
    
    :param arr1: (array) array to check"""
    
    print(np.allclose(arr1,adjoint(arr1)))


def ham_k(eigenvalues,k_vect):
    """create hamiltonian function with an input of k. Takes eigenvalues and
    fills in diagonals of an empty array, to create diagonal matrix.
    
    :param eigenvalues: (array) array of eigenvalues extracted from Yaehmop.
        size (n x m) n= number of kpoints, m = number of eigenvalues at one kpoint"""
        
    def h_k(k):
        
        # with open("KPOINTS","a") as f:
        #     f.write(str(k) +" \n")
        # f.close()
        index=np.argmin(np.linalg.norm(k_vect-k,axis=1))
        e_vals_k=eigenvalues[index,:]
        num_evals=len(e_vals_k)
        base_ham=np.zeros((num_evals,num_evals))
        np.fill_diagonal(base_ham,e_vals_k)
        return base_ham

    return h_k

def get_All_ChernNumbers(volume_result):
    """get all chern numbers of surfaces stored in a volume result. 
    
    :param volume_result: (obj) z2pack.volume.VolumeResult object
    
    :returns: (list) list of chern numbers for all surfaces"""
    
    surfaces=volume_result.surfaces
    chern_=[]
    for surf in surfaces:
        chern_.append(z2pack.invariant.chern(surf))
    return chern_

def get_All_Z2(volume_result):
    """get all z2 invariants of surfaces stored in a volume result. 
    
    :param volume_result: (obj) z2pack.volume.VolumeResult object
    
    :returns: (list) list of z2 invariants for all surfaces"""
    
    surfaces=volume_result.surfaces
    z2_=[]
    for surf in surfaces:
        z2_.append(z2pack.invariant.z2(surf))
    return z2_    

def close_to_any(a, floats):
  """checks to see if a float value is close to any floats in a given list
      within a certain tolerance
      
  :param a: (float) value to see if in list
  
  :param floats: (list) list to check 
  
  :returns: (bool) true if a is in floats, false if not"""
  return np.any(np.isclose(a, floats,rtol=1e-4))

def generate_k(nkx=10,nky=10,nkz=10,max_kx=1,max_ky=1,max_kz=1):
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
    ky=np.linspace(0,max_ky,nky)
    kz=np.linspace(0,1,nkz+1)[:-1]
    k_vect=[]
    for i in kx:
        for j in ky:
            for k in kz:
                
                k_vect.append([i,j,k])
                
    return np.array(k_vect)

def get_k_info(k_vect):
    """find max value of k in each direction. Also find number of different values
    in each direction
    
    :param k_vect: (array) kpoints array to get data from"""
    kx=[]
    ky=[]
    kz=[]
    for k in k_vect:
        if not close_to_any(k[0],kx):
            kx.append(k[0])
        if not close_to_any(k[1],ky):
            ky.append(k[1])
        if not close_to_any(k[2],kz):
            kz.append(k[2])
    return max(kx),max(ky),max(kz),len(kx),len(ky),len(kz),(abs(kz[1]-kz[0]))

identity = np.identity(2, dtype=complex)
pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
def test_hamiltonian(k, m, t1, t2, phi):
    kx, ky = k
    k_a = 2 * np.pi / 3. * np.array([-kx - ky, 2. * kx - ky, -kx + 2. * ky])
    k_b = 2 * np.pi * np.array([kx, -kx + ky, ky])
    H = 2 * t2 * np.cos(phi) * sum([np.cos(-val) for val in k_b]) * identity
    H += t1 * sum([np.cos(-val) for val in k_a]) * pauli_x
    H += t1 * sum([np.sin(-val) for val in k_a]) * pauli_y
    H += m * pauli_z
    H -= 2 * t2 * np.sin(phi) * sum([np.sin(-val) for val in k_b]) * pauli_z
    return H

def get_topology(filename):
    """get chern number and z2 invariants of all surfaces from hamiltonains,kpoints
    extracted from Yaehmop input file
    
    :param filename: (str) file to extract hamiltonians and corresponding kpoints from
    
    :returns: (list,list) chern number list, z2 invariant list of all surfaces"""
    
    #parse file to get all hamiltonians, kpoints and occupations
    bandDict = _parse_eigenvalues(filename)
    k_vect=bandDict["kpoints"]
    eig_vals_=bandDict["bands"]
    occupation=bandDict["occupation"]
    occ_bands=np.where(occupation[0,:]!=0)
    maxkx,maxky,maxkz,nkx,nky,nkz,dk=get_k_info(k_vect)
    
    #define hamiltonian, create z2pack system, run calculation
    hamiltonian=ham_k(eig_vals_,k_vect)
    system = z2pack.hm.System(hamiltonian,dim=3,bands=occ_bands,convention=1)
    result = z2pack.surface.run(
        system=system,
        surface= lambda t1,t2,t3: [maxkx*t1, maxky*t2, t3], 
        pos_tol=None,
        move_tol=None,
        num_lines=nky ,
        num_surfaces=nkx, 
        iterator=range(nkz,27) )
    

    chern_=get_All_ChernNumbers(result)
    z2_= get_All_Z2(result)
    return chern_, z2_


if __name__=="__main__":
    filename="C:/Users/danpa/Documents/argonne/topology_interface/eigenVal_parser/eigenVal_parser/mp-47/mp-47.out"
    
    chern_, z2_ = get_topology(filename);
    print("chern numbers for all surfaces ",chern_)
    print("z2 invariants for all surfaces ", z2_)    
    
    #wcc=get_topology(filename)
    #print(z2pack.invariant.z2(wcc))
    ##### figure out how to set momentum in z2pack 
    
    