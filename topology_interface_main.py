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
import haldane_model 


def change_basis(basis_a,basis_b):
    """change array a into basis from array b
    returns change of basis array 
    :param basis_a: (array) basis of array a
    
    :param basis_b: (array) basis of array b
    
    :returns U: (array) change of basis array from a to b """
    u=adjoint(basis_a) @ basis_b
    return u

def change_basis_array(array,array_basis,desired_basis):
    """change basis of given array into desired basis """
    u=change_basis(array_basis,desired_basis)
    return adjoint(u) @ array @ u


def adjoint(arr):
    """return adjoint of given array """
    return np.conj(np.transpose(arr))


##############################################################################
    
# Yaehmop output file parsers
    
##############################################################################
    
def extract_floats(array,is_real=True):
    """extract floats from a list of strings """
    new_array=[]
    for element in array:
        try:
            if is_real:
                #print(element)
                new_array.append(float(element))
            else:
                new_array.append(complex(element))
        except:
            l=0 
    return new_array
            
def get_overlap(index,read_obj,num_orbitals):
    """extracts overlap matrix for single k value from output file 
    
    :param index: (int) index in file corresponding to line where overlap matrix starts
    
    :param read_obj: (obj) python file object of output file
    
    :param num_orbitals: (int) number of orbitals. sets dimensions of overlap matrix
    
    :returns: (array) overlap matrix dimension (num_orbitals, num_orbitals)"""
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
    return overlap

def get_ham(index,read_obj,num_orbitals):
    """extracts hamiltonian matrix for single k value from output file 
    
    :param index: (int) index in file corresponding to line where hamiltonian starts
    
    :param read_obj: (obj) python file object of output file
    
    :param num_orbitals: (int) number of orbitals. sets dimensions of hamiltonian
    
    :returns: (array) hamiltonian, dimension (num_orbitals, num_orbitals)"""
    
    h=[]
    first_line=index+2
    first_line_reached=False
    num_pages=math.ceil(num_orbitals/4)
    n=0
    for i in range(num_orbitals):
        
        array=""
        for j in range(num_pages):
            array+=read_obj[first_line+i+j*(num_orbitals+1)]
        array=array.split(" ")

        array=extract_floats(array,is_real=False)
        h.append(array)
    return h

def get_occ(index,read_obj,num_orbitals):
    """extracts occupation for single k value from output file 
    :param index: (int) index in file corresponding to line where occupations start
    
    :param read_obj: (obj) python file object of output file
    
    :param num_orbitals: (int) number of orbitals. sets dimensions of occupations
    
    :returns: (array) occupations, dimension (num_orbitals,)"""
    
    occ=[]
    first_line=index
    first_line_reached=False
    n=0
    while first_line_reached==False:
        if "." in read_obj[index+n]:
            first_line_reached=True
            first_line=index+n
        n+=1
    array=[]   
    for i in range(num_orbitals):
        line=read_obj[first_line+i]
        line=line.replace("[","")
        line=line.replace("]","")
        array=extract_floats(line.split(" "))
        occ.append(int(array[-1]))
    return occ

def get_k(line):
    """get an array of k vectors from Yaehmop output file
    each row is a new k vector. number of rows = number of k vectors
    
    :param line: (str) line where k vector is stored
    
    :returns: (array) array of k vector"""
    
    ind_left=line.find("(")
    ind_right=line.find(")")
    array=line[ind_left+1:ind_right]
    array=array.split(" ")
    k=[]
    for element in array:
        try:
            k.append(float(element))
        except:
            l=0

    return np.array(k)

def get_num_orbitals(line):
    """extracts number of orbitals from output file 
    
    :param line: (str) line where number of orbitals is stored
    
    :returns: (int) number of orbitals"""
    
    ind_left=line.find(":")
    num=line[ind_left+1:]
    return int(num)

def get_dim(line):
    """extracts number of dimensions from output file 
    
    :param line: (str) line where number of orbitals is stored
    
    :returns: (int) dimensionality"""
    
    ind_left=line.find(":")
    num=line[ind_left+1:]
    return int(num)

def get_k_vect(filename):
    """get kpoints from materials project """
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
    """parses yaehmop output file to get kpoints, occupation, overlap as a function of k
    hamiltonian as a function of k, and number of dimensions
    
    :param filename: (str) filename to parse
    
    :returns:
        kpoints: array (number of kpoints, dimensions)
        occupation: array (number of kpoints, number of orbitals)
        overlap matrices: array (number of kpoints, number of orbitals, number of orbitals)
        hamiltonians: array (number of kpoints, number of orbitals, number of orbitals)
        dimensions: int
        """
    k_vect = np.array([0,0,0])
    overlap_k=[]
    ham_k=[]
    occupation=[]
    dim=3
    # Open the file in read only mode
    f = open(filename, "r")
    f_list=f.readlines()
    t=0
    for i,line in enumerate(f_list):
        # For each line, check if line contains the string
        if "Dimensionality" in line:
            dim=get_dim(line)
            k_vect = np.zeros(dim)
        if "Num_Orbitals" in line:
            num_orbitals=get_num_orbitals(line)
            
        if "Kpoint" in line:
            array=get_k(line)
            
            if len(array)==dim:
                k_vect = np.vstack([k_vect, array])
            
        elif "--- Overlap Matrix S(K) ---" in line:
            temp_overlap=get_overlap(i,f_list,num_orbitals)
            overlap_k.append(temp_overlap)
            
        elif "--- Hamiltonian H(K) ---" in line:
            temp_ham=get_ham(i,f_list,num_orbitals)
            ham_k.append(temp_ham)
            
        elif "Occupation Numbers" in line:
            temp_occ=get_occ(i,f_list,num_orbitals)
            occupation.append(temp_occ)
            
    k_vect=np.delete(k_vect,0,0)
    return np.array(k_vect) ,np.array(occupation),np.array(overlap_k), np.array(ham_k,dtype=complex), dim


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
    
##############################################################################
    
#              Kpoint Methods
    
##############################################################################
def generate_k_3D(n=10,surfaces="default"):
    """ Generate Kpoints given ranges for each k direction. i.e. a range of [(0,1),(0,1),(0,1)] means all combinations of
    kx, ky,kz from 0 to 1 will be placed in a numpy array.
    
    :param surfaces: (array, nx3x2) ranges for kx, ky, kz default is the 6 surfaces on the edge of the BZ. if surface range
    same, will automatically set nk equal to 1, ranges dimensions= [(number of surfaces)*nkx*nky*nkz, 3, 2]
    
    :param n: (int) number of kpoints for each direction nkx=nky=nkz
    
    :return kpoints: (array)
    """
    if surfaces=="default":
        surfaces=np.array([[(1,1),(0,1),(0,1)],
                            [(0,1),(1,1),(0,1)],
                            [(0,1),(0,1),(1,1)],
                            [(0,0),(0,1),(0,1)],
                            [(0,1),(0,0),(0,1)],
                            [(0,1),(0,1),(0,0)],
                            ])
    nkx=n
    nky=n
    nkz=n
    k_vect=np.array([0,0,0])
    if len(surfaces.shape)<3:
        it=1
    else:
        it=surfaces.shape[0]
    for i in range(it):
        numx=nkx
        numy=nky
        numz=nkz
        if len(surfaces.shape)<3:
            s=surfaces
        else:
            s=surfaces[i,:,:]
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
                
def generate_k_1D(nkx=10):
    """generate kpoints for Yaehmop such that they meet requirements of Z2pack.
    Note that the maximum k-value in the z direction must be 1 so that 
    k(t1,t2,0)=k(t1,t2,1)+G , where G is an inverse lattice vector
    
    :param nkx: (int) number of kpoints in x direction
    
    :returns: (array) np.array of kpoints, length= nkx
    """
    
    k_vect=[]
    kx=np.linspace(0,1,nkx)[:-1]               
    for j in kx:
        k_vect.append([j])
                
    return np.array(k_vect)


def generate_k_2D(n=10,surfaces="default"):
    """ Generate Kpoints given ranges for each k direction. i.e. a range of [(0,1),(0,1),(0,1)] means all combinations of
    kx, ky,kz from 0 to 1 will be placed in a numpy array.
    
    :param surfaces: (array, nx3x2) ranges for kx, ky, kz default is the 6 surfaces on the edge of the BZ. if surface range
    same, will automatically set nk equal to 1, ranges dimensions= [(number of surfaces)*nkx*nky*nkz, 3, 2]
    
    :param n: (int) number of kpoints for each direction nkx=nky=nkz
    
    :return kpoints: (array)
    """
    if surfaces=="default":
        surfaces=np.array([(0,1),(0,1)])
    nkx=n
    nky=n
    k_vect=np.array([0,0])
    if len(surfaces.shape)<3:
        it=1
    else:
        it=surfaces.shape[0]
    for i in range(it):
        numx=nkx
        numy=nky
        if len(surfaces.shape)<3:
            s=surfaces
        else:
            s=surfaces[i,:,:]
        if np.isclose(s[0,0],s[0,1]):
            numx=1
        if np.isclose(s[1,0],s[1,1]):
            numy=1

        #main kpoint builder

        kx=np.linspace(s[0,0],s[0,1],numx)
        ky=np.linspace(s[1,0],s[1,1],numy)

        X,Y=np.meshgrid(kx,ky)
        X=np.ravel(X)
        Y=np.ravel(Y)
        
        k_vect=np.vstack((k_vect,np.stack((X,Y),axis=1)))

    return k_vect 

def get_k_info(k_vect,dim=3):
    """find max value of k in each direction. Also find number of different values
    in each direction
    
    :param k_vect: (array) kpoints array to get data from"""
    kx=[0]
    ky=[0]
    kz=[0]
    for k in k_vect:
        
        if not close_to_any(k[0],kx) and k[0]>0:
            kx.append(k[0])
        if dim>=2:
            if not close_to_any(k[1],ky) and k[1]>0:
                ky.append(k[1])
        if dim==3:
            if not close_to_any(k[2],kz) and k[2]>0:
                kz.append(k[2])
    if dim==1:
        return max(kx),len(kx)
    if dim==2:
        return max(kx),max(ky),len(kx),len(ky)
    if dim==3:
        return max(kx),max(ky),max(kz),len(kx),len(ky),len(kz)
    
##############################################################################
        
#            Topology calculation methods
        
###############################################################################
        
def check_hermitian(arr1):
    """Checks to see if an array is hermitian adjoint(arr)?=arr. print true if
    hermitian, false if not.
    
    :param arr1: (array) array to check"""
    
    print(np.allclose(arr1,adjoint(arr1)))

def ham_k(hamiltonian_data,overlap,k_vect):
    """create hamiltonian function with an input of k. Takes hamiltonian array at
    all kpoints and returns hamiltonian at provided k point. If overlap matrices are
    present in Yaehmop output file, will change into overlap basis.
    
    :param hamiltonian_data: (array) array of hamiltonians at all kpoints extracted from Yaehmop.
        size (n x m x m) n= number of kpoints, m = dimension of hamiltonian
        
    :param overlap: (array) array of overlap matrices at all kpoints extracted from Yaehmop.
        size (n x m x m) n= number of kpoints, m = dimension of hamiltonian if present in file
        
    :param k_vect: (array) vector of kpoints
    
    :returns: (function) h as a function of k"""
    
    is_overlap=False
    if np.shape(overlap)==np.shape(hamiltonian_data):
        is_overlap=True
    def h_k(k):
        index=np.argmin(np.linalg.norm(k_vect-k,axis=1))
        h=hamiltonian_data[index,:,:]
        if is_overlap:
            h_basis=la.eigh(h)[1]
            o_basis=la.eigh(overlap[index,:,:])[1]
            h=change_basis_array(h,h_basis,o_basis)
        return h

    return h_k

def ham_k_fromEigenvalues(eigVal_data,k_vect,efermi):
    """create hamiltonian function with an input of k. Takes eigenvalues array at
    all kpoints and returns diagonal hamiltonian at provided k point. Sets fermi level=0
    
    :param eigVal_data: (array) array of hamiltonians at all kpoints extracted from Yaehmop.
        size (n x m x m) n= number of kpoints, m = dimension of hamiltonian
        
    :param k_vect: (array) vector of kpoints
    
    :param efermi: (float) fermi energy
    
    :returns: (function) h as a function of k"""
        
    def h_k(k):
        index=np.argmin(np.linalg.norm(k_vect-k,axis=1))
        eigval=eigVal_data[index,:]
        #set fermi energy equal to zero
        eigval-=efermi
        ham_dim=len(eigval)
        h=np.zeros((ham_dim,ham_dim))
        np.fill_diagonal(h,eigval)
        return h
    
    return h_k


def z2_3d(system,n,surfaces):
    """calculate 3D z2 invariant of a given z2pack system
    
    :param system: (z2pack.hm.system obj) system to calculate invariant of
    
    :param surfaces: (array) surfaces to calculate 2D z2 invariant over
        dimension= (6,3,2)
    
    :param n: (int) number of kpoints in each direction
    
    :returns z2 indices: (list) 3D z2 invariant indices, length = 4 """
    
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
    

def close_to_any(a, floats):
  """checks to see if a float value is close to any floats in a given list
      within a certain tolerance
      
  :param a: (float) value to see if in list
  
  :param floats: (list) list to check 
  
  :returns: (bool) true if a is in floats, false if not"""
  return np.any(np.isclose(a, floats,rtol=1e-4))


    
def get_topology(filename,ham_type="eigval",surfaces="default",n=10):
    """get topological invariants from data extracted from Yaehmop output file
    works for 1D, 2D, 3D depending on len of k vector. 
    1D -> [kx] , 2D -> [kx,ky] , 3D -> [kx,ky,kz]
    
    :param filename: (str) file to extract hamiltonians or eigenvalues
        and corresponding kpoints from
        
    :param ham_type: (str) construct diagonal hamiltonian from eigenvalues (default mode)
        or extract full hamiltonian from Yaehmop output file.
        
    :param surfaces: (str) kpoint surfaces over which to calculate invariants. 
        defaults are different in each dimension. Yaehmop output file contain these surfaces
        
    :param n: (int) number of kpoints in each direction. Yaehmop output file must match.
         assumes number of points in each direction are the same, i.e. #kx=#ky=#kz
    
    :returns: for 1D case, winding phase (float)
             for 2D case, z2 invariant (int) and chern number (float or int)
             for 3D case, 3D z2 invariant (list of ints, length = 4)
             """
    
    #parse file to get all hamiltonians, kpoints and occupations
    
    if ham_type=="eigval":
        bandDict = _parse_eigenvalues(filename)
        k_vect=bandDict["kpoints"]
        eig_vals_=bandDict["bands"]
        occupation=bandDict["occupation"]
        efermi=bandDict['efermi']
        occ_bands=np.where(occupation[0,:]!=0)
        dim=len(k_vect[0,:])
        hamiltonian=ham_k_fromEigenvalues(eig_vals_,k_vect,efermi)
    
    if ham_type=="general":
        k_vect ,occupation,overlap_k, hamiltonian_data,dim=get_output_arrays(filename)
        occ_bands=np.where(occupation[0,:]!=0)
        hamiltonian=ham_k(hamiltonian_data,overlap_k,k_vect)

    #define hamiltonian, create z2pack system, run calculation
    system = z2pack.hm.System(hamiltonian,dim=dim,bands=occ_bands,convention=1)
    if dim==1:
        maxkx,nkx=get_k_info(k_vect,dim=dim)
        result = z2pack.line.run(
            system=system,
            line= lambda t1: [t1], 
            pos_tol=None,
            iterator=range(nkx,40)
             )
        
    
        winding_= result.pol 
        z_=None
        return winding_, z_
    
    if dim==2:
        if surfaces=="default":
            surfaces=np.array([(0,1),(0,1)])
        maxkx,maxky,nkx,nky=get_k_info(k_vect,dim=dim)
        result = z2pack.surface.run(
            system=system,
            surface= lambda t1,t2: [surfaces[0,1]*t1, t2], 
            pos_tol=None,
            gap_tol=None,
            move_tol=None,
            num_lines=n , 
            iterator=range(n,40) )
        
    
        chern_=z2pack.invariant.chern(result)
        z2_=z2pack.invariant.z2(result, check_kramers_pairs=False)

        return chern_, z2_
    
    if dim==3:
        if surfaces=="default":
            surfaces=np.array([[(1,1),(0,1),(0,1)],
                            [(0,1),(1,1),(0,1)],
                            [(0,1),(0,1),(1,1)],
                            [(0,0),(0,1),(0,1)],
                            [(0,1),(0,0),(0,1)],
                            [(0,1),(0,1),(0,0)],
                            ])
        z2_indices=z2_3d(system,n,surfaces)
        c_=None
        
        return c_, z2_indices


if __name__=="__main__":
    
    ##########################################################################
    
    # Verifications for 1D, 2D, 3D
    
    ##########################################################################
    
    #1D verification
    k_vect=generate_k_1D(nkx=10)
    hamiltonian=haldane_model.ssh_hamiltonian(2,t1=-1.0)
    haldane_model.write_yaehmop_output_from_ham("ssh_output.OUT",k_vect,hamiltonian)
    c,z = get_topology("ssh_output.OUT",ham_type="general")
    print("Winding number of SSH Model ", c)
    
    #2D verification, haldane model and 2D BHZ model
    #-Haldane Model
    m, t1, t2, phi=0.5, 1., 1. / 3., 0.5 * np.pi
    k_vect=generate_k_2D()
    fname="haldane_output.OUT"

    ham=haldane_model.haldane_hamiltonian(m,t1,t2,phi)
    haldane_model.write_yaehmop_output_from_ham(fname,k_vect,ham)
    c, z = get_topology(fname,ham_type="general")
    print("Chern number of Haldane Model ", c)
    print("z2 invariant of Haldane Model ", z)
    
    #-2D BHZ model
    surfaces=np.array([(0,1/2),(0,1)])
    k_vect=generate_k_2D(n=10,surfaces=surfaces)
    hamiltonian=haldane_model.bhz(0.5, 1., 0., 0., 1.)
    haldane_model.write_yaehmop_output_from_ham("bhz_output.OUT",k_vect,hamiltonian)
    c, z = get_topology("bhz_output.OUT",ham_type="general",surfaces=surfaces)
    print("Chern number of 2D BHZ Model ", c)
    print("z2 invariant of 2D BHZ Model ", z)

    #3D verification, 3D BHZ model
    bhz_edge_surfaces=np.array([[(1,1),(0,1/2),(0,1)],
                            [(0,1/2),(1,1),(0,1)],
                            [(0,1/2),(0,1),(1,1)],
                            [(0,0),(0,1/2),(0,1)],
                            [(0,1/2),(0,0),(0,1)],
                            [(0,1/2),(0,1),(0,0)],
                            ])
    n=10
    fname="bhz_output3d.OUT"
    k_vect=generate_k_3D(n=n,surfaces=bhz_edge_surfaces)
    ham=haldane_model.bhz_3D(.5, 1,0,0, 1.0) 
    haldane_model.write_yaehmop_output_from_ham(fname,k_vect,ham)
    c,z  = get_topology(fname,ham_type="general",surfaces=bhz_edge_surfaces,n=n)
    #z[0]=strong invariant index
    #z[1:]=weak invariant indices
    print("3D z2 invariant for 3D BHZ model ", z)

    
    #try calculation on sample Yaehmop file
    c, z2_ = get_topology("C:/Users/danpa/Downloads/22.out")
    #note: kpoints in z2pack and this Yaehmop output file likely do not match up  
    #in this case, so will not produce a reliable result. just shows that interface works
    print("3D z2 invariant ", z2_)
    