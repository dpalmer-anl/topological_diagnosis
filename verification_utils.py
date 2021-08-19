# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:28:45 2021

@author: danpa
"""
import numpy as np

def finite_SSH_ham(num_dim,alpha,t1=-0.75):
    #create hamiltonian by hand
    num_dim+=1
    t2=-alpha*t1
    hamiltonian=np.zeros((num_dim,num_dim))
    for i in range(num_dim-1):
        if i%2==0:
            t=t1
        if i%2==1:
            t=t2
        hamiltonian[i,i+1]=t
        hamiltonian[i+1,i]=t
    return hamiltonian

def vary_alpha(num_unit_cells,alpha_, t1=-.75,glass=False):
    W,L= .5, num_unit_cells
    eigval_vect_dict={}
    eval_=[]
    evect_=[]
    for a in alpha_:
        if glass:
            r=np.random.normal(scale=0.1*abs(t1),size=2)
            ham=finite_SSH_ham(int(2*num_unit_cells),a+r[0],t1=r[1]+t1)
        else:
            ham=finite_SSH_ham(int(2*num_unit_cells),a)

        eigval,eigvec=np.linalg.eigh(ham)
        eval_.append(eigval)
        evect_.append(eigvec)
    eigval_vect_dict.update({"eigenvalues":np.array(eval_),"eigenvectors":np.array(evect_)})
    return eigval_vect_dict, ham