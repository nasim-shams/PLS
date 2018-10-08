# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:23:15 2018

@author: nshams
"""

import numpy as np 

def eigen_sort(S,Sb) :
    # format of the inputs: variables (rows) X components (columns)
    ind = np.zeros(Sb.shape[1])
    sgn = np.zeros(Sb.shape[1])
    #C = np.corrcoef(Sb.T,S.T)[:np.size(Sb,axis=1),-np.size(S,axis=1):]
    C = np.corrcoef(S.T,Sb.T)[:np.size(S,axis=1),-np.size(Sb,axis=1):]
    Csgn = np.sign(C)
    C = np.abs(C)
    #for i in range(np.size(C,axis=0)):
    for i in range(np.size(C,axis=1)):
        indices = np.where(C == C.max())
        ii = indices[0][0]
        jj = indices[1][0]
        ind[ii] = jj
        sgn[ii] = Csgn[ii,jj]
        C[:,jj] = -1
        C[ii,:] = -1
        
    ind.astype(int)
    sgn.astype(int)

    return ind.astype(int),sgn.astype(int)
    