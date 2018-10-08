# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:19:05 2018

@author: nshams
"""
import numpy as np 
from eigen_sort import eigen_sort   

def pls(X,Y,var_norm=1,Nbs=500,Nperm=500,k=1,pval=1.00,pMatch=1):
    
    nsub = np.size(X,axis = 0)
    
    
    #--------------initial svd---------------
    Ux, Dx, Vx = np.linalg.svd(X,full_matrices=False)
    Uy, Dy, Vy = np.linalg.svd(Y,full_matrices=False)
    Umix, Dmix, Vmix = np.linalg.svd(np.diag(Dx.T) @ Ux.T @ Uy @ np.diag(Dy),full_matrices=False)
    salience_x = Vx.T @ Umix
    salience_y = Vy.T @ Vmix
    del Vx, Vy
    Dpct = Dmix*100//sum(Dmix)
    
    # -------------permutation test-----------------
    Dperm = np.zeros([Nperm,Dmix.size])
    for ii in range(Nperm):
        iperm = np.random.permutation(nsub)
        Umix, Dmix, Vmix = np.linalg.svd(np.diag(Dx.T) @ Ux[iperm,:].T @ Uy @ np.diag(Dy),full_matrices=False)
        Dperm[ii,:] = Dmix.T
        
    pSig = np.mean(((np.tile(Dmix,(Nperm,1)) > Dperm)*1),0)
    iSig = np.where(pSig >= 1-pval)[0]
    
    # discarding non-significant components
    salience_x = salience_x[:,iSig]
    salience_y = salience_y[:,iSig]
    
    
    #------------------------------------------------
    
    latent_x = X @ salience_x
    latent_y = Y @ salience_y
        
    # -------------bootsratp-----------------    
    
    M_salience_x = np.zeros(salience_x.shape)
    M_salience_y = np.zeros(salience_y.shape)
    R_salience_x = np.zeros(salience_x.shape) 
    R_salience_y = np.zeros(salience_y.shape)
   
        
    for b in range(Nbs):
        
        bsIdx= np.random.choice(range(nsub), size=nsub, replace=True)
        bUx, bDx, bVx = np.linalg.svd(X[bsIdx,:],full_matrices=False)
        bUy, bDy, bVy = np.linalg.svd(Y[bsIdx,:],full_matrices=False)
        bUmix, bDmix, bVmix = np.linalg.svd(np.diag(bDx.T) @ bUx.T @ bUy @ np.diag(bDy),full_matrices=False)
        bsalience_x = bVx.T @ bUmix
        bsalience_y = bVy.T @ bVmix
        
        if pMatch == 1 :
            ind, sgn = eigen_sort(salience_x,bsalience_x)
        else :
            ind, sgn = eigen_sort(salience_y,bsalience_y)
            
        bsalience_x = bsalience_x[:,ind] @ np.diag(sgn)
        bsalience_y = bsalience_y[:,ind] @ np.diag(sgn)
        
        M_salience_x = M_salience_x + bsalience_x
        M_salience_y = M_salience_y + bsalience_y
        R_salience_x = R_salience_x + np.power(bsalience_x,2)
        R_salience_y = R_salience_y + np.power(bsalience_y,2)
        
    mu_salience_x = M_salience_x/Nbs
    mu_salience_y = M_salience_y/Nbs
    var_salience_x = R_salience_x/Nbs - np.power(mu_salience_x,2)
    var_salience_y = R_salience_y/Nbs - np.power(mu_salience_y,2)
    bse_salience_x = var_salience_x *(Nbs/(Nbs-1))
    bse_salience_y = var_salience_y *(Nbs/(Nbs-1))
    zsalience_x = mu_salience_x / np.sqrt(bse_salience_x)
    zsalience_y = mu_salience_y / np.sqrt(bse_salience_y)
    
       
    return salience_x,salience_y, latent_x, latent_y, zsalience_x, zsalience_y, pSig, Dpct 

