# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:19:05 2018

@author: nshams
"""
import numpy as np 
import scipy as sp
from scipy import signal as signal
from eigen_sort import eigen_sort   

def pls_kcv(X,Y,var_norm=1,Nperm=200,pval=1.00,pMatch=1,Nsplit=500,testSize=0.25):
    
    Nsub = np.size(X,axis = 0)
    
    if var_norm == 1 :
        Xfull = signal.detrend(X, axis=0, type='constant')
        Yfull = signal.detrend(Y, axis=0, type='constant')
    
    elif var_norm == 2 :
        Yfull = sp.stats.zscore(Y, axis=0)
        Xfull = sp.stats.zscore(X, axis=0)
        
    
    #=================== initial svd ===========================
    Ux, Dx, Vx = np.linalg.svd(Xfull,full_matrices=False)
    Uy, Dy, Vy = np.linalg.svd(Yfull,full_matrices=False)
    Umix, Dmix, Vmix = np.linalg.svd(np.diag(Dx.T) @ Ux.T @ Uy @ np.diag(Dy),full_matrices=False)
    salience_x = Vx.T @ Umix
    salience_y = Vy.T @ Vmix
    del Vx, Vy
    Dpct = Dmix*100//sum(Dmix)
    
    # -------------permutation test-----------------
    Dperm = np.zeros([Nperm,Dmix.size])
    for ii in range(Nperm):
        iperm = np.random.permutation(Nsub)
        Umix, Dmix, Vmix = np.linalg.svd(np.diag(Dx.T) @ Ux[iperm,:].T @ Uy @ np.diag(Dy),full_matrices=False)
        Dperm[ii,:] = Dmix.T
        
    pSig = np.mean(((np.tile(Dmix,(Nperm,1)) > Dperm)*1),0)
    iSig = np.where(pSig >1-pval)[0]
    Ncomp = iSig.size
    
    # discarding non-significant components
    salience_x = salience_x[:,iSig]
    salience_y = salience_y[:,iSig]
    
    #------------------------------------------------
    
    latent_x = X @ salience_x
    latent_y = Y @ salience_y
        
    # -------------validtion----------------- 
    Ntest = int(np.round(Nsub*testSize))
    salience_x_cv =  np.zeros([salience_x.shape[0],Ncomp,Nsplit])
    salience_y_cv =  np.zeros([salience_y.shape[0],Ncomp,Nsplit])
    latent_xo_cv =np.zeros([Ntest,Ncomp,Nsplit])
    latent_yo_cv =np.zeros([Ntest,Ncomp,Nsplit])
    latent_corr_cv = np.zeros([Ncomp,Nsplit])
    
    M_salience_x = np.zeros(salience_x.shape)
    M_salience_y = np.zeros(salience_y.shape)
    V_salience_x = np.zeros(salience_x.shape) 
    V_salience_y = np.zeros(salience_y.shape)
   

    for ii in range(Nsplit):
        iperm = np.random.permutation(Nsub)
        test_idx = iperm[:Ntest]
        train_idx = iperm[Ntest:]

        Xs = X[train_idx,:]
        Ys = Y[train_idx,:]
        Xo = X[test_idx,:]
        Yo = Y[test_idx,:]
        
        Xm = np.mean(Xs,axis = 0)
        Xstd = np.std(Xs,axis = 0)
        Ym = np.mean(Ys, axis = 0)
        Ystd= np.std(Ys, axis = 0)
        
        if var_norm == 1 :
            Xs = sp.signal.detrend(Xs, axis=0, type='constant')
            Ys = sp.signal.detrend(Ys, axis=0, type='constant')
            #Xo = Xo - Xm
            #Yo = Yo - Ym
            Xo = sp.signal.detrend(Xo, axis=0, type='constant')
            Yo = sp.signal.detrend(Yo, axis=0, type='constant')
            
        elif var_norm == 2 :
            Ys = sp.stats.zscore(Ys, axis=0)
            Xs = sp.stats.zscore(Xs, axis=0)
            #Yo = (Yo-Ym)/Ystd
            #Xo = (Xo-Xm)/Xstd
            Yo = sp.stats.zscore(Yo, axis=0)
            Xo = sp.stats.zscore(Xo, axis=0)
            
      

        Uxs, Dxs, Vxs = np.linalg.svd(Xs,full_matrices=False)
        Uys, Dys, Vys = np.linalg.svd(Ys,full_matrices=False)
        Umixs, Dmixs, Vmixs = np.linalg.svd(np.diag(Dxs.T) @ Uxs.T @ Uys @ np.diag(Dys),full_matrices=False)
        salience_xs = Vxs.T @ Umixs
        salience_ys= Vys.T @ Vmixs
        del Vxs, Vys
        
        if pMatch == 1 :
            ind, sgn = eigen_sort(salience_x,salience_xs)
        else :
            ind, sgn = eigen_sort(salience_y,salience_ys)
            
        salience_xs = salience_xs[:,ind] @ np.diag(sgn)
        salience_ys = salience_ys[:,ind] @ np.diag(sgn)
        #salience_x_cv[:,:,ii] = salience_xs
        #salience_y_cv[:,:,ii] = salience_ys
        M_salience_x = M_salience_x + salience_xs
        M_salience_y = M_salience_y + salience_ys
        V_salience_x = V_salience_x + np.power(salience_xs,2)
        V_salience_y = V_salience_y + np.power(salience_ys,2)
        
        
        latent_xo_cv[:,:,ii] = Xo @ salience_xs
        latent_yo_cv[:,:,ii] = Yo @ salience_ys
        C = np.corrcoef(latent_xo_cv[:,:,ii],latent_yo_cv[:,:,ii])[:Ncomp,-Ncomp:]
        latent_corr_cv[:,ii] =np.diag(C)
        # corr between latent x and y
        
    #cacluate zscore on saliencex and y (test set)
    
    
    mu_salience_x = M_salience_x/Nsplit
    mu_salience_y = M_salience_y/Nsplit
    var_salience_x = V_salience_x/Nsplit - np.power(mu_salience_x,2)
    var_salience_y = V_salience_y/Nsplit - np.power(mu_salience_y,2)
    bse_salience_x = var_salience_x *(Nsplit/(Nsplit-1))
    bse_salience_y = var_salience_y *(Nsplit/(Nsplit-1))
    Zsalience_x_cv = mu_salience_x / np.sqrt(bse_salience_x)
    Zsalience_y_cv = mu_salience_y / np.sqrt(bse_salience_y)
    
    return Zsalience_x_cv,Zsalience_y_cv, latent_corr_cv, Dpct,pSig

#TODO : not using varnorm
