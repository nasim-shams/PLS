# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import  scipy as sp
from pls import pls
from eigen_sort import eigen_sort  

def pls_loo(X,Y,var_norm=1,Nbs=500,Nperm=500,k=1):
    
    (nsuby,nY) = Y.shape
    (nsubx,nX) = X.shape
    
    if (nsubx != nsuby):
        print('Error: X and Y must have same number of rows')
        return -1
    else :
        nsub = nsubx
        
    ncomp = min(nY,nX,nsub-k)
        
    Xfull = X
    Yfull = Y
    XO = np.zeros([nsub,nX])
    YO = np.zeros([nsub,nY])
    
    for ii in range(nsub): 
        x = Xfull
        y = Yfull
        xo = Xfull[ii,] 
        yo = Yfull[ii,]
        XO[ii,]=xo;
        YO[ii,]=yo;
        x = np.delete(x, (ii), axis=0) 
        y = np.delete(y, (ii), axis=0)
        xm = np.mean(x,axis = 0);
        xstd = np.std(x,axis = 0);
        ym = np.mean(y, axis = 0);
        ystd= np.std(y, axis = 0);
        
        if var_norm == 1 :
            y = sp.signal.detrend(y, axis=0, type='constant')
            x = sp.signal.detrend(x, axis=0, type='constant')
            yo = yo - ym;
            xo = xo - xm;
        elif var_norm == 2 :
            y = sp.stats.zscore(y, axis=0)
            x = sp.stats.zscore(x, axis=0)
            yo = (yo-ym)/ystd
            xo = (xo-xm)/xstd
            
      
    pls_res = [dict() for ii in range(nsub)]

    for ii in range(nsub): 
      
        (pls_res[ii]["Salience_X"],pls_res[ii]["Salience_Y"],
         pls_res[ii]["latent_X"],pls_res[ii]["latent_Y"],
         pls_res[ii]["ZSalience_X"],pls_res[ii]["ZSalience_Y"],
         pls_res[ii]["Sig_prob"],pls_res[ii]["Dcorr_pct"]) = pls(x,y) 
        
        pls_res[ii]["latent_Yo"]  = np.dot(yo,pls_res[ii]["Salience_Y"])
        pls_res[ii]["latent_Xo"]  = np.dot(xo,pls_res[ii]["Salience_X"])
        

    if var_norm == 1 :
        Yfull = sp.signal.detrend(Yfull, axis=0, type='constant')
        Xfull = sp.signal.detrend(Xfull, axis=0, type='constant')
         
    elif var_norm == 2 :
        Yfull = sp.stats.zscore(Yfull, axis=0)
        Xfull = sp.stats.zscore(Xfull, axis=0)
        
    pls_full = dict() 
    
    (pls_full["Salience_X"],pls_full["Salience_Y"],
         pls_full["latent_X"],pls_full["latent_Y"],
         pls_full["ZSalience_X"],pls_full["ZSalience_Y"],
         pls_full["Sig_prob"],pls_full["Dcorr_pct"]) = pls(Xfull,Yfull)
    
    Salience_X_full_match = pls_full["Salience_X"][:,0:ncomp] # TODO: add condition to match on X/Y
    Salience_Y_full_match = pls_full["Salience_Y"][:,0:ncomp]  
    
    pls_sort = pls_res
    
    for ii in range(nsub):
        (ind, sgn) = eigen_sort(Salience_X_full_match,pls_res[ii]["Salience_X"]) 
        pls_sort[ii]["Salience_X"] = pls_res[ii]["Salience_X"][:,ind]*sgn
        pls_sort[ii]["Salience_Y"] = pls_res[ii]["Salience_Y"][:,ind]*sgn
        pls_sort[ii]["ZSalience_X"] = pls_res[ii]["ZSalience_X"][:,ind]*sgn
        pls_sort[ii]["ZSalience_Y"] = pls_res[ii]["ZSalience_Y"][:,ind]*sgn
        pls_sort[ii]["latent_X"] = pls_res[ii]["latent_X"][:,ind]*sgn
        pls_sort[ii]["latent_Y"] = pls_res[ii]["latent_Y"][:,ind]*sgn
        pls_sort[ii]["latent_Xo"] = pls_res[ii]["latent_Xo"][ind]*sgn
        pls_sort[ii]["latent_Yo"] = pls_res[ii]["latent_Yo"][ind]*sgn
        
    cv_ZSalience_X = np.zeros([pls_full["Salience_X"].shape[0],pls_full["Salience_X"].shape[1],nsub]) # nsub in each pls X ncomp X  number of CV loops
    cv_ZSalience_Y = np.zeros([pls_full["Salience_Y"].shape[0],pls_full["Salience_Y"].shape[1],nsub])
    
    
    for ii in range(nsub):
        cv_ZSalience_X[:,:,ii] = pls_sort[ii]["ZSalience_X"]
        cv_ZSalience_Y[:,:,ii] = pls_sort[ii]["ZSalience_Y"]
        
    avg_cv_ZSalience_X = np.mean(cv_ZSalience_X,2)
    avg_cv_ZSalience_Y = np.mean(cv_ZSalience_Y,2)
 
    pred_scores_X = [d["latent_Xo"] for d in pls_sort]
    pred_scores_Y = [d["latent_Yo"] for d in pls_sort]
    
   
    return avg_cv_ZSalience_X, avg_cv_ZSalience_Y, pred_scores_X, pred_scores_Y


    


     
    
    
    
         
       
       
       
      
       
       
       
    
       
            
            
            
            
            
        

        
        
    
    
    return;