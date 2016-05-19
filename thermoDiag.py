# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:36:35 2016

@author: chemaoxfz
"""
from thermoSim import geometricPt
import numpy as np
import pdb

def testConvergence(values):
    #test whether the sequence has converged.
    #values are 2-d arrays. rows are each run with different initial conditions. columns corresponds to different entries.
    
    cut=values.shape[1]/2
    split=np.zeros([values.shape[0]*2,cut])
    
    for i in xrange(values.shape[0]):
        split[i*2]=values[i][:cut]
        split[i*2+1]=values[i][cut:2*cut]
        
    m,n=split.shape
    mean_all=np.average(split)
    mean_b=np.average(split,axis=1)
    var_b=n/(m-1)*np.sum((mean_b-mean_all)**2)

    temp=0. 
    for i in xrange(m):
        temp+=(np.sum((split[i]-mean_b[i])**2)/2)
    var_w=temp/m
    var_est=(n-1)*var_w/n+var_b/n
    r_est=np.sqrt(var_est/var_w)
    pdb.set_trace()

def testConv_script(NN=1e4):
    NN=int(NN)
    warmup=NN/2
    params={'m':[1.,10000.,0.],'T':[1e-1,1.],'L':1.,'D':1e-7,'d':0.,'a':[0.5,0.5],'b':[0.5,0.5],'kb':1.,'force':[0.,0.,0.],'accelFlag':'none'}
    pt1=geometricPt(params=params)
    pt2=geometricPt(params=params)
    for i in xrange(NN):
        pt1.step()
        pt2.step()
    rslt=np.zeros([2,NN-warmup])
    rslt[0]=np.array(pt1.vt).T[1][warmup+1:]
    rslt[1]=np.array(pt2.vt).T[1][warmup+1:]
    testConvergence(rslt)
    


if __name__ == "__main__":
    testConv_script(NN=1e4)