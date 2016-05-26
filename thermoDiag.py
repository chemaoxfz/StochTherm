# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:36:35 2016

@author: chemaoxfz
"""
from thermoSim import geometricPt
import numpy as np
import pdb
import matplotlib.pyplot as plt

def testConvergence(values):
    #test whether the sequence has converged.
    #values are 2-d arrays. rows are each run with different initial conditions. columns corresponds to different entries.
    
    cut=values.shape[1]/2
    split=np.zeros([values.shape[0]*2,cut])
    
    for i in xrange(values.shape[0]):
        split[i*2]=values[i][:cut]
        split[i*2+1]=values[i][cut:2*cut]

    m,n=split.shape
    mean_b=np.average(split,axis=1)
    mean_all=np.average(mean_b)
    var_b=n/(m-1)*np.sum((mean_b-mean_all)**2)

    temp=0. 
    for i in xrange(m):
        temp+=np.sum((split[i]-mean_b[i])**2)/(n-1)
    var_w=temp/m
    var_est=(n-1)*var_w/n+var_b/n
    r_est=np.sqrt(var_est/var_w)
    return r_est

#def testConvTime(values,t,T):
#    m,n=values.shape
#    val_t=values*t
#    mean_b=np.sum(val_t,axis=1)/T
#    mean_all=np.average(mean_b)
#    var_b=n/(m-1)*np.sum((mean_b-mean_all)**2)
#    temp=0.
#    for i in xrange(m):
#        temp+=(np.sum((values[i]-mean_b[i])**2*t)/T)
#    var_w=temp/m
#    var_est=asdf

def testConv_script(NN):
    NN=int(NN)
    warmup=NN/2
    params={'m':[1.,100.,0.],'T':[1e-1,1.],'L':1.,'D':1e-7,'d':0.,'a':[0.5,0.5],'b':[0.5,0.5],'kb':1.,'force':[0.,0.,0.],'accelFlag':'none'}
    pt1=geometricPt(params=params)
    pt2=geometricPt(params=params)
    for i in xrange(NN):
        pt1.step()
        pt2.step()
    rslt=np.zeros([2,NN-warmup])
    rslt[0]=np.array(pt1.vt).T[1][warmup+1:]
    rslt[1]=np.array(pt2.vt).T[1][warmup+1:]
    return testConvergence(rslt)

def plotConv(NN_list,simFunc,niters=10,fN='test'):
    l=len(NN_list)
    rslt=np.zeros([l,niters])
    for i in xrange(l):
        NN=NN_list[i]
        for j in xrange(niters):
            rslt[i][j]=simFunc(NN)
    
    mean=np.average(rslt,axis=1)
    std=np.std(rslt,axis=1)*np.sqrt(niters)
    ylabel='potential reduction factor'
    var_name='num iter'    
    xscale='linear'
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(NN_list,mean,'-o',color='blue',lw=2,label='mean')
    ax.fill_between(NN_list,mean-std,mean+std,color='purple',alpha=0.3,label='std')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(var_name)
    ax.set_xscale(xscale)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,.102),ncol=3,mode='expand',borderaxespad=0.)
    plt.savefig(fN+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    


if __name__ == "__main__":
#    testConv_script(NN=1e3)
    NN_list=np.linspace(10000,100000,5)
    plotConv(NN_list,testConv_script,niters=5,fN='test')
    