# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 08:50:44 2015

@author: chemaoxfz
"""
from thermoSim import *
import matplotlib.pyplot as plt
import cPickle as pickle

def experimentOne(fN,N=1e6,a=[1,1]):
    params={'m':[1.,0.,0.],'T':[1e-1,1.],'L':1.,'D':0.,'d':0.,'a':a,'b':[0.,0.],
            'kb':1.,'force':[0.,0.,0.],'accelFlag':'none'}    
    pt=geometricPt(params=params)
    [pt.step() for n in xrange(int(N))]
    ptDict={'t':np.array(pt.t),'vt':np.array(pt.vt).T[0],'wt':np.array(pt.wt),'params':params}
    pickle.dump(ptDict,open(fN+'.p','wr'))
    
    
def plotExperiment(fN,a):
    ptd=pickle.load(open(fN+'.p','r'))
    wt=ptd['wt']
    t=ptd['t']
    vt=ptd['vt']
    params=ptd['params']
    mask_leftWall=wt==0
    mask_rightWall=wt==1
    v_leftWall=vt[mask_leftWall]
    v_rightWall=vt[mask_rightWall]
    t_leftWall=t[mask_leftWall]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    window=10000
    freq, bin_edges=np.histogram(v_leftWall[window:],bins=100,normed=True)
    cenc=(bin_edges[:-1]+bin_edges[1:])/2
    ax.plot(cenc,freq,'-b',lw=2,alpha=0.3)
    freq, bin_edges=np.histogram(-v_rightWall[window:],bins=100,normed=True)
    cenc=(bin_edges[:-1]+bin_edges[1:])/2
    ax.plot(cenc,freq,'-r',lw=2,alpha=0.3)
    ax.set_ylabel('probability density')
    ax.set_xlabel('velocity after collision with left wall')
    ax.set_xlim([0,3.5])
    x=np.linspace(0,max(np.abs(vt)),1000)    
    Tc=1e-1
    Th=1
    MBfun=lambda T: MBp(x,params['m'][0],T)
    pdf1=MBfun(Tc)
    pdf2=MBfun(Th)
    pdf3=(a[0]*MBfun(Tc)+a[1]*(1-a[0])*MBfun(Th))/(a[0]+a[1]-a[1]*a[0])
    ax.plot(x,pdf1,'-b',lw=0.5)
    ax.plot(x,pdf2,'-r',lw=0.5)
    ax.plot(x,pdf3,'-k',lw=0.5)
    plt.savefig(fN+'.pdf')
    plt.show()
#    pdb.set_trace()    
    
def MBp(x,m,T,kb=1):
    return m/(kb*T)*x*np.exp(-m*x**2/(2*kb*T))

if __name__ == "__main__":
    a=[0.3,0.3]
    fN='experimentOne_'+str(a[0])+'_'+str(a[1])
#    experimentOne(fN,N=1e6,a=a)
    plotExperiment(fN,a)