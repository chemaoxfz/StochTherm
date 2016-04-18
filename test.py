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
    
    
def plotExperimentOne(fN,a):
    # distribution of post-collision velocities.
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
    
def script1():
    #short script for running simulations for distribution of post-collision velocities.
    a=[0.3,0.3]
    fN='experimentOne_'+str(a[0])+'_'+str(a[1])
#    experimentOne(fN,N=1e6,a=a)
    plotExperimentOne(fN,a)

def MBp(x,m,T,kb=1):
    return m/(kb*T)*x*np.exp(-m*x**2/(2*kb*T))

def experimentHeat(fN,N=1e6,a=[[1.,1.],[0.5,0.5]]):
    a=np.array(a)
    qlrD=np.zeros(np.shape(a)[1])
    qrrD=np.zeros(np.shape(a)[1])
    srD=np.zeros(np.shape(a)[1])
    qlD=np.zeros(np.shape(a)[1])
    qrD=np.zeros(np.shape(a)[1])
    sD=np.zeros(np.shape(a)[1])
    paramsD=np.empty(np.shape(a)[1],dtype=object)
    for i in xrange(np.shape(a)[1]):
        params={'m':[1.,0.,0.],'T':[1e-1,1.],'L':1.,'D':0.,'d':0.,'a':a.T[i],'b':[0.,0.],
                'kb':1.,'force':[0.,0.,0.],'accelFlag':'none'}    
        pt=geometricPt(params=params)
        [pt.step() for n in xrange(int(N))]
#        aa=np.array(pt.heatTransferRate())
#        pdb.set_trace()
        qlrD[i], qrrD[i], srD[i],qlD[i],qrD[i],sD[i]=np.array(pt.heatTransferRate()).T[-1]
        paramsD[i]=params
    ptDict={'a':a,'qlr':qlrD,'qrr':qrrD,'sr':srD,'ql':qlD,'qr':qrD,'s':sD,'params':paramsD}
    pickle.dump(ptDict,open(fN+'.p','wr'))


def plotExperimentHeat(fN):
    # heat transfer as a function of alpha
    heatD=pickle.load(open(fN+'.p','r'))
    
    
#    stringList=('qlr','qrr','sr','ql','qr','s')
    stringList=('sr','ql')
    pdb.set_trace()
    for i in xrange(len(stringList)):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(heatD['a'][1],heatD[stringList[i]],'-bo',lw=2)
        ax.set_ylabel(stringList[i])
        ax.set_xlabel('alpha')
        ax.set_xscale('log')
        if stringList[i]=='sr':
            num=100
            alpha1=np.linspace(heatD['a'][0][0],heatD['a'][0][-1],num)
            alpha2=np.linspace(heatD['a'][1][0],heatD['a'][1][-1],num)
            T1=heatD['params'][0]['T'][0]
            T2=heatD['params'][0]['T'][1]
            sr_p=srPred(alpha1,alpha2,T1,T2)
            ax.plot(alpha2,sr_p,'-k',lw=5,alpha=0.4)
#            pdb.set_trace()
        if stringList[i]=='ql':
            num=100
            alpha1=np.linspace(heatD['a'][0][0],heatD['a'][0][-1],num)
            alpha2=np.linspace(heatD['a'][1][0],heatD['a'][1][-1],num)
            T1=heatD['params'][0]['T'][0]
            T2=heatD['params'][0]['T'][1]
            pred=heatPred(alpha1,alpha2,T1,T2)
            data=heatD[stringList[i]]
            pred=pred*data[-1]/pred[-1]
            ax.plot(alpha2,pred,'-k',lw=5,alpha=0.4)
        plt.savefig(fN+'_'+stringList[i]+'.pdf')
    plt.show()

def heatPred(alpha1,alpha2,T1,T2):
    kappa=1
    c=1-(1-alpha1)*(1-alpha2)
    
    result=alpha1*alpha2/(4*c)*(T2-T1)*kappa
    return result

def srPred(alpha1,alpha2,T1,T2):
    c=1-(1-alpha1)*(1-alpha2)
    ep0=alpha1*alpha2/(4*c)*(T2-T1)*(1/T1-1/T2)
    heat=heatPred(alpha1,alpha2,T1,T2)
    ep=heat/T1-heat/T2
    pdb.set_trace()
    return ep

def scriptHeat1():
    fN='experimentHeat'
    aa=np.linspace(0.,1.,20)
    aa=np.tile(aa,(2,1))
    experimentHeat(fN,N=1e6,a=aa)
    plotExperimentHeat(fN)

def scriptHeat2():
    #keeping a1=0.1 fixed, changing a2 so as to change a1:a2.
    #ratio from 1e-1 to 1e1, so a2 range from 0.01 to 1
    nn=20
    a1_default=0.1
    aa=np.logspace(-1,1,nn)*a1_default
    aa=np.concatenate(([np.ones(nn)*a1_default],[aa]))
    fN='experimentHeat_2'
#    experimentHeat(fN,N=1e6,a=aa)
    plotExperimentHeat(fN)


def experimentFrameCoreEquilibrium(fN,N=1e6):
    # generate a few routine graphs demonstrating the convergence of quantities such as velocity, heat-transfer-rate, entropy-production-rate
    params={'m':[0.01,0.5,0.5],'T':[1e-1,1.],'L':1.,'D':0.1,'d':0.01,'a':[0.5,0.5],'b':[0.5,0.5],
            'kb':1.,'force':[0.,0.,0.],'accelFlag':'none'}
    pt=geometricPt(params=params)
    [pt.step() for n in xrange(int(N))]
    _,_,_,ql,qr,s=pt.heatTransferRate()
    ptDict={'t':np.array(pt.t),'xt':np.array(pt.xt_noncyclic).T,'wt':np.array(pt.wt),'ql':ql,'qr':qr,'s':s,'params':params}
    pickle.dump(ptDict,open(fN+'.p','wr'))

def plotExperimentFrameCoreEquilibrium(fN):
    ptd=pickle.load(open(fN+'.p','r'))
    ptd['vt']=ptd['xt'][1]/ptd['t']
    ptd['qlr']=ptd['ql']/ptd['t']
    ptd['qrr']=ptd['qr']/ptd['t']
    ptd['sr']=ptd['s']/ptd['t']
    sample_window=10.
    sampler=lambda x:x[(np.arange(np.floor(len(x)/sample_window))*sample_window).astype('int')]
    stringList=('vt','qlr','qrr','sr','ql','qr','s')
    for i in xrange(len(stringList)):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(sampler(ptd['t']),sampler(ptd[stringList[i]]),'-b',lw=1)
        ax.set_ylabel(stringList[i])
        ax.set_xlabel('time')
        plt.savefig(fN+'_'+stringList[i]+'.pdf')
    plt.show()

def scriptFrameCoreEquilibrium():
    fN='experimentFrameCoreEquilibrium'
    experimentFrameCoreEquilibrium(fN,N=1e6)
    plotExperimentFrameCoreEquilibrium(fN)

def experimentFrameCore1(fN,N=1e6,massRatio=[1]):
    # mass-ratio effect on velocity, q-rate, s-rate. time-path graphs.
    resultsD=np.empty(len(massRatio),dtype=object)
    for i in xrange(len(massRatio)):
        params={'m':[0.01,0.5,0.5],'T':[1e-1,1.],'L':1.,'D':0.5,'d':0.25,'a':[0.5,0.5],'b':[0.5,0.5],
            'kb':1.,'force':[0.,0.,0.],'accelFlag':'none'}
        it=massRatio[i]
        params['m'][1]=it/(1.+it)
        params['m'][2]=1/(1.+it)
        pt=geometricPt(params=params)
        [pt.step() for n in xrange(int(N))]
        ql,qr,s=pt.heatTransfer()
        resultsD[i]={'t':np.array(pt.t),'xt':np.array(pt.xt_noncyclic).T,'wt':np.array(pt.wt),'ql':ql,'s':s,'params':params}
    
    pickle.dump(resultsD,open(fN+'.p','wr'))

def plotExperimentFrameCore1(fN):
    resultsD=pickle.load(open(fN+'.p','r'))

    sample_window=1.
    sampler=lambda x:x[(np.arange(np.floor(len(x)/sample_window))*sample_window).astype('int')]    
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    l=len(resultsD)
    colorList=np.linspace(0,1,l)
    colorList=np.concatenate(([colorList],[colorList],np.zeros([1,l]))).T
    for i in xrange(len(resultsD)):
        ptd=resultsD[i]
        ptd['t'][0]=1e-5
        color=colorList[i]
        mu=np.sqrt(ptd['params']['m'][1]/np.sum(ptd['params']['m']))
        ptd['vt']=ptd['xt'][1]/ptd['t']/mu
        ax.plot(sampler(ptd['t']),sampler(ptd['vt']),'-',color=color,lw=1)
    
    ax.set_ylabel('drift velocity')
    ax.set_xlabel('time')
    
    plt.savefig(fN+'_drift_V1.pdf')

    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    l=len(resultsD)
    colorList=np.linspace(0,1,l)
    colorList=np.concatenate(([colorList],np.zeros([2,l]))).T
    for i in xrange(len(resultsD)):
        ptd=resultsD[i]
        ptd['t'][0]=1e-5
        color=colorList[i]
        ptd['qlr']=ptd['ql']/ptd['t']
        ax.plot(sampler(ptd['t']),sampler(ptd['qlr']),'-',color=color,lw=1)
    
    ax.set_ylabel('qlr')
    ax.set_xlabel('time')
    
    plt.savefig(fN+'_qlr.pdf')


def scriptExperimentFrameCore1():
    fN='experimentFrameCore1'
    massRatio=np.logspace(-2,2,5)
#    experimentFrameCore1(fN,N=1e6,massRatio=massRatio)
    plotExperimentFrameCore1(fN)


def experimentFrameCoreQLR(fN,N=1e4,massRatio=[1]):
    # heat transfer rate as a function of mass-ratio
    qlrD=np.zeros(len(massRatio))
    srD=np.zeros(len(massRatio))
    paramsD=np.empty(len(massRatio),dtype=object)
    for i in xrange(len(massRatio)):
        params={'m':[0.01,0.5,0.5],'T':[1e-1,1.],'L':1.,'D':0.5,'d':0.25,'a':[0.5,0.5],'b':[0.5,0.5],
            'kb':1.,'force':[0.,0.,0.],'accelFlag':'none'}
        it=massRatio[i]
        params['m'][1]=it/(1.+it)
        params['m'][2]=1/(1.+it)
        pt=geometricPt(params=params)
        [pt.step() for n in xrange(int(N))]
        qlr,qrr,sr,ql,qr,s=pt.heatTransferRate()
        qlrD[i]=qlr[-1]
        srD[i]=sr[-1]
    resultsD={'qlr':qlrD,'sr':srD,'params':paramsD,'massRatio':massRatio}
    pickle.dump(resultsD,open(fN+'.p','wr'))

def plotExperimentFrameCoreQLR(fN):
    resultsD=pickle.load(open(fN+'.p','r'))
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(resultsD['massRatio'],resultsD['sr'],'-ko',lw=2)
    ax.set_ylabel('entropy production rate')
    ax.set_xlabel('mass ratio')
    ax.set_xscale('log')
    plt.savefig(fN+'_sr.pdf')

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(resultsD['massRatio'],resultsD['qlr'],'-ko',lw=2)
    ax.set_ylabel('heat transfer rate')
    ax.set_xlabel('mass ratio')
    ax.set_xscale('log')
    plt.savefig(fN+'_qlr.pdf')

def scriptExperimentFrameCoreQLR():
    fN='experimentFrameCoreQLR'
    massRatio=np.logspace(-2,2,20)
    experimentFrameCoreQLR(fN,N=1e6,massRatio=massRatio)
    plotExperimentFrameCoreQLR(fN)

def experimentFrameCore2(fN,N=1e6,massRatio=[1]):
    #quantity convergence rate for velocity.
    # compare this convergence rate with mass ratio. see what kind of functional relation it would be.
    pass

def experimentFrameCore3(fN,N=1e6,force=[1]):
    # time-path of velocity, q-rate, s-rate under different load.
    pass

def experimentFrameCore4(fN,N=1e6,force=[1]):
    # time-path of position, for equilibrium diffusion under same-temperature or force-compensated.
    pass

#then try to figure out a way to compute diffusion coefficient from these. Supposedly distribution of position? Velocity? 
# Yeah, velocity. velocity. Read gillespie article.
if __name__ == "__main__":
#    scriptExperimentFrameCoreQLR()
#   scriptExperimentFrameCore1()
    scriptHeat2()

#    fN='experimentHeat'
#    plotExperimentHeat(fN)
   
   
   
   
   