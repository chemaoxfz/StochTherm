# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 08:50:44 2015

@author: chemaoxfz
"""
from thermoSim import *
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
from multiprocessing import Pool

def script1():
    #short script for running simulations for distribution of post-collision velocities.
    a=[0.5,0.5]
    fN='exp_molecule_'+str(a[0])+'_'+str(a[1])
#    exp_pingpongOne(fN,N=1e6,a=a)
    plot_molecule_V(fN)
    
def script_core(runName='exp_core'):
    ####### Set Parameter List #######
    b_list=[[.1,.1],[0.2,0.2],[.3,.3],[.4,.4],[0.5,0.5],[.6,.6],[.7,.7],[.8,.8],[.9,.9]]
    m_core_list=[[1.,.125],[1.,.25],[1.,.5],[1.,1.],[1.,2.],[1.,4.],[1.,8.],[1.,16.],[1.,32.],[1.,64.],[1.,128.]]
    T_list=[[.001,1.],[.01,1.],[.1,1.],[1.,1.],[10.,1.],[100.,1.],[1000.,1.]]
#    b_list=[[0.7,0.7]]
#    m_core_list=[100.]
    N=1e6
    b_canon=[.5,.5]
    m_core_canon=[1.,16.]
    T_canon=[.1,1.]
    
    # Create args list
    fN_gen=lambda args:'exp_core_b_'+str(args[0][0])+'_'+str(args[0][1])+'_m_'+str(args[1][0])+'_'+str(args[1][1])+'_T_'+str(args[2][0])+'_'+str(args[2][1])+'_N_'+str(N)
    rep_gen=lambda canon,lst: [canon for i in xrange(len(lst))]
    axis_config=lambda arr: np.rollaxis(np.swapaxes(arr,0,2),1)
    l_t=lambda l:map(list,zip(*l))
    m_core_args=l_t([rep_gen(b_canon,m_core_list), m_core_list, rep_gen(T_canon,m_core_list)])
    b_args=l_t([b_list,rep_gen(m_core_canon,b_list),rep_gen(T_canon,b_list)])
    T_args=l_t([rep_gen(b_canon,T_list),rep_gen(m_core_canon,T_list),T_list])
    args_param=b_args+m_core_args+T_args
    args_fN=map(fN_gen,args_param)
    [x.insert(0,fN) for fN,x in zip(args_fN,args_param)]
    args_list=args_param 
    pickle.dump({'args_list':args_list,'N':N},open(runName+'.p','wr'))
    
    # run to get results
    pool=Pool(len(args_list))
    results=pool.map(exp_coreOne_star,args_list)
    
    # plot results
    

def exp_coreOne_star(ar):
    kw={'N':1e6,'b':ar[1],'m_core':ar[2],'T':ar[3]}
    
    return exp_coreOne(ar[0],**kw)
 
def script_core_plot():
#    for i in xrange(len(m_core_list)):
#        m_core=m_core_list[i]
#        fN_compare='exp_core_drift_V_compare_b_fix_m_core_'+str(m_core)
#        plot_core_V_drift_comparison(fN_compare,[fN_list[i],fN_list[i+len(b_list)],fN_list[i+2*len(b_list)]])
#
#    for i in xrange(len(b_list)):
#        b=b_list[i]
#        fN_compare='exp_core_drift_V_compare_m_core_fix_b_'+str(b[0])+'_'+str(b[1])
#        plot_core_V_drift_comparison(fN_compare,fN_list[i*len(m_core_list):(i+1)*len(m_core_list)])
        
#    for i in xrange(len(m_core_list)):
#        m_core=m_core_list[i]
#        fN_compare='exp_core_tau_ditr_compare_b_fix_m_core_'+str(m_core)
#        plot_tau_distr_comparison(fN_compare,[fN_list[i],fN_list[i+len(b_list)],fN_list[i+2*len(b_list)]])

#    for i in xrange(len(b_list)):
#        b=b_list[i]
#        fN_compare='exp_core_tau_distr_compare_m_core_fix_b_'+str(b[0])+'_'+str(b[1])
#        plot_tau_distr_comparison(fN_compare,fN_list[i*len(m_core_list):(i+1)*len(m_core_list)])
    pass

def exp_pingpongOne(fN,N=1e6,a=[1,1]):
    params={'m':[1.,0.,0.],'T':[1e-1,1.],'L':1.,'D':0.,'d':0.,'a':a,'b':[0.,0.],
            'kb':1.,'force':[0.,0.,0.],'accelFlag':'none'}    
    pt=geometricPt(params=params)
    [pt.step() for n in xrange(int(N))]
    ptDict={'t':np.array(pt.t),'xt':np.array(pt.xt),'vt':np.array(pt.vt),'wt':np.array(pt.wt),'params':params,'mu':np.array(pt.mu)}
    pickle.dump(ptDict,open(fN+'.p','wr'))

def exp_coreOne(fN,N=1e6,b=[1.,1.],m_core=[1.,100.],T=[.1,1.]):
    params={'m':[m_core[0],m_core[1],0.],'T':T,'L':1.,'D':1e-7,'d':0.,'a':[0.5,0.5],'b':b,
            'kb':1.,'force':[0.,0.,0.],'accelFlag':'none'}
    pt=geometricPt(params=params)
    [pt.step() for n in xrange(int(N))]

    result=summaryCalc_core(pt)
    return result

def summaryCalc_core(pt):
    #Calculate the following: tau, dist, v_drift, ht_rate, ep_rate
    result=dict.fromkeys(['tau','dist','v_drift','ht_rate','ep_rate']

    

    #Calculate coarse-graining distribtion results
    # given del_t, a coarse-graining time scale, what is the distribution of v_drift, ht_rate, and ep_rate?
    coarsen_fold=[1,10,100]
    result2=dict.fromkeys(['del_t','v_drift','ht_rate','ep_rate'])



def plot_molecule_V(fN):
    # distribution of velocities of the molecule
    ptd=pickle.load(open(fN+'.p','r'))
    
    wt=ptd['wt']
    t=ptd['t']
    xt=ptd['xt'].T[0]
    vt=ptd['vt'].T[0]
    params=ptd['params']
    a=params['a']
    Tc=params['T'][0]
    Th=params['T'][1]
    mask_leftWall=wt==0
    mask_rightWall=wt==1
    v_leftWall=vt[mask_leftWall]
    v_rightWall=vt[mask_rightWall]
    t_leftWall=t[mask_leftWall]
    t_rightWall=t[mask_rightWall]    
    
    
    window=10000 #only consider data points beyond this <<<<<< later change this to time based
    
    suffix='postCollisionV'    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    freq, bin_edges=np.histogram(v_leftWall[window:],bins=100,normed=True)
    cenc=(bin_edges[:-1]+bin_edges[1:])/2
    ax.plot(cenc,freq,'-b',lw=2,alpha=0.3,label='cold wall sim')
    freq, bin_edges=np.histogram(-v_rightWall[window:],bins=100,normed=True)
    cenc=(bin_edges[:-1]+bin_edges[1:])/2
    ax.plot(cenc,freq,'-r',lw=2,alpha=0.3, label='hot wall sim')
    ax.set_ylabel('probability density')
    ax.set_xlabel('velocity after collision')
    ax.set_title(suffix+', alpha='+str(a))
#    ax.set_xlim([0,3.5])
    x=np.linspace(0,max(np.abs(vt)),1000)    
    Tc=1e-1
    Th=1
    MBfun=lambda T: MBp(x,params['m'][0],T)
    pdf1=MBfun(Tc)
    pdf2=MBfun(Th)
#    pdf3=(a[0]*MBfun(Tc)+a[1]*(1-a[0])*MBfun(Th))/(a[0]+a[1]-a[1]*a[0])
    ax.plot(x,pdf1,'-b',lw=0.5, label='cold wall MB')
    ax.plot(x,pdf2,'-r',lw=0.5, label='hot wall MB')
#    ax.plot(x,pdf3,'-k',lw=0.5,label='theoretical prediction')
    plt.legend()
    plt.savefig(fN+'_'+suffix+'.pdf')
    plt.show()
#    pdb.set_trace()    
    
    suffix='spacialV'    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    minV=min(vt)
    maxV=max(vt)
    nBin=100
    bin_edges=np.arange(minV,maxV,(maxV-minV)/nBin)
    freq=[]
    t_denom=t[-1]-t[window]
    vt_effective=vt[window:]
    t_effective=np.diff(t)[window-1:]
    for i in xrange(nBin-1):
        idx=np.where((vt_effective<bin_edges[i+1])*(vt_effective>=bin_edges[i]))[0]
        prob=np.sum(t_effective[idx])/t_denom
        freq.append(prob)
    
    cenc=(bin_edges[:-1]+bin_edges[1:])/2
    ax.plot(cenc,freq,'-k',lw=2,alpha=0.3,label='spacial V sim')
    ax.set_ylabel('probability density')
    ax.set_xlabel('time weighted velocity')
    ax.set_title(suffix+', alpha='+str(a))
#    ax.set_xlim([0,3.5])
#    x=np.linspace(0,max(np.abs(vt)),1000)    
#    Gaussianfun=lambda T: Gaussianp(x,params['m'][0],T)
#    pdf1=MBfun(Tc)
#    pdf2=MBfun(Th)
#    pdf3=(a[0]*MBfun(Tc)+a[1]*(1-a[0])*MBfun(Th))/(a[0]+a[1]-a[1]*a[0])
#    ax.plot(x,pdf1,'-b',lw=0.5, label='cold wall Gaussian')
#    ax.plot(x,pdf2,'-r',lw=0.5, label='hot wall Gaussian')
#    ax.plot(x,pdf3,'-k',lw=0.5,label='theoretical prediction')
    plt.legend()
    plt.savefig(fN+'_'+suffix+'.pdf')
    plt.show()

def plot_core_V_drift(fN):
    # 1. time-averaged velocity of the core as time increases
    
    
    ptd=pickle.load(open(fN+'.p','r'))
    t=ptd['t']
    vt=np.array(ptd['vt'].T[1])

    init_cutoff=100   
    t=t[init_cutoff:]
    vt=vt[init_cutoff:]
    
    suffix='core_driftV'
    fig=plt.figure()
    ax=fig.add_subplot(111)
    v_bar=np.cumsum(vt)/t
    t_diff=np.diff(t)
    v_std=[np.sum(t_diff[:i+1]*(vt[:i+1]-v_bar[i])**2/t[i])**(0.5) for i in xrange(len(v_bar)-1)]
    ax.plot(t,v_bar,'-k',lw=2,label='dift core V')
    ax.fill_between(t[:-1],v_bar[:-1]-v_std,v_bar[:-1]+v_std,alpha=0.3)
    ax.set_ylabel('velocity')
    ax.set_xlabel('time')
    ax.set_title(suffix)
    plt.legend()
    plt.savefig(fN+'_'+suffix+'.pdf')   


def plot_core_V_drift_comparison(fN_compare,fN_list):
    # 1. time-averaged velocity of the core as time increases
    # compare multiple fN
    
    suffix='core_driftV_comparison'
    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    for fN in fN_list:
        ptd=pickle.load(open(fN+'.p','r'))
        t=ptd['t']
        vt=np.array(ptd['vt'].T[1])/ptd['mu'][1]
    
        init_cutoff=10000   
        t=t[init_cutoff:]
        vt=vt[init_cutoff:]
        v_bar=np.cumsum(vt)/t
        t_diff=np.diff(t)
#        v_std=[np.sum(t_diff[:i+1]*(vt[:i+1]-v_bar[i])**2/t[i])**(0.5) for i in xrange(len(v_bar)-1)]
        ax.plot(t,v_bar,lw=2,label=fN)
#        ax.fill_between(t[:-1],v_bar[:-1]-v_std,v_bar[:-1]+v_std,alpha=0.3)
    ax.set_ylabel('velocity')
    ax.set_xlabel('time')
    ax.set_title(suffix)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,.102),ncol=1,mode='expand',borderaxespad=0.)
    plt.savefig(fN_compare+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_core_V_sliding(fN):
    # 2. sliding-window average of velocity the core
    # not very useful it seems
    ptd=pickle.load(open(fN+'.p','r'))
    t=ptd['t']
    vt=ptd['vt'].T[1]/ptd['mu'][1]

    init_cutoff=100   
    t=t[init_cutoff:]
    vt=vt[init_cutoff:]
    
    
    suffix='core_driftV_sliding_window'
    fig=plt.figure()
    ax=fig.add_subplot(111)
    window_width=1000
    v_bar=np.array([np.sum(vt[i:i+window_width])/(t[i+window_width]-t[i]) for i in xrange(len(vt)-window_width)])
    t_diff=np.diff(t)
    v_std=np.array([np.sum(t_diff[i:i+window_width]*(vt[i:i+window_width]-v_bar[i])**2/(t[i+window_width]-t[i]))**(0.5) for i in xrange(len(v_bar)-1)])
#    pdb.set_trace()
    ax.plot(t[window_width:],v_bar,'-k',lw=2,label='dift core V sliding')
    ax.fill_between(t[window_width:-1],v_bar[:-1]-v_std,v_bar[:-1]+v_std,alpha=0.3)
    ax.set_ylabel('velocity sliding')
    ax.set_xlabel('time')
    ax.set_title(suffix)
    plt.legend()
    plt.savefig(fN+'_'+suffix+'.pdf') 

def plot_tau_distr(fN):
    # tau is time between two consecutive collisions between molecule and frame
    # we plot distribution of tau for all times
    ptd=pickle.load(open(fN+'.p','r'))
    init_cutoff=10000 
    t=ptd['t'][init_cutoff:]
    wt=ptd['wt'][init_cutoff:]
    # w=8,9,12,13 corresponds to collision between molecule and frame
    mask=np.where(np.logical_or(np.logical_or(np.logical_or(wt==8,wt==9),wt==12),wt==13))[0]
    t_hit=t[mask]
    tau=np.diff(t_hit)
    
    suffix='tau'    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    nBin=100
    
    freq, bin_edges=np.histogram(tau,bins=nBin,normed=False,range=[0,20])
    cenc=(bin_edges[:-1]+bin_edges[1:])/2
    ax.plot(cenc,freq,'-b',lw=2,label='tau')
    ax.set_ylabel('probability density')
    ax.set_xlabel('time between collision')
    ax.set_title(suffix)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,.102),ncol=1,mode='expand',borderaxespad=0.)
    plt.savefig(fN+'_'+suffix+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_tau_distr_comparison(fN_compare,fN_list):
    suffix='tau_distr_comparison_closer'
    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    nBin=100
    rg=[0,5]
    for fN in fN_list:
        init_cutoff=10000 
        ptd=pickle.load(open(fN+'.p','r'))
        t=ptd['t'][init_cutoff:]
        wt=ptd['wt'][init_cutoff:]
        mask=np.where(np.logical_or(np.logical_or(np.logical_or(wt==8,wt==9),wt==12),wt==13))[0]
        t_hit=t[mask]
        tau=np.diff(t_hit)
        freq, bin_edges=np.histogram(tau,bins=nBin,normed=False,range=rg)
        cenc=(bin_edges[:-1]+bin_edges[1:])/2
        ax.plot(cenc,freq,lw=2,label=fN)
    ax.set_ylabel('prob density')
    ax.set_xlabel('tau')
    ax.set_title(suffix)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,.102),ncol=1,mode='expand',borderaxespad=0.)
    plt.savefig(fN_compare+'_'+suffix+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_core_V(fN):
    # distribution of velocities of the core
    ptd=pickle.load(open(fN+'.p','r'))
    
    wt=ptd['wt']
    t=ptd['t']
    xt=ptd['xt'].T[1]
    vt=ptd['vt'].T[1]
    params=ptd['params']
    a=params['a']
    Tc=params['T'][0]
    Th=params['T'][1] 
    
    
    window=10000 #only consider data points beyond this <<<<<< later change this to time based
    
    suffix='coreV'    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    minV=min(vt)
    maxV=max(vt)
    nBin=100
    bin_edges=np.arange(minV,maxV,(maxV-minV)/nBin)
    freq=[]
    t_denom=t[-1]-t[window]
    vt_effective=vt[window:]
    t_effective=np.diff(t)[window-1:]
    for i in xrange(nBin-1):
        idx=np.where((vt_effective<bin_edges[i+1])*(vt_effective>=bin_edges[i]))[0]
        prob=np.sum(t_effective[idx])/t_denom
        freq.append(prob)
    
    cenc=(bin_edges[:-1]+bin_edges[1:])/2
    ax.plot(cenc,freq,'-k',lw=2,alpha=0.3,label='spacial V sim')
    ax.set_ylabel('probability density')
    ax.set_xlabel('time weighted core velocity')
    ax.set_title(suffix)
    plt.legend()
    plt.savefig(fN+'_'+suffix+'.pdf')
    plt.show()
    
def plot_molecule_pos(fN):
    # spacial distribution of the molecule
    ptd=pickle.load(open(fN+'.p','r'))
    
    wt=ptd['wt']
    t=ptd['t']
    xt_m=ptd['xt'].T[0]
    xt_c=ptd['xt'].T[1]
    vt_m=ptd['vt'].T[0]
    vt_c=ptd['vt'].T[1]
    params=ptd['params']
    a=params['a']
    Tc=params['T'][0]
    Th=params['T'][1]
    mask_leftWall=wt==0
    mask_rightWall=wt==1
    
    suffix='posDensity'
    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    minX=0.
    maxX=params['L']
    nBin=100
    bin_edges=np.arange(minV,maxV,(maxV-minV)/nBin)
    freq=[]
    t_denom=t[-1]-t[window]
    xt_effective=xt[window:]
    for i in xrange(nBin-1):
        idx=np.where((xt_effective<bin_edges[i+1])*(xt_effective>=bin_edges[i]))[0]
        prob=np.sum(t[idx])/t_denom
        freq.append(prob)
    pos=xt[window:]*np.diff(t)[window-1:]
    freq, bin_edges=np.histogram(pos,bins=100,normed=True)
    cenc=(bin_edges[:-1]+bin_edges[1:])/2
    ax.plot(cenc,freq,'-k',lw=2,alpha=0.3,label='pos density sim')
    ax.set_ylabel('probability density')
    ax.set_xlabel('position')
    ax.set_title(suffix+', alpha='+str(a))
    plt.legend()
    plt.savefig(fN+'_'+suffix+'.pdf')
    plt.show()


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
    stringList=('s','sr','ql')
#    pdb.set_trace()
    for i in xrange(len(stringList)):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(heatD['a'][1],heatD[stringList[i]],'-bo',lw=2)
        ax.set_ylabel(stringList[i])
        ax.set_xlabel('alpha')
        ax.set_xscale('log')
        if stringList[i]=='sr' :
            num=100
            alpha1=np.linspace(heatD['a'][0][0],heatD['a'][0][-1],num)
            alpha2=np.linspace(heatD['a'][1][0],heatD['a'][1][-1],num)
            T1=heatD['params'][0]['T'][0]
            T2=heatD['params'][0]['T'][1]
            sr_p=srPred(alpha1,alpha2,T1,T2)
            ax.plot(alpha2,sr_p,'-k',lw=1)
        if stringList[i]=='s':
            num=100
            alpha1=np.linspace(heatD['a'][0][0],heatD['a'][0][-1],num)
            alpha2=np.linspace(heatD['a'][1][0],heatD['a'][1][-1],num)
            T1=heatD['params'][0]['T'][0]
            T2=heatD['params'][0]['T'][1]
            sr_p=srPred(alpha1,alpha2,T1,T2)
            data=heatD[stringList[i]]
            const=data[-1]/sr_p[-1]
            pred=sr_p*const
            ax.plot(alpha2,pred,'-k',lw=1)
        if stringList[i]=='ql':
            num=100
            alpha1=np.linspace(heatD['a'][0][0],heatD['a'][0][-1],num)
            alpha2=np.linspace(heatD['a'][1][0],heatD['a'][1][-1],num)
            T1=heatD['params'][0]['T'][0]
            T2=heatD['params'][0]['T'][1]
            pred=heatPred(alpha1,alpha2,T1,T2)
            data=heatD[stringList[i]]
            pred=pred*data[-1]/pred[-1]
            ax.plot(alpha2,pred,'-k',lw=1)
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
#    scriptHeat2()
    script_core()
#    plot_core_V_drift('exp_core_0.7_0.7_m_100.0_N_100000.0')
#    plot_tau_distr('exp_core_0.7_0.7_m_100.0_N_100000.0')
#    script1()

#    fN='experimentHeat'
#    plotExperimentHeat(fN)
   
   
   
   
   
