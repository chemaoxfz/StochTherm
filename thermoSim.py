# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 07:53:06 2015

@author: xfz
"""

import numpy as np
from scipy import stats
import pdb
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

class geometricPt:
    def __init__(self,params={'m':[0.5,1.,1.],'T':[1e-1,1.],'L':1.,'D':0.5,'d':0.1,'a':[1,1],'b':[0.5,0.5],'kb':1.,'force':[0.,1.7e-1,0.],'accelFlag':'constant'}):
        # L is full length of space molecule lives in
        # 'D' is width of frame
        # 'd' is width of core
        # 'a' is probability that molecule's collision with cold and hot wall picks up velocity instead of total reflection
        # 'b' is probability that collision between molecule and frame results in elastic collision instead of just passing through
        self.kb=1 #boltzmann constant
        self.params=params
        D=params['D']
        d=params['d']
        m=params['m']
        L=params['L']
        #x1 is molecule, x2 is frame (upward), x3 is weight (towards us).
        # random initialization 
        x1=np.random.rand()*L
        self.x0=np.array([np.random.rand()*L,x1,x1+((np.random.rand()*2-1)*(D-d)/2)])
        lr=int(np.random.rand()*2)
        self.v0=np.array([-(lr*2-1)*self.MB1D(params['T'][lr],params['m'][0])[0],np.random.normal(loc=0.05,scale=0.1),np.random.normal(loc=0.05,scale=0.1)])
        
        self.t0=0.
        self.w0=lr # if v0[0] is positive, it's left wall, w0=0, if negative, right wall, w0=1.
        self.s0=0
        self.w_o0=self.w0
        
        mu=np.sqrt(m/np.sum(m)) # scale conversion to geometric point for specular reflection
        self.mu=mu
        if self.params['accelFlag']=='none':
            self.aFunc=[]
        elif 'force' not in params.keys():
            self.aFunc=self.accelFunc(flag=params['accelFlag'],arg=np.array(params['accel'])*self.mu)
        else: self.aFunc=self.accelFunc(flag=params['accelFlag'],arg=np.array(params['force'])/self.params['m']*self.mu)
        #Order of walls:L,R, Ur,Lr, F,B, FUr,FLr, FUrE,FLrE.
        self.xW=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[0,0,1],[0,0,0],[0,(D-d)/2,0],[0,0,(D-d)/2],[0,D/2,0],[D/2,0,0],[0,(D-d)/2,L],[0,L,(D-d)/2],[L,D/2,0],[D/2,L,0]])
        self.xW=mu*self.xW

        #normal vectors
        fUr=np.array([0,-mu[2]/np.sqrt(mu[2]**2+mu[1]**2),mu[1]/np.sqrt(mu[2]**2+mu[1]**2)])
        fUrE=np.array([-mu[1]/np.sqrt(mu[0]**2+mu[1]**2),mu[0]/np.sqrt(mu[0]**2+mu[1]**2),0])
        self.n=np.array([[1,0,0],[-1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1],fUr,-1*fUr,fUrE,-1*fUrE,fUr,-1*fUr,fUrE,-1*fUrE])

        # setting initial values with scale conversion.        
        self.t=[self.t0]
        self.xt=[mu*self.x0]
        self.xt_noncyclic=[mu*self.x0]
        self.vt=[mu*self.v0]
        self.wt=[self.w0]
        self.w_ommit=[self.w_o0]
        self.st=[self.s0]
        self.x_move=[]
    
    
    def accelFunc(self,flag='constant',arg=np.zeros(3)):
        if flag=='constant':
            return lambda: arg
          
    
    def step(self):
        '''
        one step we do the following
        '''
        x=self.xt[-1]
        v=self.vt[-1]
        w_o=self.w_ommit[-1]
        dt,wIdx,x_move,v_move=self.dtCalc(x,v,w_o)
        x_post,v_post,s,cc,w_o=self.event(wIdx,x_move,v_move)
        
        self.wt.append(wIdx)
        self.w_ommit.append(w_o)
        self.t.append(self.t[-1]+dt)
        self.xt.append(x_post)
        self.vt.append(v_post)
        self.st.append(s)
        self.x_move.append(x_move)
        self.xt_noncyclic.append(self.xt_noncyclic[-1]+x_move-x)
#        self.cyclicCounter.append(cc)
    
    def dtCalc(self,x,v,w_o):
        temp=np.zeros(len(self.n))
        temp[w_o]=1
        wmask = np.logical_not(temp)
        wrange=np.arange(len(self.n))[wmask]
        n=self.n[wmask]
        vproj=n.dot(v)
        possibleW=np.where(vproj<0)[0]
        if self.params['accelFlag']=='constant':
            dt,x_move,v_move,minIdx=self.timeConstForce(x,v,self.xW[wmask][possibleW],n[possibleW])
        elif self.params['accelFlag']=='none':
            dt,x_move,v_move,minIdx=self.timeNoForce(x,v,self.xW[wmask][possibleW],n[possibleW])
        else: print 'acceleration flag is wrong'
        wIdx=wrange[possibleW[minIdx]]
        return dt,wIdx,x_move,v_move
    
    def timeNoForce(self,x,v,xW,n):
        times=np.diag(n.dot((xW-x).T))/n.dot(v)
        tPos=np.where(times>0)[0]
#        if len(tPos)==0:
#            pdb.set_trace()
        minIdx=tPos[np.argmin(times[tPos])]
        return times[minIdx],x+v*times[minIdx],v,minIdx
    
    def timeConstForce(self,x,v,xW,n):
        accel=self.aFunc()
        vproj=n.dot(v)
        aproj=n.dot(accel)
        pproj=np.diag(n.dot((xW-x).T))
        aNZMask=(aproj!=0)
        times=np.empty(len(n))
        times[aNZMask]=(-vproj[aNZMask]-np.sqrt(vproj[aNZMask]**2+2*aproj[aNZMask]*pproj[aNZMask]))/aproj[aNZMask] #plus ac because here it's position of wall minus position of particle
        zeroMask=np.logical_not(aNZMask)
        times[zeroMask]=(pproj[zeroMask]/vproj[zeroMask])
        tPos=np.where(times>0)[0]
        minIdx=tPos[np.argmin(times[tPos])]
        dt=times[minIdx]
        x_move=x+v*dt+0.5*accel*dt**2
        v_move=v+accel*dt
        return dt,x_move,v_move,minIdx
    
    def event(self,wIdx,x,v):
        x=np.copy(x)
        v=np.copy(v)
        s=-1
        cc=-1
        w_o=wIdx #default is wIdx. It won't be wIdx only when it go through cyclic boundary
        if wIdx<=1:
            #L or R, total reflection, with chance a to pick up velocity
            r=np.random.rand()
            if r<self.params['a'][wIdx]:
                v[0]=self.MB1D(self.params['T'][wIdx],self.params['m'][0])*self.mu[0]*(-1)**wIdx
                s=1
            else:
                v[0]=-v[0]
                s=0
        elif wIdx<=5:
            #Ur,Lr,F, or B. Cylic boundary, continue on.
            cc=wIdx-2
            #if it's Ur(2): x[1]-L, Lr(3):x[1]+L; F(4):x[2]-L, B(5):x[2]+L
            x[wIdx/2]=x[wIdx/2]+self.params['L']*(-1)**(wIdx-1)*self.mu[wIdx/2]
            if wIdx%2 ==0: #Ur of F
                w_o=wIdx+1 #cyclic end up with Lr or B
            else: w_o=wIdx-1 #Lr of B, goes to Ur or F
            
        elif wIdx<=7 or wIdx==10 or wIdx==11:
            #FUr, FLr, collision between weight and frame.
            #Weight and frame has collision, which is simply reflection over the surface
            # i.e. reflect through normal vector
            v=v-2*self.n[wIdx].dot(v)*self.n[wIdx]
            
        elif wIdx<=9 or wIdx==12 or wIdx==13:
            #FUrE, FLrE, collision between molecule and frame
            r=np.random.rand()
            if r<self.params['b'][wIdx%2]:
                v=v-2*self.n[wIdx].dot(v)*self.n[wIdx]
                s=1
            else: s=0

        return x,v,s,cc,w_o
        
    def MB1D(self,T,m,n=1):
        unif=np.random.rand(n)
        x=np.sqrt(-2*self.kb*T/m*np.log(unif))
        return x
    
#    def cyclicToNoncyclic(self,x_cyclic,axis):
#        idx=axis-1
#        cc=np.array(self.cyclicCounter)
#        ccAddIdx=np.where(cc==idx*2)
#        ccDedIdx=np.where(cc==idx*2+1)
#        temp=np.zeros(len(x_cyclic))
#        temp[ccAddIdx]=1.
#        temp[ccDedIdx]=-1.
#        temp=np.cumsum(temp)
#        x=x_cyclic+temp*self.params['L']*self.mu[axis]
#        return x

    def plotFrame(self,axis=[0,1],cyclic=False,ommit=100):
        fig=plt.figure()
        if len(axis)<2:
            #time series plot of axis
        
            ax = fig.add_subplot(111)
            x=np.array(self.t)
            x=x[np.arange(int(len(x)/ommit))*ommit]
            yy=np.array(self.xt).T[axis[0]]
            y=yy[np.arange(int(len(yy)/ommit))*ommit]
            xlabel='t'
            ylabel='x'+str(axis[0])
            if axis[0]!=0:
                y=self.xt_noncyclic.T[axis[0]]
                y=y[np.arange(int(len(yy)/ommit))*ommit]
            plotAxis=axis
                
                
        elif len(axis)==2:
            ax = fig.add_subplot(111,aspect='equal')
            xx=np.array(self.xt).T[axis[0]]
            yy=np.array(self.xt).T[axis[1]]
            xlabel='x'+str(axis[0])
            ylabel='x'+str(axis[1])
        
            D=self.params['D']
            L=self.params['L']
            d=self.params['d']
            if axis==[0,1]:
                if cyclic==True:
                    x=xx[np.arange(int(len(xx)/ommit))*ommit]
                    y=yy[np.arange(int(len(yy)/ommit))*ommit]
                    
                    ax.plot(np.array([D/2,L])*self.mu[0],np.array([0,L-D/2])*self.mu[1],'-k',lw=2)
                    ax.plot(np.array([0,L-D/2])*self.mu[0],np.array([D/2,L])*self.mu[1],'-k',lw=2)
                    ax.plot(np.array([0,L])*self.mu[0],np.array([L,L])*self.mu[1],'-k',lw=2)
                    ax.plot(np.array([L,L])*self.mu[0],np.array([0,L])*self.mu[1],'-k',lw=2)
                    ax.plot(np.array([0,0]),np.array([0,L*self.mu[1]]),'-k',lw=2)
                    ax.plot(np.array([L*self.mu[0],0]),np.array([0,0]),'-k',lw=2)
                    #corner wall
                    ax.plot(np.array([0,D/2])*self.mu[0],np.array([L-D/2,L])*self.mu[1],'-k',lw=2)
                    ax.plot(np.array([L-D/2,L])*self.mu[0],np.array([0,D/2])*self.mu[1],'-k',lw=2)
                else:
                    x=xx #axis 0 is molecule, no cyclic boundary
                    y=self.cyclicToNoncyclic(yy,axis[1])
                    LX=L*self.mu[0]
                    LY=L*self.mu[1]
                    DY=D*self.mu[1]
                    maxYN=int(max(y)/LY)
                    minYN=int(min(y)/LY)
                    
                    for j in xrange(maxYN-minYN+2):
                        ax.plot([0,LX],[(j-1+minYN)*LY+DY/2,(j+minYN)*LY+DY/2],'-k',lw=2)
                        ax.plot([0,LX],[(j-1+minYN)*LY-DY/2,(j+minYN)*LY-DY/2],'-k',lw=2)
                plotAxis=axis
                
            elif axis==[2,1]:
                if cyclic==True:
                    x=xx[np.arange(int(len(xx)/ommit))*ommit]
                    y=yy[np.arange(int(len(yy)/ommit))*ommit]
                    
                    ax.plot(np.array([(D-d)/2,L])*self.mu[2],np.array([0,L-(D-d)/2])*self.mu[1],'-k',lw=2)
                    ax.plot(np.array([0,L-(D-d)/2])*self.mu[2],np.array([(D-d)/2,L])*self.mu[1],'-k',lw=2)
                    ax.plot(np.array([0,L])*self.mu[2],np.array([L,L])*self.mu[1],'-k',lw=2)
                    ax.plot(np.array([L,L])*self.mu[2],np.array([0,L])*self.mu[1],'-k',lw=2)
                    ax.plot(np.array([0,0]),np.array([0,L*self.mu[1]]),'-k',lw=2)
                    ax.plot(np.array([L*self.mu[2],0]),np.array([0,0]),'-k',lw=2)
                    #corner wall
                    ax.plot(np.array([0,(D-d)/2])*self.mu[2],np.array([L-(D-d)/2,L])*self.mu[1],'-k',lw=2)
                    ax.plot(np.array([L-(D-d)/2,L])*self.mu[2],np.array([0,(D-d)/2])*self.mu[1],'-k',lw=2)
                else:
                    x=self.cyclicToNoncyclic(xx,axis[0])
                    y=self.cyclicToNoncyclic(yy,axis[1])
                    LX=L*self.mu[2]
                    LY=L*self.mu[1]
                    maxXN=int(max(x)/(LX))
                    minXN=int(min(x)/(LX))
                    maxYN=int(max(y)/(LY))
                    minYN=int(min(y)/(LY))
                    mx=max(maxXN,maxYN)+1
                    mn=min(minXN,minYN)-1
                    ax.plot([mn*LX,mx*LX-(D-d)/2*self.mu[2]],[mn*LY+(D-d)/2*self.mu[1],mx*LY],'-k',lw=2)
                    ax.plot([mn*LX+(D-d)/2*self.mu[2],mx*LX],[mn*LY,mx*LY-(D-d)/2*self.mu[1]],'-k',lw=2)
                plotAxis=axis[::-1]
        else: print('ERROR,axis not the right length')
        
        return fig, ax,x,y,xlabel,ylabel,plotAxis
    
    def plotPt(self,axis=[0,1],cyclic=False,figName='figure',ommit=100):
        fig,ax,x,y,xlabel,ylabel,plotAxis=self.plotFrame(axis,cyclic=cyclic,ommit=ommit)

        colors=plt.cm.coolwarm(self.t/max(self.t))
        if self.params['accelFlag']=='constant':
            nPtsTraj=10
            if len(plotAxis)<2:
                isSimple=True
                if isSimple:
                    ax.plot(x,y,'b-',linewidth=1,alpha=0.5)
                else:
                    for ci in xrange(len(x)-1):
                        times=np.linspace(0,self.t[ci+1]-self.t[ci],nPtsTraj)
                        yLine=y[ci]+np.array(self.vt).T[plotAxis[0]][ci]*times+0.5*self.aFunc()[plotAxis[0]]*times**2
                        xLine=x[ci]+times
                        ax.plot(xLine,yLine,'b-',linewidth=1,alpha=0.5)
                        
            else:
                for ci in xrange(len(x)-1):
                    times=np.linspace(0,self.t[ci+1]-self.t[ci],nPtsTraj)
                    xLine=x[ci]+np.array(self.vt).T[plotAxis[0]][ci]*times+0.5*self.aFunc()[plotAxis[0]]*times**2
                    yLine=y[ci]+np.array(self.vt).T[plotAxis[1]][ci]*times+0.5*self.aFunc()[plotAxis[1]]*times**2
                    ax.plot(xLine,yLine,'b-',linewidth=1,alpha=0.5)
        else: ax.plot(x,y,'b-',linewidth=1,alpha=0.5)
#        ax.scatter(x,y,linewidth=1,edgecolors=colors )
        ax.plot(x[0],y[0],'ro',linewidth=1)
        ax.plot(x[-2],y[-2],'ko',linewidth=1)
        ax.set_title('Axis '+str(axis))
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.savefig(figName+'.pdf')
        plt.show()
        plt.ion()
        return fig,ax
        
    def heatTransfer(self,init_cutoff=0):
        wt=np.array(self.wt[init_cutoff:])
        st=np.array(self.st[init_cutoff:])
        vt_molecule=np.array(self.vt[init_cutoff:]).T[0]/self.mu[0]
        leftHT_idx=np.where(wt==0*st==1)[0]
        rightHT_idx=np.where(wt==1*st==1)[0]
        leftQ=np.zeros(len(vt_molecule))
        rightQ=np.zeros(len(vt_molecule))
        leftQ[leftHT_idx[1:]]=vt_molecule[leftHT_idx[1:]]**2-vt_molecule[leftHT_idx[1:]-1]**2
        #do [1:] to avoid first one, might index out of bound
        #positive heat is heat given from wall to molecule, i.e. added into system
        rightQ[rightHT_idx[1:]]=vt_molecule[rightHT_idx[1:]]**2-vt_molecule[rightHT_idx[1:]-1]**2
        leftQ_cum=np.cumsum(leftQ)
        rightQ_cum=np.cumsum(rightQ)
        #note that leftQ is negative and rightQ is positive, due to left is code and right is hot
        # and that we consider heat flow into the particle is positive
        S_cum=-leftQ_cum/self.params['T'][0]-rightQ_cum/self.params['T'][1]

        return leftQ_cum,rightQ_cum,S_cum

    def heatTransferRate(self):
        ql,qr,s=self.heatTransfer()

        qlr=ql/self.t
        qrr=qr/self.t
        sr=s/self.t
        return qlr, qrr, sr,ql,qr,s
        
    def driftV(self):
        v_d=self.xt_noncyclic[-1][1]/self.mu[1]/self.t[-1]
        # gives just the last number
        return v_d

def defaultParams():
    params={'m':[0.1,0.5,0.5],'T':[1e-1,1.],'L':1.,'D':0.1,'d':0.02,'a':[1,1],'b':[0.5,0.5],
            'kb':1.,'force':[0.,0.,0.],'accelFlag':'none'}
    return params

#def driftV(ptd):
#    v=(ptd['xt_noncyclic'][-1][1]-ptd['xt_noncyclic'][100][1])/(ptd['t'][-1]-ptd['t'][100])
#    return v

def simulation(T=1e5,N=3e1,fName='thermoSim.p',params=defaultParams(),isDump=True):
    pt=geometricPt(params=params)
    t=0.
    nStep=0
    maxStep=N
    while t<T and nStep<maxStep:
        pt.step()
        t=pt.t[-1]
        nStep+=1

    ptDict={'kb':pt.kb,
        'params':pt.params,
        'x0':pt.x0,
        'v0':pt.v0,
        't0':pt.t0,
        'w0':pt.w0,
        's0':pt.s0,
#        'cyclicCounter':pt.cyclicCounter,
        'mu':pt.mu,
        'xW':pt.xW,
        'n':pt.n,
        't':pt.t,
        'xt':pt.xt,
        'vt':pt.vt,
        'wt':pt.wt,
        'st':pt.st,
        'x_move':pt.x_move,
        'xt_noncyclic':pt.xt_noncyclic}
    if isDump:
        pickle.dump(ptDict,open(fName,'wr'))
    return pt,ptDict


def reconstructPt(fN):
    ptd=pickle.load(open(fN,'r'))
    pt=geometricPt(ptd['params'])
    pt.kb=ptd['kb']
#    pt.cyclicCounter=ptd['cyclicCounter']
    pt.mu=ptd['mu']
    pt.xW=ptd['xW']
    pt.n=ptd['n']
    pt.t=ptd['t']
    pt.xt=ptd['xt']
    pt.vt=ptd['vt']
    pt.wt=ptd['wt']
    pt.st=ptd['st']
    pt.x_move=ptd['x_move']
    if 'xt_noncyclic' in ptd.keys():
        pt.xt_noncyclic=ptd['xt_noncyclic']
    else:
        xt_noncyclic=[ptd['xt'][0]]
        for idx in xrange(len(ptd['x_move'])):
            xt_noncyclic.append(xt_noncyclic[-1]+ptd['x_move'][idx]-ptd['xt'][idx])
        ptd['xt_noncyclic']=xt_noncyclic
        pickle.dump(ptd,open(fN,'wr'))
        return reconstructPt(fN)
    return pt

def plotFromFile(fN,axis=[0],cyclic=False,ommit=100):
    pt=reconstructPt(fN)
    fig,ax=pt.plotPt(axis,cyclic=cyclic,ommit=ommit)
    axisStr=[str(x) for x in axis]
    plt.savefig(fN+'-'.join(axisStr)+'.pdf')
    return pt

def sim(fN='asdf.p', varList=[1.],flag='mass'):
#    n=5e2
#    N=1000
    n=5e3
    N=100
    
    paramsList=[]
    if flag=='mass':
        #vary mass ratio but keep total mass of frame and core the same
        #total mass is always 1
        for it in varList:
            param=defaultParams()
            param['m'][1]=1/(1.+it)
            param['m'][2]=it/(1.+it)
            paramsList.append(param)
            
    elif flag=='length':
        for it in varList:
            param=defaultParams()
            param['d']=it*param['D']
            paramsList.append(param)
    elif flag=='temperature':
        for it in varList:
            param=defaultParams()
            param['T'][0]=it*param['T'][1]
            paramsList.append(param)
    elif flag=='alpha':
        for it in varList:
            param=defaultParams()
            param['a']=it
            paramsList.append(param)
    elif flag=='beta':
        for it in varList:
            param=defaultParams()
            param['b']=it
            paramsList.append(param)
    elif flag=='force':
        for it in varList:
            param=defaultParams()
            param['accelFlag']='constant'
            param['force'][1]=it
            paramsList.append(param)
    
    for idx in xrange(len(paramsList)):
        params=paramsList[idx]
        subFN=fN+'_'+flag+'_'+str(idx)
        altSim(n=n,N=N,fN=subFN)
    
    pickle.dump({'path':fN+'_'+flag+'_','flag':flag,'varList':varList},open(fN,'wr'))
    return fN+'_'+flag+'_'

   

def altSim(fN='asdf.p',n=2e3,N=1000):
#    n=2e3
#    N=1000

    xtList=[]
    vtList=[]
#    wtList=[]
    tList=[]
#    stList=[]
    muList=[]
    paramsList=[]
    qlList=[]
    qrList=[]
    sList=[]
    qlrList=[]
    qrrList=[]
    srList=[]
    for idx in xrange(N):
        pt,ptDict=simulation(N=n,fName=fN,isDump=False)
        xtList.append(ptDict['xt_noncyclic'])
        vtList.append(ptDict['vt'])
#        wtList.append(ptDict['wt'])
        tList.append(ptDict['t'])
#        stList.append(ptDict['st'])
        muList.append(ptDict['mu'])
        paramsList.append(ptDict['params'])
        qlr,qrr,sr,ql,qr,s=pt.heatTransferRate()
        qlList.append(ql)
        qrList.append(qr)
        sList.append(s)
        qlrList.append(qlr)
        qrrList.append(qrr)
        srList.append(sr)
    pickle.dump({'n':n,'N':N,'xt':xtList,'vt':vtList,'t':tList,'mu':muList,'params':paramsList,
     'ql':qlList,'qr':qrList,'s':sList,'qlr':qlrList,'qrr':qrrList,'sr':srList},open(fN,'wr'))


def altParamAnalysis(fN,varList,flag):
    
    ll=len(varList)
    timesList=[]
    vtTimedList=[]
    xtTimedList=[]
    for idx in xrange(ll):
        times,vtTimed,xtTimed=altAnalysis(fN=fN+'_'+flag+'_'+str(idx),isPlot=False)
        timesList.append(times)
        vtTimedList.append(vtTimed)
        xtTimedList.append(xtTimed)
    
    window=100
    isPhysical=False
    pad=5
    if flag=='mass':
#        temp = m0/m1 = 0.1*(1+kappa) #m default is 0.1,0.5,0.5
        kappa=np.array(varList)
        temp=0.1*(1+kappa)
#        sqrt(1+temp+gamma) = 1/mu
        turnPhysical=np.sqrt(1+temp+kappa)
        isPhysical=True

    # Drift Velocity
    vt=np.array(vtTimedList)
   
    temp1=np.rollaxis(vt,1)
    shape=temp1.shape
    reshapedVt=np.reshape(temp1[-window:-pad-1],((window-pad-1)*shape[2],shape[1],shape[3]))
    vtPts=np.mean(reshapedVt,axis=0).T[1]
    vtStdPts=np.std(reshapedVt,axis=0).T[1]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel(figXLabel(flag))
    ax.set_xscale('log')
    if isPhysical:
        ax.set_ylabel('Average Physical Drift Velocity')
        vtPts=vtPts*turnPhysical
    else: 
        ax.set_ylabel('Average Drift Velocity')
        
    plt.plot(varList,vtPts,'-ko',lw=1)
    ax.fill_between(varList,vtPts+vtStdPts,vtPts-vtStdPts,facecolor='blue',alpha=0.2,interpolate=True,linewidth=0.0)
    plt.savefig(fN+'_ensemble_'+flag+'_v.pdf')
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel(figXLabel(flag))
    ax.set_xscale('log')
    if isPhysical:
        ax.set_ylabel('Average Physical Drift Velocity')
        vtPts=vtPts*turnPhysical
    else: 
        ax.set_ylabel('Average Drift Velocity')
        
    plt.plot(varList,vtPts,'-ko',lw=1)
    ax.fill_between(varList,vtPts+vtStdPts,vtPts-vtStdPts,facecolor='blue',alpha=0.2,interpolate=True,linewidth=0.0)
    ax.set_ylim([min(vtPts)-max(vtPts)/2.,max(vtPts)*1.2])
    plt.savefig(fN+'_ensemble_'+flag+'_v_lessVar.pdf')
    
    
def altAnalysis(fN='asdf.p',isPlot=True):
    ptd=pickle.load(open(fN))
    #When average over trajectories, we need a unified time
    t=np.array(ptd['t'])
    N=ptd['N']
    n=ptd['n']
#    n=2e3
#    N=1000
    tEnd=np.min(t.T[-1])
    times=np.linspace(0,tEnd,n)
    vt=np.swapaxes(np.array(ptd['vt']),0,1)
    xt=np.swapaxes(np.array(ptd['xt']),0,1)
 
    #When average over trajectories, we need a unified time
    vtTimed=np.empty(vt.shape) #swap axis for easier value assignment
    xtTimed=np.empty(xt.shape)
    for idx in xrange(len(times)-1):
        time=times[idx]
        mask=(np.diff((t<=time))==1)
        paddedMask=np.pad(mask,((0,0),(0,1)),'constant',constant_values=0).T
        vtTimed[idx]=vt[paddedMask]
        xtTimed[idx]=xt[paddedMask]
    if isPlot==True:
        ensemblePlot(fN,times,vtTimed,xtTimed)
    else:
        return times,vtTimed,xtTimed

def ensemblePlot(fN,times,vtTimed,xtTimed):
    pad=5
    timePts=times[:-pad]
    # Position    
    xtPts=np.mean(xtTimed,axis=1).T[1][:-pad-1]
    xtStdPts=np.std(xtTimed,axis=1).T[1][:-pad-1]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.plot(timePts,xtPts,'-k',lw=1)
    ax.set_xlabel('time')
    ax.set_ylabel('average position')
    ax.fill_between(timePts,xtPts+xtStdPts,xtPts-xtStdPts,facecolor='blue',alpha=0.2,interpolate=True,linewidth=0.0)
    plt.savefig(fN+'_ensemble_x.pdf')
    
    # Velocity
    vtPts=np.mean(vtTimed,axis=1).T[1][:-pad-1]
    vtStdPts=np.std(vtTimed,axis=1).T[1][:-pad-1]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.plot(timePts,vtPts,'-k',lw=1)
    ax.set_xlabel('time')
    ax.set_ylabel('average velocity')
    ax.fill_between(timePts,vtPts+vtStdPts,vtPts-vtStdPts,facecolor='blue',alpha=0.3,interpolate=True,linewidth=0.0)
    plt.savefig(fN+'_ensemble_v.pdf')
    

    # velocity distributions
    aa=np.rollaxis(vtTimed,2)[1]
    for t in np.arange(1,len(times),10):
        plt.figure()
        plt.hist(aa[t],50)
        plt.savefig(fN+'_v_distr_'+str(t)+'.pdf')
    return 'done!'

def plotDemo(fN):
    pt=reconstructPt(fN)

    #Position
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t,np.array(pt.xt_noncyclic).T[1],'-k',lw=1)
    ax.set_ylabel('Position of frame')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_x.pdf')
    
    
    #instantaneous Velocity
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t,np.array(pt.vt).T[1],'-k',lw=1)
    ax.set_ylabel('Instantaneous Velocity of frame')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_v.pdf')
    
    #Drift velocity
    v_d=pt.driftV()
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t,v_d,'-k',lw=1)
    ax.set_ylabel('Drift Velocity of frame')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_v_d.pdf')  
    
    
    qlr,qrr,sr,ql,qr,s=pt.heatTransferRate()
    #Heat Difference
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t,ql+qr,'-k',lw=1)
    ax.set_ylabel('Net Heat Transfer')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_Qdiff.pdf')
    
    #Heat Quantity
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t,-ql,'-b',lw=1)
    ax.plot(pt.t,qr,'-k',lw=1)
    ax.set_ylabel('Heat Transfer')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_Q.pdf')
    
    
    #Entropy Production
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t,s,'-k',lw=1)
    ax.set_ylabel('Entropy Production')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_S.pdf')
    
    #Entropy Production Rate, time-averaged
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t,sr,'-k',lw=1)
    ax.set_ylabel('Entropy Production Rate')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_SRate.pdf')
    
    #Heat Transfer Rate, time-averaged
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t,-qlr,'-b',lw=1)
    ax.plot(pt.t,qrr,'-k',lw=1)
    ax.set_ylabel('Heat Trasfer Rate')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_QRate.pdf')    


def figXLabel(flag):
    if flag=='mass':
        return 'Kappa, mass-ratio between core and frame'
    elif flag=='length':
        return 'length-ratio between core and frame'
    elif flag=='temperature':
        return 'temperature-ratio between left and right wall'
    elif flag=='alpha':
        return 'alpha'
    elif flag=='beta':
        return 'beta'


def update_plot(i,offset,pd,x,y,x_move,y_move,point,line,past,jump,cyclic):

    ci=offset+i+pd
    pi=offset+i+pd-1

    nPtsTraj=10

        
    if cyclic:
        point.set_data(x[ci],y[ci])
        if pt.params['accelFlag']=='constant':
            times=np.linspace(0,pt.t[ci+1]-pt.t[ci],nPtsTraj)
            xLine=x[ci]+np.array(pt.vt).T[plotAxis[0]][ci]*times+0.5*pt.aFunc()[plotAxis[0]]*times**2
            yLine=y[ci]+np.array(pt.vt).T[plotAxis[1]][ci]*times+0.5*pt.aFunc()[plotAxis[1]]*times**2
            line.set_data(xLine,yLine)
        else: line.set_data([x[ci],x_move[ci]],[y[ci],y_move[ci]])
        past.set_data(x[ci-pd:ci+1],y[ci-pd:ci+1])
        jump.set_data([x[ci],x_move[pi]],[y[ci],y_move[pi]])
    else:
        point.set_data(x[ci],y[ci])
        if pt.params['accelFlag']=='constant':
            times=np.linspace(0,pt.t[ci+1]-pt.t[ci],nPtsTraj)
            xLine=x[ci]+np.array(pt.vt).T[plotAxis[0]][ci]*times+0.5*pt.aFunc()[plotAxis[0]]*times**2
            yLine=y[ci]+np.array(pt.vt).T[plotAxis[1]][ci]*times+0.5*pt.aFunc()[plotAxis[1]]*times**2
            line.set_data(xLine,yLine)
        else: line.set_data([x[ci],x[ci+1]],[y[ci],y[ci+1]])
        past.set_data(x[ci-pd:ci+1],y[ci-pd:ci+1])
        
    time_text.set_text('time=%.1f' % i)
    
    return point,line,past,jump,time_text

def animate(fN,axis=[0,1],cyclic=False,frames=500):
    global x,y,x_move,y_move
    global pt
    global plotAxis
    global time_text
    
    pt = reconstructPt(fN)
    
    #number of points to draw trajectory of acceleration    
    nPtsTraj=10
    
    offset=0    
    
    interval=500
    
    pd=3 #number of data points to trace
    ci=offset+pd #ci is current index
    pi=offset+pd-1 #pi is past index
    
    frames=min(len(pt.xt)-pd-offset-1,frames)
    
    fig,ax,x,y,xlabel,ylabel,plotAxis=pt.plotFrame(axis=axis,cyclic=cyclic,ommit=1)
    ax.set_title('Axis '+str(axis))
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)  
    if cyclic:
        x_move=np.array(pt.x_move).T[axis[0]]
        y_move=np.array(pt.x_move).T[axis[1]]
        point,=ax.plot(x[ci],y[ci],'o',linewidth=5)
        if pt.params['accelFlag']=='constant':
            times=np.linspace(0,pt.t[ci+1]-pt.t[ci],nPtsTraj)
            xLine=x[ci]+np.array(pt.vt).T[plotAxis[0]][ci]*times+0.5*pt.aFunc()[plotAxis[0]]*times**2
            yLine=y[ci]+np.array(pt.vt).T[plotAxis[1]][ci]*times+0.5*pt.aFunc()[plotAxis[1]]*times**2
            line,=ax.plot(xLine,yLine,'-b',linewidth=2)
        else: line,=ax.plot([x[ci],x_move[ci]],[y[ci],y_move[ci]],'-b',linewidth=2)
        jump,=ax.plot([x[ci],x_move[pi]],[y[ci],y_move[pi]],'-r',linewidth=2)
        past,=ax.plot(x[ci-pd:ci+1],y[ci-pd:ci+1],'-k',linewidth=1)
        time_text=ax.text(0.02,0.95,'',transform=ax.transAxes)
        
        
        ani=animation.FuncAnimation(fig,update_plot,np.arange(1,frames),fargs=(offset,pd,x,y,x_move,y_move,point,line,past,jump,cyclic),
                                    interval=interval, blit=False)
    else:
        x_move=[]
        y_move=[]
        jump=[]
        point,=ax.plot(x[ci],y[ci],'o',linewidth=5)
        if pt.params['accelFlag']=='constant':
            times=np.linspace(0,pt.t[ci+1]-pt.t[ci],nPtsTraj)
            xLine=x[ci]+np.array(pt.vt).T[axis[0]][ci]*times+0.5*pt.aFunc()[axis[0]]*times**2
            yLine=y[ci]+np.array(pt.vt).T[axis[1]][ci]*times+0.5*pt.aFunc()[axis[1]]*times**2
            line,=ax.plot(xLine,yLine,'-b',linewidth=2)
        else: line,=ax.plot([x[ci],x[ci+1]],[y[ci],y[ci+1]],'-b',linewidth=2)
        
        past,=ax.plot(x[ci-pd:ci+1],y[ci-pd:ci+1],'-k',linewidth=1)
        time_text=ax.text(0.02,0.95,'',transform=ax.transAxes)
        
        
        ani=animation.FuncAnimation(fig,update_plot,np.arange(1,frames),fargs=(offset,pd,x,y,x_move,y_move,point,line,past,jump,cyclic),
                                    interval=interval, blit=False)
    ani.save(fN+str(axis[0])+str(axis[1])+'.gif',writer='imagemagick',fps=2)


    
    