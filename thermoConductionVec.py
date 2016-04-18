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
    def __init__(self,params={'m':[0.5,1.,1.],'T':[1e-1,1.],'L':1.,'D':0.5,'d':0.1,'a':1,'b':0.5,'kb':1.,'force':[0.,1.7e-1,0.],'accelFlag':'constant'}):
        self.kb=1 #boltzmann constant
        self.params=params
        D=params['D']
        d=params['d']
        m=params['m']
        L=params['L']
        #x1 is molecule, x2 is frame (upward), x3 is weight (towards us).
#        self.x0=np.array([np.random.rand(),np.random.rand(),np.random.rand()*(D-d)-(D-d)/2])
#        self.v0=np.pad(self.MB1D(params['T1'],params['m'][0]),(0,2),'constant',constant_values=0)
        self.x0=np.array([0.,0.,0.])
        self.v0=np.array([self.MB1D(params['T'][0],params['m'][0])[0],self.MB1D(params['T'][0],params['m'][1])[0],self.MB1D(params['T'][0],params['m'][2])[0]])
        self.t0=0.
        self.w0=0
        self.s0=1
        self.cyclicCounter0=-1
        self.w_o0=self.w0
        #counting how many times has the cyclic boundary been crossed
        #entries are for wall 2,3,4,5, i.e. Ur, Lr, F, B 

        
        mu=np.sqrt(m/np.sum(m))
        self.mu=mu
        if 'force' not in params.keys():
            self.aFunc=self.accelFunc(flag=params['accelFlag'],arg=np.array(params['accel'])*self.mu)
        else: self.aFunc=self.accelFunc(flag=params['accelFlag'],arg=np.array(params['force'])/self.params['m']*self.mu)
        #Order of walls:L,R, Ur,Lr, F,B, FUr,FLr, FUrE,FLrE.
        self.xW=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[0,0,1],[0,0,0],[0,(D-d)/2,0],[0,0,(D-d)/2],[0,D/2,0],[D/2,0,0],[0,(D-d)/2,L],[0,L,(D-d)/2],[L,D/2,0],[D/2,L,0]])
        self.xW=mu*self.xW
        #normal vectors
        fUr=np.array([0,-mu[2]/np.sqrt(mu[2]**2+mu[1]**2),mu[1]/np.sqrt(mu[2]**2+mu[1]**2)])
        fUrE=np.array([-mu[1]/np.sqrt(mu[0]**2+mu[1]**2),mu[0]/np.sqrt(mu[0]**2+mu[1]**2),0])
        self.n=np.array([[1,0,0],[-1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1],fUr,-1*fUr,fUrE,-1*fUrE,fUr,-1*fUr,fUrE,-1*fUrE])
        self.t=[self.t0]
        self.xt=[mu*self.x0]
        self.xt_noncyclic=[mu*self.x0]
        self.vt=[mu*self.v0]
        self.wt=[self.w0]
        self.w_ommit=[self.w_o0]
        self.st=[self.s0]
        self.x_move=[]
        self.cyclicCounter=[self.cyclicCounter0]
    
    
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
        self.cyclicCounter.append(cc)
    
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
            if r<self.params['a']:
                v[0]=self.MB1D(self.params['T'][wIdx],self.params['m'][0])*(-1)**wIdx
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
            #FUrE, FLrE, collision between 
            r=np.random.rand()
            if r<self.params['b']:
                v=v-2*self.n[wIdx].dot(v)*self.n[wIdx]
                s=1
            else: s=0

        return x,v,s,cc,w_o
        
    def MB1D(self,T,m,n=1):
        unif=np.random.rand(n)
        x=np.sqrt(-2*self.kb*T/m*np.log(1-unif))
        return x
    
    def cyclicToNoncyclic(self,x_cyclic,axis):
        idx=axis-1
        cc=np.array(self.cyclicCounter)
        ccAddIdx=np.where(cc==idx*2)
        ccDedIdx=np.where(cc==idx*2+1)
        temp=np.zeros(len(x_cyclic))
        temp[ccAddIdx]=1.
        temp[ccDedIdx]=-1.
        temp=np.cumsum(temp)
        x=x_cyclic+temp*self.params['L']*self.mu[axis]
        return x

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
                y=self.cyclicToNoncyclic(yy,axis[0])
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
        
    def heatTransfer(self):
        leftHT_idx=np.where((np.array(self.wt)==0)*(np.array(self.st)==1))[0]
        rightHT_idx=np.where((np.array(self.wt)==1)*(np.array(self.st)==1))[0]
        vt_molecule=np.array(self.vt).T[0]
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
        #N is smoothing parameter
        N=1000
        smooth=lambda x: np.convolve(x, np.ones((N,))/N, mode='valid')
        ql,qr,s=self.heatTransfer()
        qlRate=np.diff(ql)/np.diff(self.t)
#        qlr=np.mean(qlRate[100:-10])
        qlr=smooth(qlRate)
        qrRate=np.diff(qr)/np.diff(self.t)
#        qrr=np.mean(qrRate[100:-10])
        qrr=smooth(qrRate)
        sRate=np.diff(s)/np.diff(self.t)
#        sr=np.mean(sRate[100:-10])
        sr=smooth(sRate)
        return qlr, qrr, sr
        
    def driftV(self):
#        v=(self.xt_noncyclic[-1][1]-self.xt_noncyclic[100][1])/(self.t[-1]-self.t[100])
        v=np.mean(np.array(self.vt).T[1][100:-10])
        return v

def defaultParams():
    params={'m':[0.1,0.5,0.5],'T':[1e-1,1.],'L':1.,'D':0.1,'d':0.02,'a':1,'b':0.5,
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
        'cyclicCounter':pt.cyclicCounter,
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
    pt.cyclicCounter=ptd['cyclicCounter']
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

def plotHT(fN):
    pt=reconstructPt(fN)
    q,ql,qr=pt.heatTransfer()
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(np.array(pt.t),q,'r',lw=2)
    ax.plot(np.array(pt.t),ql,'b',lw=2)
    ax.plot(np.array(pt.t),qr,'k',lw=2)
    plt.savefig(fN+'_heatRtoL.pdf')
    return q,ql,qr
    
def sim(fN='asdf.p', varList=[1.],flag='mass'):
    N=1e6
    
    paramsList=[]
    if flag=='mass':
        for it in varList:
            param=defaultParams()
            param['m'][2]=it*param['m'][1]
            paramsList.append(param)
    elif flag=='mass_alt':
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
        pt,ptDict=simulation(N=N,params=params,fName=subFN,isDump=True)
    
    pickle.dump({'path':fN+'_'+flag+'_','flag':flag,'varList':varList},open(fN,'wr'))
    return fN+'_'+flag+'_'


def altSim(fN='asdf.p',n=2e2,N=10):
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
    for idx in xrange(N):
        pt,ptDict=simulation(N=n,fName=fN,isDump=False)
        xtList.append(ptDict['xt_noncyclic'])
        vtList.append(ptDict['vt'])
#        wtList.append(ptDict['wt'])
        tList.append(ptDict['t'])
#        stList.append(ptDict['st'])
        muList.append(ptDict['mu'])
        paramsList.append(ptDict['params'])
        ql,qr,s=pt.heatTransfer()
        qlList.append(ql)
        qrList.append(qr)
        sList.append(s)
    pickle.dump({'n':n,'N':N,'xt':xtList,'vt':vtList,'t':tList,'mu':muList,'params':paramsList,
     'ql':qlList,'qr':qrList,'s':sList},open(fN,'wr'))
    
    
def altAnalysis(fN='asdf.p'):
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
    ql=np.swapaxes(np.array(ptd['ql']),0,1)
    qr=np.swapaxes(np.array(ptd['qr']),0,1)
    s=np.swapaxes(np.array(ptd['s']),0,1)
    #When average over trajectories, we need a unified time
    vtTimed=np.empty(vt.shape) #swap axis for easier value assignment
    xtTimed=np.empty(xt.shape)
    qlTimed=np.empty(ql.shape)
    qrTimed=np.empty(qr.shape)
    for idx in xrange(len(times)-1):
        time=times[idx]
        mask=(t<=time)
        paddedMask=np.pad((np.diff(mask)==1),((0,0),(0,1)),'constant',constant_values=0).T
        vtTimed[idx]=vt[paddedMask]
        xtTimed[idx]=xt[paddedMask]
        qlTimed[idx]=ql[paddedMask]
        qrTimed[idx]=qr[paddedMask]
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
    
    # Heat Transfer
    qlPts=np.mean(qlTimed,axis=1)[:-pad-1]
    qlStdPts=np.std(qlTimed,axis=1)[:-pad-1]
    qrPts=np.mean(qrTimed,axis=1)[:-pad-1]
    qrStdPts=np.std(qrTimed,axis=1)[:-pad-1]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.plot(timePts,-qlPts,'-b',lw=1)
    plt.plot(timePts,qrPts,'-k',lw=1)
    ax.set_xlabel('time')
    ax.set_ylabel('average heat transfer')
    ax.fill_between(timePts,-qlPts+qlStdPts,-qlPts-qlStdPts,facecolor='blue',alpha=0.2,interpolate=True,linewidth=0.0)
    ax.fill_between(timePts,qrPts+qrStdPts,qrPts-qrStdPts,facecolor='black',alpha=0.2,interpolate=True,linewidth=0.0)
    plt.savefig(fN+'_ensemble_q.pdf')
    
    # Entropy
#    s=-qlTimed/ptd['params']['T'][0]-qrTimed/ptd['params']['T'][1]
    sPts=np.mean(s,axis=1)[:-pad-1]
    sStdPts=np.std(s,axis=1)[:-pad-1]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.plot(timePts,sPts,'-k',lw=1)
    ax.set_xlabel('time')
    ax.set_ylabel('average entropy production')
    ax.fill_between(timePts,sPts+sStdPts,sPts-sStdPts,facecolor='blue',alpha=0.3,interpolate=True,linewidth=0.0)
    plt.savefig(fN+'_ensemble_s.pdf')

def plotDemo(fN):
    pt=reconstructPt(fN)

    #Position
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t,np.array(pt.xt_noncyclic).T[1],'-k',lw=1)
    ax.set_ylabel('Position of frame')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_x.pdf')
    
    
    #Velocity
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t,np.array(pt.vt).T[1],'-k',lw=1)
    ax.set_ylabel('Velocity of frame')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_v.pdf')
    
    #Smoothed Velocity
    #smoothing parameter
    N=1000
    pad=15000
    smooth=lambda x: np.convolve(x, np.ones((N,))/N, mode='valid')  
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t[:-N+1],smooth(np.array(pt.vt).T[1]),'-k',lw=1)
    ax.set_ylabel('Smoothened Velocity of frame')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_v_smooth.pdf')  
    
    ql,qr,s=pt.heatTransfer()
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
    
    qlr,qrr,sr=pt.heatTransferRate()
    #Entropy Production Rate
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t[pad:len(sr)],sr[pad:],'-k',lw=1)
    ax.set_ylabel('Entropy Production Rate')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_SRate.pdf')
    
    #Heat Transfer Rate
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(pt.t[pad:len(qlr)],qlr[pad:],'-b',lw=1)
    ax.plot(pt.t[pad:len(qrr)],qrr[pad:],'-k',lw=1)
    ax.set_ylabel('Heat Trasfer Rate')
    ax.set_xlabel('time')
    plt.savefig(fN+'_demo_QRate.pdf')    

def plotNoForce(path,varList=[1.],flag='mass'):
    vVec=np.zeros(len(varList))
    qrRateVec=np.zeros(len(varList))
    qlRateVec=np.zeros(len(varList))
    sRateVec=np.zeros(len(varList))
    for idx in xrange(len(varList)):
        pt=reconstructPt(fN+'_'+flag+'_'+str(idx))
        vVec[idx]=pt.driftV()
        qlr,qrr,sr=pt.heatTransferRate()
        qlRateVec[idx]=qlr
        qrRateVec[idx]=qrr
        sRateVec[idx]=sr
#    fig=plt.figure()
#    ax=fig.add_subplot(111)
#    ax.plot(varList,vVec,'-k',lw=2)
#    ax.set_xscale('log')
#    ax.set_ylabel('mass-weighted drift velocity of frame')
#    ax.set_xlabel('Kappa, mass-ratio between core and frame')    
#    plt.savefig(fN+'_'+flag+'3.pdf')
    
    #temp = m0/m1 = 0.1
    # sqrt(1+temp+gamma) = 1/mu
#    muVec=np.sqrt(varList+1.1)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(varList,vVec,'-k',lw=2)
    ax.set_xscale('log')
    ax.set_ylabel('Physical drift velocity of frame')
    plt.savefig(fN+'_'+flag+'_V.pdf')
        
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(varList,qlRateVec,'-b',lw=2)
    ax.plot(varList,qrRateVec,'-k',lw=2)
    ax.set_xscale('log')
    ax.set_ylabel('Heat Transfer Rate')
    ax.set_xlabel(figXLabel(flag))
    plt.savefig(fN+'_'+flag+'_Q.pdf')
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(varList,sRateVec,'-k',lw=2)
    ax.set_xscale('log')
    ax.set_ylabel('Entropy Production Rate')
    ax.set_xlabel(figXLabel(flag))
    plt.savefig(fN+'_'+flag+'_Q.pdf')
    pdb.set_trace()

def plotForceStuff(fN='asdf.p'):
    pass

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


def plotDriftVelocity(path,varList=[1.],flag='mass'):
    vVec=np.zeros(len(varList))
    for idx in xrange(len(varList)):
        ptd=pickle.load(open(path+'_'+flag+'_'+str(idx)))
        vVec[idx]=(driftV(ptd,window))
#    fig=plt.figure()
#    ax=fig.add_subplot(111)
#    ax.plot(varList,vVec,'-k',lw=2)
#    ax.set_xscale('log')
#    ax.set_ylabel('mass-weighted drift velocity of frame')
#    ax.set_xlabel('Kappa, mass-ratio between core and frame')    
#    plt.savefig(fN+'_'+flag+'3.pdf')
    
    #temp = m0/m1 = 0.1
    # sqrt(1+temp+gamma) = 1/mu
#    muVec=np.sqrt(varList+1.1)
    fig=plt.figure()
    ax=fig.add_subplot(111)
#    ax.plot(varList,vVec*muVec,'-k',lw=2)
    ax.plot(varList,vVec,'-k',lw=2)
    ax.set_xscale('log')
    ax.set_ylabel('Physical drift velocity of frame')
    ax.set_xlabel('Kappa, mass-ratio between core and frame')
    plt.savefig(fN+'_'+flag+'.pdf')
    pdb.set_trace()
    

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

if __name__ == "__main__":
    
#    fN='wagagaHei.p'
#    params=defaultParams()
#    pt=simulation(N=1e6,fName=fN,params=params)
#    plotDemo(fN)
    
##    fN='wagagaHei_accel.p'
#    params=defaultParams()
#    params['accelFlag']='constant'
#    params['force']=[0.,0.17,0.]
##    params['m']=[0.5,1.,1.]
##    params['D']=0.3
##    params['d']=0.06
#    pt=simulation(N=1e5,fName=fN,params=params)
#    plotDemo(fN)
    
    
#    fN='wagagaHei_accel_large.p'
#    params=defaultParams()
#    params['accelFlag']='constant'
#    params['force']=[0.,0.5,0.]
##    params['m']=[0.5,1.,1.]
##    params['D']=0.3
##    params['d']=0.06
#    pt=simulation(N=1e5,fName=fN,params=params)
#    plotDemo(fN)    
#    
#    ommit=1
    
#    fN='wagagaHei_accel_largest.p'
#    params=defaultParams()
#    params['accelFlag']='constant'
#    params['force']=[0.,1.5,0.]
##    params['m']=[0.5,1.,1.]
##    params['D']=0.3
##    params['d']=0.06
#    pt=simulation(N=1e5,fName=fN,params=params)
#    plotDemo(fN)    
#    
#    ommit=1
    
#    plotFromFile(fN,axis=[0,1],cyclic=False,ommit=ommit)
#    plotFromFile(fN,axis=[2,1],cyclic=False,ommit=ommit)
#    pt=plotFromFile(fN,axis=[1],ommit=ommit)
    
#    path='massR_noForce.p'
#    flag='mass'
#    for idx in xrange(50):
#        fN=path+'_'+flag+'_'+str(idx)
#        reconstructPt(fN)

#    q,ql,qr=plotHT(fN)
#    animate(fN,[0,1],cyclic=False,frames=100) 
#    animate(fN,[0,1],cyclic=True,frames=100)
#    animate(fN,[2,1],cyclic=False,frames=100)

#    fN='massAlt0_noForce.p'
#    massRatioList=np.logspace(-2,2,50)
#    results=sim(fN=fN,varList=massRatioList,flag='mass_alt')
#    print('just need to ploT!!!!!!!!!!!!!!')
#    varList=massRatioList
#    plotDriftVelocity(fN,varList,flag='mass_alt')


#    fN='temperature_noForce.p'
#    varList=np.logspace(-2,0,20)
#    results=sim(fN=fN,varList=varList,flag='temperature')
#    print('just need to ploT!!!!!!!!!!!!!!')
#    
#    fN='length_noForce.p'
#    varList=np.logspace(-2,0,20)
#    results=sim(fN=fN,varList=varList,flag='length')
#    print('just need to ploT!!!!!!!!!!!!!!')
#    
#    fN='alpha_noForce.p'
#    varList=np.logspace(-2,0,20)
#    results=sim(fN=fN,varList=varList,flag='alpha')
#    print('just need to ploT!!!!!!!!!!!!!!')
#    
#    fN='beta_noForce.p'
#    varList=np.logspace(-2,0,20)
#    results=sim(fN=fN,varList=varList,flag='beta')
#    print('just need to ploT!!!!!!!!!!!!!!')
    
#    fN='force.p'
#    varList=np.linspace(1e-4,10,20)
#    results=sim(fN=fN,varList=varList,flag='force')
#    print('just need to ploT!!!!!!!!!!!!!!')
    
    fN='ensemble.p'
    altSim(fN)
    altAnalysis(fN)
#    pdb.set_trace()
    
#    plotNoForce(fN,varList=varList,flag='temperature')