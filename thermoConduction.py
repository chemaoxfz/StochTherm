import numpy as np
from scipy import stats
import pdb
import pickle

class geometricPt:
    def __init__(self,params={'m':[1e-3,1.,1.],'T1':1e-1,'T2':1.,'L':1/2.,'a':0.5,'b':0.5,'dt':1e-3}):
        self.kb=1 #boltzmann constant        
        self.dt=params['dt']
        self.m=params['m']
        self.a=params['a']
        self.b=params['b']
        self.T=[params['T1'],params['T2']]
        L=params['L']
        self.L=L
        self.x=np.array([np.random.rand(),np.random.rand(),np.random.rand()*(L)-L/2])
        self.v=np.pad(self.MB1D(params['T1'],params['m'][0]),(0,2),'constant',constant_values=0)
        
        # note that for the mass, x3, we only care about  relative position 
        # compared to x2, the frame. velocity is also in relative terms
        
    def step(self):
        '''
        A step is time evolution of one small time step
        '''
        x=self.x
        v=self.v
        m=self.m
        dt=self.dt
        x+=v*dt
        
        if x[0]<=0 or x[0] >= 1:
            if np.random.rand()<=self.a:
                #with prob a inherit wall's T
                v[0]=-np.sign(v[0])*np.abs(self.MB1D(self.T[int(round(x[0]))],m[0]))
#                pdb.set_trace()
            else: v[0]=-v[0]
        if np.abs(x[0]-x[1])%1<np.abs(v[0]-v[1])*dt:
            if np.random.rand()<=self.b:
                v[0],v[1]=self.collision(m[0],m[1],v[0],v[1])
        if np.abs(x[2]-self.L/2)<np.abs(v[2])*dt:
            v[1],v[2]=self.collision(m[1],m[2],v[1],v[2]+v[1])
            v[2]=v[2]-v[1]
            
        self.x=x
        self.v=v
    
    def collision(self,m1,m2,v1,v2):
        v1_pre=v1-v2 #consider 2 to be stationary
        v1_post=(m1-m2)/(m1+m2)*v1_pre
        v2_post=2*m1/(m1+m2)*v1_pre
        return v1_post+v2, v2_post
        
    def MB1D(self,T,m,n=1):
        unif=np.random.rand(n)
        x=stats.norm.ppf(unif,loc=0,scale=m/(self.kb*T))
        return x
    

def simulation(nIns,nStep,dt,fName='thermoSim.p'):
    instanceData={'v':np.zeros([nIns,nStep,3]),'x':np.zeros([nIns,nStep,3]),
            'params':{'m':[1e-3,1.,1.],'T1':1e-1,'T2':1.,'L':1/2.,'a':0.5,'b':0.5,'dt':1e-3,'nStep':nStep}}
    for ins in xrange(nIns):
        p=geometricPt(instanceData['params'])
        for step in xrange(nStep):
            instanceData['v'][ins][step]=p.v
            instanceData['x'][ins][step]=p.x
            p.step()
    
    pickle.dump(instanceData,open(fName,'wr'))            
        
    
if __name__ == "__main__":
    simulation(1,1000000,1e-3,fName='thermoSim.p')