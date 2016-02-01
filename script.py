from thermoSim import *



if __name__ == "__main__":

#    fN='wagagaHei.p'
#    params=defaultParams()
#    pt=simulation(N=1e6,fName=fN,params=params)
#    plotDemo(fN)
    
#    fN='wagaga_anime.p'
#    params=defaultParams()
##    params['accelFlag']='constant'
##    params['force']=[0.,0.17,0.]
#    params['m']=[0.5,1.,1.]
#    params['D']=0.3
#    params['d']=0.06
#    pt=simulation(N=1e2,fName=fN,params=params)
#
#    animate(fN,[0,1],cyclic=False,frames=100) 
#    animate(fN,[2,1],cyclic=False,frames=100)
    
    fN='ensemble_alt.p'
#    altSim(fN)
    altAnalysis(fN)
#    pdb.set_trace()
    
#    
#    fN='massEnsemble_noForce.p'
#    varList=np.logspace(-2,1,20)
##    results=sim(fN=fN,varList=varList,flag='mass')
#    print('just need to ploT!!!!!!!!!!!!!!')
#    altParamAnalysis(fN,varList,flag='mass')
#
##
#    fN='temperatureEnsemble_noForce.p'
#    flag='temperature'
#    varList=np.logspace(-2,0,20)
##    results=sim(fN=fN,varList=varList,flag=flag)
#    print('just need to ploT!!!!!!!!!!!!!!')
#    altParamAnalysis(fN,varList,flag=flag)
##    
#    fN='lengthEnsemble_noForce.p'
#    flag='length'
#    varList=np.logspace(-2,0,20)
##    results=sim(fN=fN,varList=varList,flag=flag)
#    print('just need to ploT!!!!!!!!!!!!!!')
#    altParamAnalysis(fN,varList,flag=flag)
##    
#    fN='alphaEnsemble_noForce.p'
#    varList=np.logspace(-2,0,20)
#    flag='alpha'
##    results=sim(fN=fN,varList=varList,flag='alpha')
#    print('just need to ploT!!!!!!!!!!!!!!')
#    altParamAnalysis(fN,varList,flag=flag)
##    
#    fN='betaEnsemble_noForce.p'
#    flag='beta'
#    varList=np.logspace(-2,0,20)
##    results=sim(fN=fN,varList=varList,flag=flag)
#    print('just need to ploT!!!!!!!!!!!!!!')    
#    altParamAnalysis(fN,varList,flag=flag)
#    
    
#    fN='forceEnsemble.p'
#    flag='force'
#    varList=np.logspace(-1,1,20)
##    results=sim(fN=fN,varList=varList,flag=flag)
#    print('just need to ploT!!!!!!!!!!!!!!')    
#    altParamAnalysis(fN,varList,flag=flag)
#    
#    
#    fN='forceEnsembleNew.p'
#    flag='force'
#    varList=np.logspace(-1,1,20)
#    results=sim(fN=fN,varList=varList,flag=flag)
#    print('just need to ploT!!!!!!!!!!!!!!')    
#    altParamAnalysis(fN,varList,flag=flag)
    