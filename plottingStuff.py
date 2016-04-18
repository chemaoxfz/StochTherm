# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:56:05 2015

@author: xfz
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import pdb
import matplotlib.animation as animation
import pickle


from scipy.special import comb

def plotFromFile(fileName,figName):  
    data=pickle.load(open(fileName,'rb'))
    v=data['v']
    v=v.reshape([v.shape[0],v.shape[2],v.shape[1]])
    x=data['x']
    x=x.reshape([x.shape[0],x.shape[2],x.shape[1]])
    t=np.arange(0,data['params']['nStep']*data['params']['dt'],data['params']['dt'])
    fig, ax = plt.subplots(1)
#    ax.plot(t[:1e2], x[0][0][:1e2],'-b',  label='small particle')
    ax.plot(x[0][0][:3],x[0][1][:3]%1,'-ko', label='frame')
    ax.plot([0,1],[0,1],'-r')
#    ax.plot(t,x[0][2],'-ro', label='mass')
    
    
    
    ax.set_title('Positions')
#    ax.set_title('1D Ising Model')
    
    ax.legend(loc='upper left')
    ax.set_xlabel('time')
    ax.set_ylabel('position')
    plt.savefig(figName)
    plt.show()

if __name__ == "__main__":
    fileName='thermoSim'
    plotFromFile(fileName+'.p',fileName+'.pdf')