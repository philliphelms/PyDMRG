from dmrg import *
from mpo.east import return_mpo
import time
from sys import argv
import os
from tools.ed import ed

# Set Calculation Parameters
N = 5
c = 0.2 
sVec = np.linspace(-0.1,.1,500)
#sVec = np.array([1])
sVec = np.linspace(-1,1,10)
mbd = 10 #np.array([2,5,10,20,30,40,50,60,70,80,90,100,150,200,300,400,500]) 
maxIter = 5

# Run Calculations
for sind,s in enumerate(sVec):
    hamParams = np.array([c,s])
    mpo = return_mpo(N,hamParams)
    u,v = ed(mpo)
    if False:
        print('Best state:')
        nConfigs,nStates = v.shape
        for i in range(nConfigs):
            printStr = ''
            for j in range(nConfigs):
                printStr += '{}\t'.format(v[i,j]**2.)
            print(printStr)

    print('s={}\tE={}\t{}\t{}\t{}\t{}\t{}\t{}'.format(s,np.real(u[0]),np.real(u[1]),np.real(u[2]),np.real(u[3]),np.real(u[4]),np.real(u[5]),np.real(u[6])))
