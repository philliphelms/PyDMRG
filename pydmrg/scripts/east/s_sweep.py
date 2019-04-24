from dmrg import *
from mpo.east import return_mpo
import time
from sys import argv
import os

# Set Calculation Parameters
N = 19
c = 0.2 
#sVec = np.logspace(-6,1,100)
sVec = np.linspace(-1.,1.,100)
mbd = 10 #np.array([2,5,10,20,30,40,50,60,70,80,90,100,150,200,300,400,500]) 
maxIter = 5

# Calculation Settings
leftState = False
nStates = 2
alg = 'exact'

# Create directory for storing states
dirid = str(int(time.time()))
path = 'saved_states/east/simple_'+'N'+str(N)+'_'+dirid+'/'
os.mkdir(path)
fname = path+'MPS_'

# Allocate memory for results
E   = np.zeros(sVec.shape)
EEr = np.zeros(sVec.shape)
EEl = np.zeros(sVec.shape)
gap = np.zeros(sVec.shape)

# Run initial Calculation
for sind,s in enumerate(sVec):
    print(s,sind)
    hamParams = np.array([c,s])
    mpo = return_mpo(N,hamParams)
    Etmp,EEtmp,gaptmp = run_dmrg(mpo,
                                 mbd=mbd,
                                 maxIter = maxIter,
                                 fname=fname+'s0',
                                 nStates=nStates,
                                 alg=alg,
                                 calcLeftState=leftState)
    E[sind]   = Etmp
    gap[sind] = gaptmp
    if leftState: 
        EEr[sind] = EEtmp[0]
        EEl[sind] = EEtmp[1]
    else:
        EEr[sind] = EEtmp

# Print Results
for i in range(len(sVec)):
    print('s={:f}\tE={:e}\tEEr={:e}\tEEl={:e}\tgap={:e}'.format(sVec[i],E[i],EEr[i],EEl[i],gap[i]))
