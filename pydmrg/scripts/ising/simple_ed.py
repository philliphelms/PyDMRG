from dmrg import *
from ed import *
from mpo.ising import return_mpo
import time
from sys import argv
import os

# Set Calculation Parameters
N = 8
h = -1. 
mbd = 10#np.array([2,5,10,20,30,40,50,60,70,80,90,100,150,200,300,400,500]) # Can only be a single value currently
hamParams = np.array([h])

# Calculation Settings
leftState = False
nStates = 1
alg = 'davidson'

# Create directory for storing states
dirid = str(int(time.time()))
path = 'saved_states/ising_'+'N'+str(N)+'_'+dirid+'/'
os.mkdir(path)
fname = path+'MPS_'

# Run initial Calculation
mpo = return_mpo(N,hamParams)
e,v = ed(mpo)
print(e)
"""
Etmp,EEtmp,gaptmp,env = run_dmrg(mpo,
                                 mbd=mbd,
                                 maxIter = 3,
                                 fname=fname+'s0',
                                 nStates=nStates,
                                 alg=alg,
                                 returnEnv=True,
                                 calcLeftState=leftState)
"""
