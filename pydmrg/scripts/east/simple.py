from dmrg import *
from mpo.east import return_mpo
import time
from sys import argv
import os

# Set Calculation Parameters
N = 30
c = 0.2 
s = -1.
mbd = 50 #np.array([2,5,10,20,30,40,50,60,70,80,90,100,150,200,300,400,500]) 
maxIter = 5

# Jumping Rates
hamParams = np.array([c,s])
# Calculation Settings
leftState = True
nStates = 2
alg = 'davidson'

# Create directory for storing states
dirid = str(int(time.time()))
path = 'saved_states/east/simple_'+'N'+str(N)+'_'+dirid+'/'
os.mkdir(path)
fname = path+'MPS_'

# Run initial Calculation
mpo = return_mpo(N,hamParams)
Etmp,EEtmp,gaptmp = run_dmrg(mpo,
                             mbd=mbd,
                             maxIter = maxIter,
                             fname=fname+'s0',
                             nStates=nStates,
                             alg=alg,
                             calcLeftState=leftState)
