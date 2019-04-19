from dmrg import *
from mpo.asep import return_mpo
from mpo.asep import curr_mpo
from tools.contract import full_contract as contract
import time
from sys import argv
import os

# Set Calculation Parameters
N = 500
p = 0.1 
mbd = np.array([2,5,10,20,30,40,50,60,70,80,90,100,150,200,300,400,500]) # Can only be a single value currently

# Jumping Rates
alpha = 0.5      # in at left
gamma = 1.-alpha  # Out at left
q     = 1.-p      # Jump left
beta  = 0.5     # Out at right
delta = 1.-beta   # In at right
s0 = -0.5
hamParams = np.array([alpha,gamma,p,q,beta,delta,s0])
# Calculation Settings
leftState = True
nStates = 2
alg = 'davidson'

# Create directory for storing states
dirid = str(int(time.time()))
path = 'saved_states/simpleUshnish_'+'N'+str(N)+'_'+dirid+'/'
os.mkdir(path)
fname = path+'MPS_'

# Run initial Calculation
mpo = return_mpo(N,hamParams)
Etmp,EEtmp,gaptmp,env = run_dmrg(mpo,
                                 mbd=mbd,
                                 maxIter = 1,
                                 fname=fname+'s0',
                                 nStates=nStates,
                                 alg=alg,
                                 returnEnv=True,
                                 calcLeftState=leftState)
