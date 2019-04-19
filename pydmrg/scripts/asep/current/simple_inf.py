from idmrg import *
from mpo.asep import return_mpo, curr_mpo
import time
import os

# Set Calculation Parameters
mbd = 10 # Can only be a single value currently

# Jumping Rates
p = 0.1
alpha = 0.5      # in at left
gamma = 1.-alpha  # Out at left
q     = 1.-p      # Jump left
beta  = 0.5     # Out at right
delta = 1.-beta   # In at right
s0 = -0.5
hamParams = np.array([alpha,gamma,p,q,beta,delta,s0])

# Calculation Settings
leftState = True
nStates = 1
alg = 'davidson'

# Create directory for storing states
dirid = str(int(time.time()))
path = 'saved_states/simpleInfinite_'+'mbd'+str(mbd)+'_'+dirid+'/'
os.mkdir(path)
fname = path+'MPS_'


# Run initial Calculation
mpo = return_mpo(4,hamParams)
Etmp,EEtmp,gaptmp = run_idmrg(mpo,
                                 mbd=mbd,
                                 minIter = 1,
                                 maxIter = 10000,
                                 fname=fname+'s0',
                                 nStates=nStates,
                                 alg=alg,
                                 calcLeftState=leftState)

#######################################################################
# Post Process results

# Load MPS
from tools.mps_tools import load_mps
mps,_ = load_mps(fname+'s0_mbd0')
mpsl,_= load_mps(fname+'s0_mbd0_left')

# Get Relevant operators
MPOc = curr_mpo(4,hamParams,singleBond=True,bond=1)
MPO_Eloc = return_mpo(4,hamParams,singleBond=True,bond=1)
MPOc = return_bulk_mpo(MPOc)
MPO_Eloc = return_bulk_mpo(MPO_Eloc)

# Contract networks to get results
from tools.contract import inf_contract as contract
norm = contract(mps=mps,lmps=mpsl,state=0,lstate=0)
current = contract(mpo=MPOc,mps=mps,lmps=mpsl,state=0,lstate=0)/norm
Eloc = contract(mpo=MPO_Eloc,mps=mps,lmps=mpsl,state=0,lstate=0)/norm
print('Current = {}'.format(current))
print('Local Energy = {}'.format(Eloc))
