import numpy as np
from mpo.asep import return_mpo
from tools.mps_tools import *
from tools.diag_tools import *

# Calculation Details
mbd = 10
nStates = 1

# Get the MPO
p = 0.1
alpha = 0.5      # in at left
gamma = 1.-alpha  # Out at left
q     = 1.-p      # Jump left
beta  = 0.5     # Out at right
delta = 1.-beta   # In at right
s = -0.5
hamParams = np.array([alpha,gamma,p,q,beta,delta,s])
mpoList = return_mpo(3,hamParams)

# Initialize by generating random MPS of size 2
mpsList = create_all_mps(2,mbd,nStates)
print('Length of mps List: {}'.format(len(mpsList)))
print('Length of mps: {}'.format(len(mpsList[0])))
print('Length of mpo List: {}'.format(len(mpoList)))
print('Length of mpo: {}'.format(len(mpoList[0])))

# Set up empty initial environment
envList = []
for state in range(nStates):
    env = [np.array([[[1.]]],dtype=np.complex_),np.array([[[1.]]],dtype=np.complex_)]
    envList.append(env)
# Do two-site optimization of random MPS of size 2
E,vecs,ovlp = calc_eigs(mpsList,mpoList,envList,0,nStates,oneSite=False)
# Do SVD Of resulting sites (also figure out how to use the RDM to do this?)
print('Energies: {}'.format(E))
print('Resulting Eigenvecs: {}'.format(vecs.shape))

