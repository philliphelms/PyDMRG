import numpy as np
from pydmrg.mpo.asep import return_mpo
from pydmrg.idmrg import return_bulk_mpo, return_edge_mpo
from pydmrg.tools.mpo_tools import mpo2mat

# A simple script to run the iDMRG algorithm 
# (sec 10, schollwock, DMRG in the age of MPS)

# Hamiltonian Parameters
p = 0.1
alpha = 0.5
gamma = 1.-alpha
q     = 1.-p
beta  = 0.5
delta = 1.-beta
s = -0.5

# Get mpo
hamParams = np.array([alpha,gamma,p,q,beta,delta,s])
mpo = return_mpo(4,hamParams)

# Start with ED of a 2 site chain
mpoEdge = return_edge_mpo(mpo)
H = mpo2mat(mpoEdge)
E,v = np.linalg.eig(H)
inds = np.argsort(E)[::-1]
E = E[inds]
v = v[:,inds]
print('Energy = {}'.format(E[0]))
