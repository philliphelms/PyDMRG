import numpy as np
from mpo.asep import return_mpo
from tools.mps_tools import *
from tools.diag_tools import *

# To Do
# - Make modular functions
# - Excited state targeting and state averaging
# - Larger unit cell sizes


# Calculation Details
mbd = 10
nStates = 1

# Get the MPO
p = 0.1
alpha = 0.2      # in at left
gamma = 1.-alpha  # Out at left
q     = 1.-p      # Jump left
beta  = 0.4     # Out at right
delta = 1.-beta   # In at right
s = -0.5
hamParams = np.array([alpha,gamma,p,q,beta,delta,s])
mpoList = return_mpo(2,hamParams)

# Initialize by generating random MPS of size 2
mpsList = create_all_mps(2,mbd,nStates)

# Set up empty initial environment
envList = []
env = [np.array([[[1.]]],dtype=np.complex_),np.array([[[1.]]],dtype=np.complex_)]

# Determine full Hamiltonian
H = einsum('ijk,jlmn,lopq,ros->mpirnqks',env[0],mpoList[0][0],mpoList[0][1],env[1])
(n1,n2,n3,n4,n5,n6,n7,n8) = H.shape
H = np.reshape(H,(n1*n2*n3*n4,n5*n6*n7*n8))

# Solve eigenproblem
E,vecs = np.linalg.eig(H)
inds = np.argsort(E)[::-1]
E = E[inds]
vecs = vecs[:,inds]
print('My Energy = {}'.format(E))

# Do SVD
(_,_,n1,_) = mpoList[0][0].shape
(_,_,n2,_) = mpoList[0][1].shape
(n3,_,_) = env[0].shape
(n4,_,_) = env[1].shape
print(n1,n2,n3,n4)
for state in range(nStates):
    # Reshape Matrices
    psi = np.reshape(vecs[:,state],(n1,n2,n3,n4))
    psi = np.transpose(psi,(2,0,1,3))
    psi = np.reshape(psi,(n3*n1,n4*n2))
    # Do SVD
    (U_,S,V_) = np.linalg.svd(psi,full_matrices=False)
    U = np.reshape(U_,(n3,n1,-1))
    U = U[:,:,:mbd]
    mpsList[state][0] = np.swapaxes(U,0,1)
    V = np.reshape(V_,(-1,n2,n4))
    V = V[:mbd,:,:]
    mpsList[state][0] = np.swapaxes(V,0,1)
    print(V)
    (n4,n5,n6) = mpsList[0][1].shape
    print(np.reshape(V_,(n5,n4,n6)))
    S = S[:mbd]
    # Recontract to check energy
    leftSide = einsum('ijk,k,lkm->il',mpsList[state][0],S,mpsList[state][1])
    rightSide= einsum('IJK,K,LKM->IL',mpsList[state][0].conj(),S.conj(),mpsList[state][1].conj())
    Ham = einsum('abIi,bcLl->IiLl',mpoList[0][0],mpoList[0][1])
    E = einsum('il,IiLl,IL->',leftSide,Ham,rightSide)
    print('Recontracted Energy = {}'.format(E))

"""
# Calculate Full Hamiltonian:
tmp1 = einsum('',
tmp1 = einsum('jaJ,abIi->bijIJ',env[0],mpoList[0][0])
tmp2 = einsum('bijIJ,bcKk->cijkIJK',tmp1,mpoList[0][1])
Ham = einsum('cijkIJK,mcM->IKJMikjm',tmp2,env[1])
(n1,n2,n3,n4,n5,n6,n7,n8) = Ham.shape
Ham1 = np.reshape(Ham,(n1*n2*n3*n4,n5*n6*n7*n8))
Ham = einsum('abIi,bcKk->ikIK',mpoList[0][0],mpoList[0][1])
(n1,n2,n3,n4) = Ham.shape
Ham = np.reshape(Ham,(n1*n2,n3*n4))
print(np.abs(Ham-Ham1))
E,vecs = np.linalg.eig(Ham)
inds = np.argsort(np.real(E))[::-1]
E = E[inds]
vecs = vecs[:,inds]
print('Optimization E = {}'.format(E[0]))

# SVD Of resulting sites (also figure out how to use the RDM to do this?)
(n1,n2,n3) = mpsList[0][0].shape
(n4,n5,n6) = mpsList[0][1].shape
for state in range(nStates):
    psi = np.reshape(vecs[:,state],(n1*n2,n4*n6))
    U,S,V = np.linalg.svd(psi,full_matrices=False)
    mpsList[state][0] = np.reshape(U,(n1,n2,n3))
    mpsList[state][1] = np.reshape(V,(n5,n4,n6))
    mpsList[state][1] = np.swapaxes(mpsList[state][1],0,1)
    # Recontract to check energy
    leftSide = einsum('ijk,k,lkm->il',mpsList[state][0],S,mpsList[state][1])
    rightSide = einsum('IJK,K,LKM->IL',mpsList[state][0].conj(),S.conj(),mpsList[state][1].conj())
    Ham = einsum('abIi,bcLl->IiLl',mpoList[0][0],mpoList[0][1])
    E = einsum('ik,IiKk,IK->',leftSide,Ham,rightSide)
    norm = einsum('ik,ik->',leftSide,rightSide)
    print('Recontracted Energy = {}'.format(E/norm))
# Update Environment
env[0] = einsum('jaJ,ijk,abIi,IJK->kbK',env[0],mpsList[state][0],mpoList[0][0],mpsList[state][0].conj())
env[1] = einsum('mcM,lkm,bcLl,LKM->kbK',env[1],mpsList[state][1],mpoList[0][1],mpsList[state][1].conj())
E = einsum('kbK,kbK,k,K->',env[0],env[1],S,S.conj())
print('Recontracted Energy (2) = {}'.format(E))

# Run convergence loop?
mpoList = return_mpo(4,hamParams)
converged = False
while not converged:
    # Calculate full hamiltonian:
    tmp1 = einsum('jaJ,abIi->bijIJ',env[0],mpoList[0][1])
    tmp2 = einsum('bijIJ,bcLl->cijlIJL',tmp1,mpoList[0][2])
    Ham = einsum('cijlIJL,mcM->ILmJilMj',tmp2,env[1])
    #Ham = einsum('cijlIJL,mcM->ILMJilmj',tmp2,env[1])
    (n1,n2,n3,n4,n5,n6,n7,n8) = Ham.shape
    Ham = np.reshape(Ham,(n1*n2*n3*n4,n5*n6*n7*n8))
    E,vecs = np.linalg.eig(Ham)
    print('Optimization E = {}'.format(E[0]))
    converged = True
"""
