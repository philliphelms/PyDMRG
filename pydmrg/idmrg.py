import numpy as np
from mpo.asep import return_mpo
from tools.mps_tools import *
from tools.diag_tools import *
from tools.env_tools import *

# To Do
# - Make modular functions
# - Excited state targeting and state averaging
# - Larger unit cell sizes


# Calculation Details
mbd = 10
nStates = 1
d = 2

# Get the MPO
p = 1.#0.1
alpha = 0.8#0.2      # in at left
gamma = 0.#1.-alpha  # Out at left
q     = 0.#1.-p      # Jump left
beta  = 0.3#0.4     # Out at right
delta = 0.#1.-beta   # In at right
s = -0.5
N = 2

hamParams = np.array([alpha,gamma,p,q,beta,delta,s])
# PH - Convert MPO to infinite version!
mpoList = return_mpo(N,hamParams)

# Initialize by generating random MPS of size 2
mpsList = create_all_mps(N,mbd,nStates)

# Set up empty initial environment
envList = calc_env_inf(mpsList,mpoList,mbd)

converged = False
iterCnt = 0
while not converged:
    # Update Hamiltonian from edge to bulk
    if iterCnt == 1:
        mpoList = return_mpo(N,hamParams)
        nOps = len(mpoList)
        for opInd in range(nOps):
            op = [None]*2
            op[0] = mpoList[opInd][1]
            op[1] = mpoList[opInd][2]
            mpoList[opInd] = op

    # Solve Eigenproblem 1 step
    E,vecs,ovlp = calc_eigs(mpsList,mpoList,envList,0,nStates,
                            alg='davidson',oneSite=False)
    print('My Energy davidson = {}'.format(E))
    E,vecs,ovlp = calc_eigs(mpsList,mpoList,envList,0,nStates,
                            alg='exact',oneSite=False)
    print('My Energy = {}'.format(E[0]))

    # Do SVD
    (_,_,n1,_) = mpoList[0][0].shape
    (_,_,n2,_) = mpoList[0][1].shape
    (n3,_,_) = envList[0][0].shape
    (n4,_,_) = envList[0][1].shape
    for state in range(nStates):
        # Reshape Matrices
        psi = np.reshape(vecs[:,state],(n1,n2,n3,n4))
        psi = np.transpose(psi,(2,0,1,3))
        psi = np.reshape(psi,(n3*n1,n4*n2))
        # Do SVD
        (U,S,V) = np.linalg.svd(psi,full_matrices=False)
        U = np.reshape(U,(n3,n1,-1))
        U = U[:,:,:mbd]
        mpsList[state][0] = np.swapaxes(U,0,1)
        V = np.reshape(V,(-1,n2,n4))
        V = V[:mbd,:,:]
        mpsList[state][1] = np.swapaxes(V,0,1)
        (n4,n5,n6) = mpsList[0][1].shape
        S = S[:mbd]

    # Update Environments
    envList = update_env_inf(mpsList[0],mpoList,envList,mpsl=None)

    # Recontract environments to check energy
    E = einsum('ijk,i,k,ijk->',envList[0][0],S,S.conj(),envList[0][1])
    print('Recontracted Energy = {}'.format(E))

    # Increase system size
    N += 2
    iterCnt += 1

    # Update MPS
    for state in range(nStates):
        (n1,n2,n3) = mpsList[state][0].shape
        mpsList[state][0] = np.pad(mpsList[state][0],((0,0),(0,min(mbd,n3)-n2),(0,min(mbd,n3*d)-n3)),'constant')
        mpsList[state][1] = np.pad(mpsList[state][1],((0,0),(0,min(mbd,n3*d)-n3),(0,min(mbd,n3)-n2)),'constant')

    # Check Convergence

