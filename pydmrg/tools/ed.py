import numpy as np
from tools.mpo_tools import mpo2mat

def ed(mpo):
    # Get system size
    nOp = len(mpo)
    nSite = len(mpo[0])
    # Get Hamiltonian
    H = mpo2mat(mpo)
    # Diagonalize Hamiltonian
    u,v = np.linalg.eig(H)
    # Sort
    inds = np.argsort(u)[::-1]
    u = u[inds]
    #for i in range(len(u)):
    #    print(u[i])
    v = v[:,inds]
    # Print Results
    return u,v
