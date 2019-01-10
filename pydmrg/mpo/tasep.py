mport numpy as np
from mpo.ops import *

############################################################################
# Totally Asymmetric Simple Exclusion Process:

#                     _1_
#           ___ ___ _|_ \/_ ___ ___ ___ ___ ___
# alpha--->|   |   |   |   |   |   |   |   |   |---> beta
#          |___|___|___|___|___|___|___|___|___|
#
# Functions:
#   return_mpo(N,hamParams):
#       hamParams[0] = alpha (in rate at left)
#       hamParams[1] = beta  (out rate at right)
#       hamParams[2] = s     (bias)
###########################################################################

def return_mpo(N,hamParams,periodic=False):
    if periodic:
        return periodic_mpo(N,hamParams)
    else:
        return open_mpo(N,hamParams)

def open_mpo(N,hamParams):
    # Unpack Ham Params ############################
    a = hamParams[0]
    b = hamParams[1]
    s = hamParams[2]
    exp_a = a*np.exp(s)
    exp_b = b*np.exp(s)
    exp_p = 1.*np.exp(s)
    # Create MPO ###################################
    # List to hold all Operators (only one here though)
    mpoL = []
    # Single operator
    mpo = [None]*N
    mpo[0] = np.array([[exp_a*Sm-a*v, exp_p*Sp, -n, I]])
    for site in range(1,N-1):
        mpo[site] = np.array([[I,  z,         z, z],
                              [Sm, z,         z, z],
                              [v,  z,         z, z],
                              [z,  exp_p*Sp, -n, I]])
    mpo[1] = np.array([[I],
                       [Sm],
                       [v],
                       [exp_b*Sp-b*n]])
    # Add single mpo to list of mpos
    mpoL.append(mpo)
    return mpoL

def periodic_mpo(N,hamParams):
    # Unpack Ham Params ############################
    a = hamParams[0]
    b = hamParams[1]
    s = hamParams[2]
    exp_a = a*np.exp(s)
    exp_b = b*np.exp(s)
    exp_p = 1.*np.exp(s)
    # Create MPO ###################################
    # List to hold all Operators (only one here though)
    mpoL = []
    # Main operator
    mpo = [None]*N
    # General operator form:
    gen_mpo = np.array([[I,  z,         z, z],
                        [Sm, z,         z, z],
                        [v,  z,         z, z],
                        [z,  exp_p*Sp, -n, I]])
    mpo[0] = np.expand_dims(gen_mpo[-1,:],0)
    for site in range(1,N-1):
        mpo[i] = gen_mpo
    mpo[-1] = np.expand_dims(gen_mpo[:,0],1)
    mpoL.append(mpo)
    # Periodic terms:
    mpo_p1 = [None]*N
    mpo_p2 = [None]*N
    mpo_p1[-1] = np.array([[exp_p*Sp]])
    mpo_p2[-1] = np.array([[-n]])
    mpo_p1[0] = np.array([[Sm]])
    mpo_p2[0] = np.array([[v]])
    mpoL.append(mpo_p1)
    mpoL.append(mpo_p2)
    return mpoL
