import numpy as np
from mpo.ops import *
import collections

############################################################################
# East Model
#
# Hamiltonian: 
#   W = \sum_i (n_{i-1})[ e^{-\lambda} ( c\sigma_i^+ + (1-c)\sigma_i^-)
#                                       -c(1-n_i)    - (1-c)n_i        ]
#
# Functions:
#   return_mpo(N,hamParams):
#       hamParams[0] = c      (flip rate)
#       hamParams[1] = s      (bias)
###########################################################################

def return_mpo(N,hamParams):
    if not isinstance(hamParams[0],(collections.Sequence,np.ndarray)):
        hamParams = val2vecParams(N,hamParams)
    else:
        hamParams = extractParams(N,hamParams)
    return open_mpo(N,hamParams)

def open_mpo(N,hamParams):
    # Extract parameter values
    (c,s) = hamParams
    # List to hold all mpos
    mpoL = []
    # Main mpo
    mpo = [None]*N
    for site in range(N):
        # Generic Operator Form
        gen_mpo = np.array([[I,                                                                    z, z],
                            [c[site]*(np.exp(-s[site])*Sp-v)+(1.-c[site])*(np.exp(-s[site])*Sm-n), z, z],
                            [z,                                                                    n, I]])
        # Add operator to mpo
        if (site == 0):
            # Ensure First site is occupied
            gen_mpo[-1,0,:,:] = c[site]*(np.exp(-s[site])*Sp-v)+(1.-c[site])*(np.exp(-s[site])*Sm-n)
            mpo[site] = np.expand_dims(gen_mpo[-1,:],0)
        elif (site == N-1):
            mpo[site] = np.expand_dims(gen_mpo[:,0],1)
        else:
            mpo[site] = gen_mpo
    # Include in list of mpos
    mpoL.append(mpo)
    return mpoL

# USEFUL FUNCTIONS ------------------------------------------------------

def val2vecParams(N,hamParams):
    # Extract values
    if not isinstance(hamParams[0],(collections.Sequence,np.ndarray)):
        c = float(hamParams[0])
        cVec = c*np.ones(N,dtype=np.float_)
        cVec[0] = c
    else:
        cVec = c
    if not isinstance(hamParams[1],(collections.Sequence,np.ndarray)):
        s = float(hamParams[1])
        sVec = s*np.ones(N,dtype=np.float_)
    else:
        sVec = s
    # Convert to vectors
    returnParams = (cVec,sVec)
    return returnParams

def extractParams(N,hamParams):
    c = hamparams[0].astype(dtype=np.float_)
    s = hamparams[1].astype(dtype=np.float_)
    return (c,s)

# ACTIVITY OPERATORS ------------------------------------------------------

def act_mpo(N,hamParams,singleSite=False,site=None):
    if not isinstance(hamParams[0],(collections.Sequence,np.ndarray)):
        hamParams = val2vecParams(N,hamParams)
    else:
        hamParams = extractParams(N,hamParams)
    if singleSite:
        return single_site_act(N,hamParams,bond=bond)
    else:
        return open_act(N,hamParams)

def single_site_act(N,hamParams,site=None):
    # Extract parameter values
    (a,g,p,q,b,d,s) = hamParams
    (ea,eg,ep,eq,eb,ed) = exponentiateBias(hamParams)
    # Decide which bond to measure current over
    if bond is None:
        bond = int(N/2)
    # List to hold corresponding mpos
    mpoL = []
    mpo = [None]*N
    # Fill in mpo
    mpo[site] = np.array([[c[site]*np.exp(-s[site])*Sp+(1.-c[site])*(np.exp(-s[site])*Sm)]])
    mpo[site+1] = np.array([[n]])
    # Include in list of mpos
    mpoL.append(mpo)
    return mpoL

def open_act(N,hamParams):
    # Extract parameter values
    (c,s) = hamParams
    # List to hold all mpos
    mpoL = []
    # Main mpo
    mpo = [None]*N
    for site in range(N):
        # Generic Operator Form
        gen_mpo = np.array([[I,                                                                z, z],
                            [c[site]*(np.exp(-s[site])*Sp)+(1.-c[site])*(np.exp(-s[site])*Sm), z, z],
                            [z,                                                                n, I]])
        # Add operator to mpo
        if (site == 0):
            # Ensure First site is occupied
            gen_mpo[-1,0,:,:] = c[site]*(np.exp(-s[site])*Sp)+(1.-c[site])*(np.exp(-s[site])*Sm)
            mpo[site] = np.expand_dims(gen_mpo[-1,:],0)
        elif (site == N-1):
            mpo[site] = np.expand_dims(gen_mpo[:,0],1)
        else:
            mpo[site] = gen_mpo
    # Include in list of mpos
    mpoL.append(mpo)
    return mpoL
