import numpy as np
from mpo.ops import *
import collections

############################################################################
# General Asymmetric Simple Exclusion Process:
#
#                     _p_
#           ___ ___ _|_ \/_ ___ ___ ___ ___ ___
# alpha--->|   |   |   |   |   |   |   |   |   |---> beta
# gamma<---|___|___|___|___|___|___|___|___|___|<--- delta
#                   /\___|
#                      q
# Functions:
#   return_mpo(N,hamParams):
#       hamParams[0] = alpha  (In at left)
#       hamParams[1] = gamma  (Out at left)
#       hamParams[2] = p      (Forward at site)
#       hamParams[2] = q      (Backward at site)
#       hamParams[2] = beta   (Out at right)
#       hamParams[2] = delta  (In at right)
#       hamParams[2] = s      (bias)
###########################################################################

def return_mpo(N,hamParams,periodic=False):
    if not isinstance(hamParams[0],(collections.Sequence,np.ndarray)):
        hamParams = val2vecParams(N,hamParams)
    else:
        hamParams = extractParams(N,hamParams)
    if periodic:
        return periodic_mpo(N,hamParams)
    else:
        return open_mpo(N,hamParams)

def open_mpo(N,hamParams):
    # Extract parameter values
    (a,g,p,q,b,d,s) = hamParams
    (ea,eg,ep,eq,eb,ed) = exponentiateBias(hamParams)
    # List to hold all mpos
    mpoL = []
    # Main mpo
    mpo = [None]*N
    for site in range(N):
        # Generic Operator Form
        gen_mpo = np.array([[I,              z,  z,  z,  z, z],
                            [ep[site-1]*Sm,  z,  z,  z,  z, z],
                            [p[site-1]*v,    z,  z,  z,  z, z],
                            [eq[site]*Sp,    z,  z,  z,  z, z],
                            [q[site]*n,      z,  z,  z,  z, z],
                            [z,             Sp, -n, Sm, -v, I]])
        # Include destruction & creation at site
        gen_mpo[-1,0,:,:] += (ea[site] + ed[site])*Sm -\
                             ( a[site] +  d[site])*v  +\
                             (eb[site] + eg[site])*Sp -\
                             ( b[site] +  g[site])*n
        # Add operator to mpo
        if (site == 0):
            mpo[site] = np.expand_dims(gen_mpo[-1,:],0)
        elif (site == N-1):
            mpo[site] = np.expand_dims(gen_mpo[:,0],1)
        else:
            mpo[site] = gen_mpo
    # Include in list of mpos
    mpoL.append(mpo)
    return mpoL

def periodic_mpo(N,hamParams):
    # Extract parameter values
    (a,g,p,q,b,d,s) = hamParams
    (ea,eg,ep,eq,eb,ed) = exponentiateBias(hamParams)
    # Get main mpo from open_mpo function
    mpoL = open_mpo(N,hamParams)
    if p[-1] != 0:
        tmp_op1 = [None]*N
        tmp_op2 = [None]*N
        tmp_op1[-1] = np.array([[ep[-1]*Sp]])
        tmp_op2[-1] = np.array([[-p[-1]*n]])
        tmp_op1[0] = np.array([[Sm]])
        tmp_op2[0] = np.array([[v]])
        mpoL.append(tmp_op1)
        mpoL.append(tmp_op2)
    if q[0] != 0:
        tmp_op1 = [None]*N
        tmp_op2 = [None]*N
        tmp_op1[-1] = np.array([[eq[0]*Sm]])
        tmp_op2[-1] = np.array([[-q[0]*v]])
        tmp_op1[0] = np.array([[Sp]])
        tmp_op2[0] = np.array([[n]])
        mpoL.append(tmp_op1)
        mpoL.append(tmp_op2)
    return mpoL

def exponentiateBias(hamParams):
    (a,g,p,q,b,d,s) = hamParams
    ea = a*np.exp(s)
    eg = g*np.exp(-s)
    ep = p*np.exp(s)
    eq = q*np.exp(-s)
    eb = b*np.exp(s)
    ed = d*np.exp(-s)
    return (ea,eg,ep,eq,eb,ed)

def val2vecParams(N,hamParams):
    # Extract values
    a = float(hamParams[0])
    g = float(hamParams[1])
    p = float(hamParams[2])
    q = float(hamParams[3])
    b = float(hamParams[4])
    d = float(hamParams[5])
    s = float(hamParams[6])
    # Convert to vectors
    aVec = np.zeros(N,dtype=np.float_)
    aVec[0] = a
    gVec = np.zeros(N,dtype=np.float_)
    gVec[0] = g
    pVec = p*np.ones(N,dtype=np.float_)
    qVec = q*np.ones(N,dtype=np.float_)
    bVec = np.zeros(N,dtype=np.float_)
    bVec[-1] = b
    dVec = np.zeros(N,dtype=np.float_)
    dVec[-1] = d
    sVec = s*np.ones(N,dtype=np.float_)
    returnParams = (aVec,gVec,pVec,qVec,bVec,dVec,sVec)
    return returnParams

def extractParams(N,hamParams):
    a = hamparams[0].astype(dtype=np.float_)
    g = hamparams[1].astype(dtype=np.float_)
    p = hamparams[2].astype(dtype=np.float_)
    q = hamparams[3].astype(dtype=np.float_)
    b = hamparams[4].astype(dtype=np.float_)
    d = hamparams[5].astype(dtype=np.float_)
    s = hamparams[6].astype(dtype=np.float_)
    return (a,g,p,q,b,d,s)
