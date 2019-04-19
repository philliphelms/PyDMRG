import numpy as np
from tools.mps_tools import *
from tools.env_tools import *

def full_contract(mpo=None,mps=None,lmps=None,state=None,lstate=None,orth=False,gSite=None,glSite=None):
    # Load matrix product states
    if isinstance(mps,str):
        mps,gSite = load_mps(mps)
    if isinstance(lmps,str):
        lmps,glSite = load_mps(lmps)
    assert(not ( (lmps is None) and (mps is None)))
    if lmps is None: 
        lmps = conj_mps(mps)
    if mps is None: 
        mps = conj_mps(lmps)
    # Orthonormalize if Needed
    if orth:
        mps = orthonormalize_states(mps,gSite=gSite)
        lmps= orthonormalize_states(lmps,gSite=glSite)
    # Figure out size of mps
    N = nSites(mps)
    mbd = maxBondDim(mps)
    # Extract lowest state from mps
    if (state is None) and (lstate is None):
        state,lstate = 0,0
    elif state is None:
        state = lstate
    elif lstate is None:
        lstate = state
    lmps_ss = lmps[lstate]
    mps_ss = mps[state]
    # Make empty mpo if none is provided
    if mpo is None:
        mpo = [[None]*N]
    # Create empty environment
    env = alloc_env(mps_ss,mpo,mbd)
    # Calculate Environment From Right
    for site in range(int(N)-1,-1,-1):
        env = update_envL(mps_ss,mpo,env,site,Ml=lmps_ss)
    Nenv = len(env)
    result = 0
    for j in range(Nenv):
        result += env[j][0][0,0,0]
    return result

def inf_contract(mpo=None,mps=None,lmps=None,state=None,lstate=None,orth=False):
    # Load matrix product states
    if isinstance(mps,str):
        mps,_ = load_mps(mps)
    if isinstance(lmps,str):
        lmps,_ = load_mps(lmps)
    assert(not ( (lmps is None) and (mps is None)))
    if lmps is None: 
        lmps = conj_mps(mps)
    if mps is None: 
        mps = conj_mps(lmps)
    # Orthonormalize if Needed
    if orth:
        print('Orthonormalization not implemented for inf MPS')
    # Figure out size of mps
    N = nSites(mps)
    mbd = maxBondDim(mps)
    # Extract lowest state from mps
    if (state is None) and (lstate is None):
        state,lstate = 0,0
    elif state is None:
        state = lstate
    elif lstate is None:
        lstate = state
    # Make empty mpo if none is provided
    if mpo is None:
        mpo = [[None]*N]
    # Contract
    nops = len(mpo)
    result = 0
    for opInd in range(nops):
        if mpo[opInd][0] is None:
            mpo[opInd][0] = np.array([[np.eye(1)]])
        if mpo[opInd][1] is None:
            mpo[opInd][1] = np.array([[np.eye(1)]])
        result += einsum('ijk,lkm,nopi,oNql,pjr,qrm->',mps[state][0],mps[state][1],mpo[opInd][0],mpo[opInd][1],lmps[lstate][0],lmps[lstate][1])
    return result
