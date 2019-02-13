import numpy as np
from tools.mps_tools import *
from tools.env_tools import *

def full_contract(mpo=None,mps=None,lmps=None,state=None,lstate=None,gSite=None,glSite=None):
    # Load matrix product states
    if isinstance(mps,str):
        mps,gSite = load_mps(mps)
    if isinstance(lmps,str):
        lmps,glSite = load_mps(lmps)
    assert(not ( (lmps is None) and (mps is None)))
    if lmps is None: 
        lmps = conj_mps(mps)
        glSite = gSite
    if mps is None: 
        mps = conj_mps(lmps)
        gSite = glSite
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
    # Ensure both guages are at the same location
    assert(gSite == glSite)

    env = alloc_env(mps_ss,mpo,mbd)
    # Calculate Environment From Right
    for site in range(int(N)-1,-1,-1):
        env = update_envL(mps_ss,mpo,env,site,Ml=lmps_ss)
    Nenv = len(env)
    result = 0
    for j in range(Nenv):
        result += env[j][0][0,0,0]
    return result
