import numpy as np
from tools.mps_tools import *
from tools.env_tools import *

def full_contract(N,mbd,mpo=None,mps=None,lmps=None):
    if mpo is None:
        mpo = [[None]*N]
    # Load matrix product states
    if isinstance(mps,str):
        mps,gSite = load_mps(N,mps)
    if isinstance(lmps,str):
        lmps,glSite = load_mps(N,lmps)
    assert(not ( (lmps is None) and (mps is None)))
    if lmps is None: 
        lmps = conj_mps(mps)
        glSite = gSite
    if mps is None: 
        mps = conj_mps(lmps)
        gSite = glSite
    # Extract lowest state from mps
    lmps = lmps[0]
    mps = mps[0]
    # Ensure both guages are at the same location
    assert(gSite == glSite)

    env = alloc_env(mps,mpo,mbd)
    # Calculate Environment From Right
    for site in range(int(N)-1,-1,-1):
        env = update_envL(mps,mpo,env,site,Ml=lmps)
    Nenv = len(env)
    result = 0
    for j in range(Nenv):
        result += env[j][0][0,0,0]
    return result
