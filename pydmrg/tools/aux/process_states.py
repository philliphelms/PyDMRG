import numpy as np
from tools.mps_tools import *
from tools.contract import full_contract as contract

def calc_entanglement_all(mps,mpo=None,orth=True):
    # Load matrix product states
    if isinstance(mps,str):
        mps,gSite = load_mps(mps)
    print(gSite)
    N = len(mps[0])
    nStates = len(mps)
    # Orthonormalize states if desired
    if orth: 
        mpsUse = orthonormalize_states(mps,mpo=mpo,gSite=gSite)
    else:
        mpsUse = mps
    # Calculate Entanglement Entropy
    EE = np.zeros((nStates,N-1))
    for state in range(nStates):
        for site in range(gSite-1,N-1):
            mpstmp,EEtmp = move_gauge_right(mps[state],site,returnEE=True)
            EE[state,site] = EEtmp
            mps[state] = mpstmp
        for site in range(N-1,0,-1):
            mpstmp,EEtmp = move_gauge_left(mps[state],site,returnEE=True)
            EE[state,site-1] = EEtmp
            mps[state] = mpstmp
    return EE

def calc_density_all(mps,lmps,orth=True,state=0):
    # Load matrix product states
    if isinstance(mps,str):
        mps,gSite = load_mps(mps)
    if isinstance(lmps,str):
        lmps,glSite = load_mps(lmps)
    N = len(mps[0])
    nStates = len(mps)
    # Orthonormalize first two states if needed
    if orth:
        mps = orthonormalize_states(mps,gSite=gSite)
        lmps= orthonormalize_states(lmps,gSite=glSite)
    # Calculate Density
    density = np.zeros(N)
    for site in range(0,N):
        # Create mpo for density at site 
        mpo = [[None]*N]
        mpo[0][site] = np.array([[[[0.,0.], 
                              [0.,1.]]]])
        density[site] = contract(mps=mps,lmps=lmps,mpo=mpo,state=state)/contract(mps=mps,lmps=lmps,state=state)
    return density
