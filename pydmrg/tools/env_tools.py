import numpy as np
#from pyscf.lib import einsum
einsum = np.einsum
from tools.mps_tools import conj_mps

def alloc_env(M,W,mbd):
    N = len(M)
    # Initialize Empty FL to hold all F lists
    env_lst = []
    for mpoInd in range(len(W)):
        if W[mpoInd][0] is not None:
            _,mbdW,_,_ = W[mpoInd][0].shape
        else:
            mbdW = 1
        F = []
        F.append(np.array([[[1]]]))
        for site in range(int(N/2)):
            F.append(np.zeros((min(2**(site+1),mbd),mbdW,min(2**(site+1),mbd))))
        if N%2 is 1:
            F.append(np.zeros((min(2**(int(N/2)+1),mbd),mbdW,min(2**(int(N/2)+1),mbd))))
        for site in range(int(N/2)-1,0,-1):
            F.append(np.zeros((min(2**(site),mbd),mbdW,min(2**site,mbd))))
        F.append(np.array([[[1]]]))
        # Add environment to env list
        env_lst.append(F)
    return env_lst

def update_envL(M,W,F,site,Ml=None):
    if Ml is None: Ml=conj_mps(M)
    for mpoInd in range(len(W)):
        if W[mpoInd][site] is None:
            tmp1 = einsum('eaf,cdf->eacd',M[site],F[mpoInd][site+1])
            F[mpoInd][site] = einsum('bacy,bxc->xya',tmp1,Ml[site])
        else:
            tmp1 = einsum('eaf,cdf->eacd',M[site],F[mpoInd][site+1])
            tmp2 = einsum('eacd,ydbe->acyb',tmp1,W[mpoInd][site])
            F[mpoInd][site] = einsum('acyb,bxc->xya',tmp2,Ml[site])
    return F

def update_envR(M,W,F,site,Ml=None):
    if Ml is None: Ml=conj_mps(M)
    for mpoInd in range(len(W)):
        if W[mpoInd][site] is None:
            tmp1 = einsum('jlp,ijk->lpik',F[mpoInd][site],Ml[site])
            F[mpoInd][site+1] = einsum('npq,mpnk->kmq',M[site],tmp1)
        else:
            tmp1 = einsum('jlp,ijk->lpik',F[mpoInd][site],Ml[site])
            tmp2 = einsum('lmin,lpik->mpnk',W[mpoInd][site],tmp1)
            F[mpoInd][site+1] = einsum('npq,mpnk->kmq',M[site],tmp2)
    return F

def update_env_inf(mps,mpo,env,mpsl=None):
    # Insert fake center in env
    for mpoInd in range(len(mpo)):
        env[mpoInd].insert(1,[])
    # Update Right Environment (moving left)
    env = update_envL(mps,mpo,env,1,Ml=mpsl)
    for mpoInd in range(len(mpo)):
        env[mpoInd][2] = env[mpoInd][1]
    # Update Left Environment (moving right)
    env = update_envR(mps,mpo,env,0,Ml=mpsl)
    for mpoInd in range(len(mpo)):
        env[mpoInd][0] = env[mpoInd][1]
    # Put empty array as center environment tensor
    for mpoInd in range(len(mpo)):
        env[mpoInd][1] = env[mpoInd][2]
        _ = env[mpoInd].pop()
    return env

def calc_env_inf(M,W,mbd,Ml=None,gaugeSite=0):
    env_lst = []
    for mpoInd in range(len(W)):
        F = []
        F.append(np.array([[[1.]]],dtype=np.complex_))
        F.append(np.array([[[1.]]],dtype=np.complex_))
        env_lst.append(F)
    return env_lst

def calc_env(M,W,mbd,Ml=None,gaugeSite=0):
    # PH - What to do with this gauge site stuff
    N = len(M)
    env_lst = alloc_env(M,W,mbd)
    # Calculate Environment From Right
    for site in range(int(N)-1,gaugeSite,-1):
        env_lst = update_envL(M,W,env_lst,site,Ml=Ml)
    # Calculate Environment from Left
    for site in range(gaugeSite):
        env_lst = update_envR(M,W,env_lst,site,Ml=Ml)
    return env_lst
