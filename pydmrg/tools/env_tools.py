import numpy as np
from pyscf.lib import einsum
#from tools.mps_tools import *
#from tools.mpo_tools import *
#from tools.diag_tools import *

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
            F.append(np.zeros((min(2**(site+2),mbd),mbdW,min(2**(site+2),mbd))))
        for site in range(int(N/2)-1,0,-1):
            F.append(np.zeros((min(2**(site),mbd),mbdW,min(2**site,mbd))))
        F.append(np.array([[[1]]]))
        # Add environment to env list
        env_lst.append(F)
    return env_lst

def update_envL(M,W,F,site):
    for mpoInd in range(len(W)):
        if W[mpoInd][site] is None:
            tmp1 = einsum('eaf,cdf->eacd',M[site],F[mpoInd][site+1])
            F[mpoInd][site] = einsum('bacy,bxc->xya',tmp1,np.conj(M[site]))
        else:
            tmp1 = einsum('eaf,cdf->eacd',M[site],F[mpoInd][site+1])
            tmp2 = einsum('eacd,ydbe->acyb',tmp1,W[mpoInd][site])
            F[mpoInd][site] = einsum('acyb,bxc->xya',tmp2,np.conj(M[site]))
    return F

def update_envR(M,W,F,site):
    for mpoInd in range(len(W)):
        if W[mpoInd][site] is None:
            tmp1 = einsum('jlp,ijk->lpik',F[mpoInd][site],np.conj(M[site]))
            F[mpoInd][site+1] = einsum('npq,mpnk->kmq',M[site],tmp1)
        else:
            tmp1 = einsum('jlp,ijk->lpik',F[mpoInd][site],np.conj(M[site]))
            tmp2 = einsum('lmin,lpik->mpnk',W[mpoInd][site],tmp1)
            F[mpoInd][site+1] = einsum('npq,mpnk->kmq',M[site],tmp2)
    return F

def calc_env(M,W,mbd,gaugeSite=0):
    # PH - What to do with this gauge site stuff
    N = len(M)
    env_lst = alloc_env(M,W,mbd)
    # Calculate Environment From Right
    for site in range(int(N)-1,gaugeSite,-1):
        env_lst = update_envL(M,W,env_lst,site)
    # Calculate Environment from Left
    for site in range(gaugeSite):
        env_lst = update_envR(M,W,env_lst,site)
    return env_lst
