from dmrg import *
from mpo.asep2D import return_mpo as return_mpo_asep2D
from tools.contract import full_contract as contract
import time
from sys import argv
import os
from mpo.asep2D import curr_mpo
from mpo.asep2D_activity import act_mpo

# Get directory name
folder = argv[1]
# Load results file to get s
npzfile = np.load(folder+'results.npz')
s = npzfile['s']
N = int(npzfile['Nx']*npzfile['Ny'])
Nx = npzfile['Nx']
Ny = npzfile['Ny']
nStates = 2 
p = 0.1

# Allocate Memory for results
curr = np.array([])
currx = np.array([])
curry = np.array([])
act = np.array([])
actx = np.array([])
acty = np.array([])

print('s\tCurr\t\tCurrx\t\t\tCurry\t\tAct\t\tActx\t\tActy')
for i in range(len(s)):
    fname = folder + 'MPS_s'+str(i)+'_mbd0'
    hamParams = np.array([0.5,0.5,p,1.-p,0.,0.,0.5,0.5,0.,0.,0.5,0.5,0.,s[i]])
    # Calculate Currents & Activities =======================
    # xy current
    currMPO = curr_mpo((Nx,Ny),hamParams,periodicy=False,periodicx=False,includex=True,includey=True)
    opCurr = contract(mpo = currMPO,
                        mps = fname,
                        lmps= fname+'_left')
    opNorm = contract(mps = fname,
                        lmps= fname+'_left')
    curr=np.append(curr,opCurr/opNorm)
    #print('Current = {}'.format(curr[-1]))
    # x current
    currMPO = curr_mpo((Nx,Ny),hamParams,periodicy=False,periodicx=False,includex=True,includey=False)
    opCurr = contract(mpo = currMPO,
                        mps = fname,
                        lmps= fname+'_left')
    opNorm = contract(mps = fname,
                        lmps= fname+'_left')
    currx=np.append(currx,opCurr/opNorm)
    #print('Current (x dir) = {}'.format(currx[-1]))
    # y current
    currMPO = curr_mpo((Nx,Ny),hamParams,periodicy=False,periodicx=False,includex=False,includey=True)
    opCurr = contract(mpo = currMPO,
                        mps = fname,
                        lmps= fname+'_left')
    opNorm = contract(mps = fname,
                        lmps= fname+'_left')
    curry=np.append(curry,opCurr/opNorm)
    #print('Current (y dir) = {}'.format(curry[-1]))
    # xy activity
    actMPO = act_mpo((Nx,Ny),hamParams,periodicy=False,periodicx=False,includex=True,includey=True)
    opAct = contract(mpo = actMPO,
                        mps = fname,
                        lmps= fname+'_left')
    opNorm = contract(mps = fname,
                        lmps= fname+'_left')
    act=np.append(act,opAct/opNorm)
    #print('Activity = {}'.format(act[-1]))
    # x activity
    actMPO = act_mpo((Nx,Ny),hamParams,periodicy=False,periodicx=False,includex=True,includey=False)
    opAct = contract(mpo = actMPO,
                        mps = fname,
                        lmps= fname+'_left')
    opNorm = contract(mps = fname,
                        lmps= fname+'_left')
    actx=np.append(actx,opAct/opNorm)
    #print('Activity (x dir) = {}'.format(actx[-1]))
    # y activity
    actMPO = act_mpo((Nx,Ny),hamParams,periodicy=False,periodicx=False,includex=False,includey=True)
    opAct = contract(mpo = actMPO,
                        mps = fname,
                        lmps= fname+'_left')
    opNorm = contract(mps = fname,
                        lmps= fname+'_left')
    acty=np.append(acty,opAct/opNorm)
    #print('Activity (y dir) = {}'.format(acty[-1]))
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(s[i],np.real(curr[-1]),np.real(currx[-1]),np.real(curry[-1]),np.real(act[-1]),np.real(actx[-1]),np.real(acty[-1])))
