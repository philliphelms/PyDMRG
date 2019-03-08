import numpy as np
from sys import argv
from tools.aux.process_states import *
from mpo.asep2D import *

# Get directory name
folder = argv[1]#'saved_states/multiLane_Nx2Ny10mbd10_1548458158/'
# Load results file to get s
npzfile = np.load(folder+'results.npz')
s = npzfile['s']
N = int(npzfile['Nx'])*int(npzfile['Ny'])
nStates = 2
p = 0.1
for i in range(len(s)):
    print(i,s[i])

# Allocate data structures
E       = np.zeros((len(s),2))
EE      = np.zeros((len(s),2,N-1))
EEorth  = np.zeros((len(s),2,N-1))
EEl     = np.zeros((len(s),2,N-1))
EElorth = np.zeros((len(s),2,N-1))
rho     = np.zeros((len(s),N))
rhoOrth = np.zeros((len(s),N))
act     = np.zeros((len(s),N-1))
actOrth = np.zeros((len(s),N-1))
cur     = np.zeros((len(s),N-1))
curOrth = np.zeros((len(s),N-1))

for i in range(len(s)-1,0,-1):
    print('{}/{}'.format(i+1,len(s)))
    # Specify new file name
    mps_fname = folder + 'MPS_s'+str(i)+'_mbd0'
    lmps_fname= folder + 'MPS_s'+str(i)+'_mbd0_left'
    # Calculate Energy
    #hamParams = np.array([0.5,0.5,p,1.-p,0.5,0.5,s[i]])
    #mpo = return_mpo(N,hamParams)
    #for state in range(nStates):
    #    E[i,state] = contract(mps=mps_fname,mpo=mpo,state=state)/contract(mps=mps_fname,state=state)
    #print('E',E[i,:])
    # Calculate Entanglement Entropies
    EE     [i,:,:] = calc_entanglement_all(mps_fname ,orth=False)
    print('EE',EE[i,0,int(N/2)])
    EEl    [i,:,:] = calc_entanglement_all(lmps_fname,orth=False)
    print('EEl',EEl[i,0,int(N/2)])
    EEorth [i,:,:] = calc_entanglement_all(mps_fname ,orth=True)
    print('EEorth',EEorth[i,0,int(N/2)])
    EElorth[i,:,:] = calc_entanglement_all(lmps_fname,orth=True)
    print('EElorth',EElorth[i,0,int(N/2)])
    # Calculate Densities
    rho    [i,:] = calc_density_all(mps_fname,lmps_fname,orth=False,state=0)
    print('rho',rho[i,:])
    rhoOrth[i,:] = 0.5*(calc_density_all(mps_fname,lmps_fname,orth=True,state=0) + calc_density_all(mps_fname,lmps_fname,orth=True,state=1))
    print('rhoOrth',rhoOrth[i,:])
    # Calculate Activities
    """
    for site in range(N-1):
        mpo = act_mpo(N,hamParams,singleBond=True,bond=site)
        act    [i,site] = contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=False)/contract(mps=mps_fname,lmps=lmps_fname,orth=False)
        actOrth[i,site] = contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=True )/contract(mps=mps_fname,lmps=lmps_fname,orth=True )
    #print('act',act[i,:])
    #print('actOrth',actOrth[i,:])
    # Calculate Current
    for site in range(N-1):
        mpo = curr_mpo(N,hamParams,singleBond=True,bond=site)
        cur    [i,site] = contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=False)/contract(mps=mps_fname,lmps=lmps_fname,orth=False)
        curOrth[i,site] = contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=True )/contract(mps=mps_fname,lmps=lmps_fname,orth=True )
    #print('cur',cur[i,:])
    #print('curOrth',curOrth[i,:])
    """
# Save Results
np.savez(folder+'observables.npz',s=s,E=E,EE=EE,EEorth=EEorth,EEl=EEl,EElorth=EElorth,rho=rho,rhoOrth=rhoOrth,act=act,actOrth=actOrth,cur=cur,curOrth=curOrth)
