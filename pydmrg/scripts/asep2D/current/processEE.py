import numpy as np
from sys import argv
from tools.aux.process_states import *
from mpo.asep2D import *

# Get directory name
folder = argv[1]#'saved_states/multiLane_Nx2Ny10mbd10_1548458158/'
# Load results file to get s
npzfile = np.load(folder+'results.npz')
s = npzfile['s']
N = int(npzfile['Nx']*npzfile['Ny'])
nStates = 2
p = 0.1

# Allocate data to hold orthonormalized EE
EEorth = np.zeros((len(s),2,N-1))

for i in range(len(s)):
    mps_fname = folder + 'MPS_s'+str(i)+'_mbd0'
    hamParams = np.array([0.5,0.5,p,1.-p,0.,0.,0.5,0.5,0.,0.,0.5,0.5,0.,s[i]])
    #mpo = return_mpo(N,hamParams)
    EEtmp = calc_entanglement_all(mps_fname,orth=True)
    EEorth[i,:,:] = EEtmp
    print('s = {}, EE = {}'.format(s[i],EEorth[i,:,int(N/2)-1]))
np.savez(folder+'results_OrthEE.npz',s=s,EE=EEorth)
