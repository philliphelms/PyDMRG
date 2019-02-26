import numpy as np
from sys import argv
from tools.aux.process_states import *
from mpo.asep import *
from dmrg import *

# Get directory name
folder = argv[1]#'saved_states/multiLane_Nx2Ny10mbd10_1548458158/'
# Load results file to get s
npzfile = np.load(folder+'results.npz')
ind = int(argv[2])
s = npzfile['s']
N = int(npzfile['N'])
EE = npzfile['EE']
for i in range(len(s)):
    print(i,s[i],EE[i])
nStates = 2
p = 0.1
alg='davidson'

mbd = np.array([10,20,50,75,100,150,200])
maxIter=np.array([1,3,3,3,3,3,3])
mps_fname = folder+'MPS_s'+str(ind)

hamParams = np.array([0.5,0.5,p,1.-p,0.5,0.5,s[ind]])
mpo = return_mpo(N,hamParams)
_ = run_dmrg(mpo,
             mbd=mbd,
             initGuess=mps_fname,
             fname=mps_fname,
             alg=alg,
             nStates=nStates)
