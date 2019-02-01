import numpy as np
import matplotlib.pyplot as plt
from mpo.asep import return_mpo
from tools.contract import full_contract as contract
from sys import argv

# Get initial path from input argument
path = argv[1]

# Load results file and find out initial information
fname = path+'results.npz'
npzfile = np.load(fname)
s = npzfile['s']
N = int(npzfile['N'])
print(s)
nStates = 4
p = 0.1
E = np.zeros((len(s),nStates))
# Loop through all results in s vector
for sInd in range(len(s)):
    mpo = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,s[sInd]))
    for state in range(nStates):
        fname_mps = path+'MPS_s'+str(sInd)+'_mbd0state'+str(state)
        fname_lmps= path+'MPS_s'+str(sInd)+'_mbd0_leftstate'+str(state)
        E[sInd,state] = contract(mps=fname_mps,mpo=mpo)/contract(mps=fname)

plt.plot(s,E[:,0],label='Ground State')
plt.plot(s,E[:,1],label='1st Excited State')
plt.plot(s,E[:,2],label='2nd Excited State')
plt.plot(s,E[:,3],label='3rd Excited State')
plt.show()
