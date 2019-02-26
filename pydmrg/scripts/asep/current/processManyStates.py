import numpy as np
import matplotlib.pyplot as plt
from mpo.asep import return_mpo, curr_mpo
from tools.contract import full_contract as contract
from sys import argv

# Get initial path from input argument
path = argv[1]

# Load results file and find out initial information
fname = path+'results.npz'
npzfile = np.load(fname)
s = npzfile['s']
N = int(npzfile['N'])
nStates = 4
p = 0.1
E = np.zeros((len(s),nStates))
PT = np.zeros((len(s),nStates))
# Loop through all results in s vector
for sInd in range(len(s)):
    print('Processing s={}'.format(s[sInd]))
    mpo = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,s[sInd]))
    cmpo= curr_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,s[sInd]))
    for state in range(nStates):
        fname_mps = path+'MPS_s'+str(sInd)+'_mbd0'
        fname_lmps= path+'MPS_s'+str(sInd)+'_mbd0_left'
        E[sInd,state] = contract(mps=fname_mps,mpo=mpo,state=state)/contract(mps=fname_mps,state=state)
        # Calculate PT Stuff
        #PT[sInd,state]= contract(mps=fname_mps,mpo=cmpo,state=0,lstate=state)*contract(mps=fname_mps,mpo=cmpo,state=state,lstate=0)/(E[sInd,0]-E[sInd,state])
        PT[sInd,state]= contract(mps=fname_mps,lmps=fname_lmps,mpo=cmpo,state=0,lstate=state)*contract(mps=fname_mps,lmps=fname_lmps,mpo=cmpo,state=state,lstate=0)/(E[sInd,0]-E[sInd,state])
    print(PT[sInd,:])
plt.figure()
plt.plot(s,E[:,0],label='Ground State')
plt.plot(s,E[:,1],label='1st Excited State')
plt.plot(s,E[:,2],label='2nd Excited State')
plt.plot(s,E[:,3],label='3rd Excited State')
plt.legend()
plt.figure()
plt.plot(s,PT[:,1],label='1st ES')
plt.plot(s,PT[:,2],label='2nd ES')
plt.plot(s,PT[:,3],label='3rd ES')
plt.legend()
plt.show()
