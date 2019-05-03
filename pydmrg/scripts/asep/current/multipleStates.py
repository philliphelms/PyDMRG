from dmrg import *
from mpo.asep import return_mpo
from mpo.asep import curr_mpo
from tools.contract import full_contract as contract
import time
from sys import argv
import os

# Set Calculation Parameters
N = int(argv[1])
p = 0.1 
mbd = int(argv[2]) # Can only be a single value currently
# Set up bias
ds0 = [0.1,0.001,0.01,0.01]
ds_change = [0.0,0.1,0.2,0.3,0.4]
s_symm = -(N-1.)/(2.*(N+1.))*np.log(p/(1.-p))
s0 = -0.5
sF = 0.5 #s_symm #+ (s_symm - s0)
s_thresh = 1000
# Jumping Rates
alpha = 0.5      # in at left
gamma = 1.-alpha  # Out at left
q     = 1.-p      # Jump left
beta  = 0.5     # Out at right
delta = 1.-beta   # In at right
hamParams = np.array([alpha,gamma,p,q,beta,delta,s0])
# Calculation Settings
make_plt = False
leftState = False
nStates = 2
alg = 'davidson'

# Allocate Memory for results
E   = np.array([])
E0  = np.array([])
E1  = np.array([])
E2  = np.array([])
E3  = np.array([])
EE  = np.array([])
if leftState:
    EEl = np.array([])
    curr = np.array([])
gap = np.array([])
sVec = np.array([])

# Create directory for storing states
dirid = str(int(time.time()))
path = 'saved_states/restartCalc_'+'N'+str(N)+'mbd'+str(mbd)+'_'+dirid+'/'
os.mkdir(path)
fname = path+'MPS_'

# Set up Plotting Stuff
if make_plt:
    import matplotlib.pyplot as plt
    f = plt.figure()
    ax0 = f.add_subplot(311)
    ax1 = f.add_subplot(323)
    ax2 = f.add_subplot(324)
    ax3 = f.add_subplot(325)
    ax4 = f.add_subplot(326)

# Run initial Calculation
print(s0)
hamParams[-1] = s0
mpo = return_mpo(N,hamParams)
Etmp,EEtmp,gaptmp,env = run_dmrg(mpo,
                                 mbd=mbd,
                                 #initGuess = 'saved_states/singleLane_MCSHD_N100mbd10_1553109008/MPS_s242',
                                 fname=fname+'s0',
                                 nStates=nStates,
                                 alg=alg,
                                 returnEnv=True,
                                 calcLeftState=leftState)
"""
Etmp,EEtmp,gaptmp,env = run_dmrg(mpo,
                                 mbd=mbd,
                                 initGuess = fname+'s0',
                                 initEnv = env,
                                 fname=fname+'s0',
                                 nStates=nStates,
                                 alg=alg,
                                 returnEnv=True,
                                 calcLeftState=leftState)
"""
if leftState:
    EE = np.append(EE,EEtmp[0])
    EEl= np.append(EEl,EEtmp[1])
    # Calculate Current
    currMPO = curr_mpo(N,hamParams,singleBond=True)
    opCurr = contract(mpo = currMPO,
                      mps = fname+'s0'+'_mbd0',
                      lmps= fname+'s0'+'_mbd0_left')
    opNorm = contract(mps = fname+'s0'+'_mbd0',
                      lmps= fname+'s0'+'_mbd0_left')
    curr=np.append(curr,opCurr/opNorm*(N+1))
    print('Current = {}'.format(curr[-1]))
else:
    EE = np.append(EE,EEtmp)
# Calculate Higher Energies
E0tmp = contract(mps=fname+'s0'+'_mbd0',mpo=mpo,state=0)/contract(mps=fname+'s0'+'_mbd0',state=0)
E1tmp = contract(mps=fname+'s0'+'_mbd0',mpo=mpo,state=1)/contract(mps=fname+'s0'+'_mbd0',state=1)

# Save Results
E = np.append(E,Etmp)
E0 = np.append(E0,E0tmp)
E1 = np.append(E1,E1tmp)
gap = np.append(gap,gaptmp)
sVec = np.append(sVec,s0)

# Run Calculations
sCurr = s0
orthonormalize=False
dsInd = 0
while sCurr <= sF:
    sCurr += ds0[dsInd]
    # Run Calculation
    hamParams[-1] = sCurr
    mpo = return_mpo(N,hamParams)
    Etmp,EEtmp,gaptmp,env = run_dmrg(mpo,
                                     initEnv=env,
                                     mbd=mbd,
                                     initGuess=fname+'s'+str(len(sVec)-1),
                                     fname=fname+'s'+str(len(sVec)),
                                     alg=alg,
                                     nStates=nStates,
                                     preserveState=False,
                                     returnEnv=True,
                                     calcLeftState=leftState,
                                     orthonormalize=orthonormalize)
    if leftState: 
        EErtmp = EEtmp[0]
    else:
        EErtmp = EEtmp
    if (sCurr > s_thresh) and (EErtmp < 0.99):
        # Check if calculation has failed
        if not orthonormalize:
            # Redo previous calculation
            sCurr -= ds0
            # Start to use orhogonalization
            orthonormalize=True
    else:
        if leftState:
            EE = np.append(EE,EEtmp[0])
            EEl= np.append(EEl,EEtmp[1])
            # Calculate Current
            currMPO = curr_mpo(N,hamParams,singleBond=True)
            opCurr = contract(mpo = currMPO,
                              mps = fname+'s'+str(len(sVec))+'_mbd0',
                              lmps= fname+'s'+str(len(sVec))+'_mbd0_left')
            opNorm = contract(mps = fname+'s'+str(len(sVec))+'_mbd0',
                              lmps= fname+'s'+str(len(sVec))+'_mbd0_left')
            curr=np.append(curr,opCurr/opNorm*(N+1))
            print('Current = {}'.format(curr[-1]))
        else:
            EE = np.append(EE,EEtmp)
        # Calculate Higher Energies
        E0tmp = contract(mps=fname+'s'+str(len(sVec))+'_mbd0',mpo=mpo,state=0)/contract(mps=fname+'s'+str(len(sVec))+'_mbd0',state=0)
        E1tmp = contract(mps=fname+'s'+str(len(sVec))+'_mbd0',mpo=mpo,state=1)/contract(mps=fname+'s'+str(len(sVec))+'_mbd0',state=1)

        # Save Results
        E = np.append(E,Etmp)
        E0 = np.append(E0,E0tmp)
        E1 = np.append(E1,E1tmp)
        gap = np.append(gap,gaptmp)
        sVec = np.append(sVec,sCurr)
    if sCurr >= ds_change[dsInd]:
        dsInd += 1
    # Create Plots
    if make_plt:
        if len(sVec) > 1:
            ax0.clear()
            ax0.plot(sVec,E0)
            ax0.plot(sVec,E1)
            ax0.plot(sVec,E2)
            ax0.plot(sVec,E3)
            currCalc = np.gradient(E,sVec)#(E[:-1]-E[1:])/(sVec[:-1]-sVec[1:])
            ax1.clear()
            if leftState: ax1.plot(sVec,curr,'r.')
            ax1.plot(sVec,currCalc,'b.')
            ax2.clear()
            ax2.plot(sVec,EE,'b.')
            if leftState: ax2.plot(sVec,EEl,'r.')
            ax3.clear()
            suscCalc = np.gradient(currCalc,sVec)
            if leftState: susc = np.gradient(curr,sVec)
            if leftState: ax3.plot(sVec,susc,'r.')
            ax3.plot(sVec,suscCalc,'b.')
            ax4.clear()
            ax4.semilogy(sVec,gap,'b.')
            plt.pause(0.01)
    # Save Results
    if leftState:
        np.savez(path+'results',N=N,p=p,mbd=mbd,s=sVec,E=E,EE=EE,gap=gap,EEl=EEl,curr=curr)
    else:
        np.savez(path+'results',N=N,p=p,mbd=mbd,s=sVec,E=E,EE=EE,gap=gap)
if make_plt:
    plt.show()
