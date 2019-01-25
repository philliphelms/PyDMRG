from dmrg import *
from mpo.asep2D import return_mpo as return_mpo_asep2D
from tools.contract import full_contract as contract
import time
from sys import argv
import os
from mpo.asep2D import curr_mpo

# Set Calculation Parameters
Ny = int(argv[1])
Nx = int(argv[2])
p = 0.1 
mbd = int(argv[3]) # Can only be a single value currently
ds0 = [0.05,0.001,0.01]
ds0 = [0.1,0.1,0.1]
ds_change = [0.,0.2,10]
s_symm = -(Ny-1.)/(2.*(Ny+1.))*np.log(p/(1.-p))
s0 = -0.5
sF = s_symm #+ (s_symm - s0)
make_plt = False
alg = 'davidson'
leftState = True
s_thresh = sF+10
#if Ny > 10:
#    s_thresh = 0.3
#if Ny >= 20:
#    s_thresh = 0.1
#if Ny >= 30:
#    s_thresh = 0.05
#if Ny >= 50:
#    s_thresh = 0.01

# Allocate Memory for results
E   = np.array([])
EE  = np.array([])
if leftState: 
    EEl = np.array([])
    curr = np.array([])
gap = np.array([])
sVec = np.array([])

# Create Directory for saving states
dirid = str(int(time.time()))
path = 'saved_states/multiLane_'+'Nx'+str(Nx)+'Ny'+str(Ny)+'mbd'+str(mbd)+'_'+dirid+'/'
os.mkdir(path)
fname = path+'MPS_'

# Set up Plotting Stuff
if make_plt:
    import matplotlib.pyplot as plt
    f = plt.figure()
    ax1 = f.add_subplot(221)
    ax2 = f.add_subplot(222)
    ax3 = f.add_subplot(223)
    ax4 = f.add_subplot(224)

# Run initial Calculation
print(s0)
hamParams = np.array([0.5,0.5,p,1.-p,0.,0.,0.5,0.5,0.,0.,0.5,0.5,0.,s0])
mpo = return_mpo_asep2D((Nx,Ny),hamParams,periodicy=False,periodicx=False)
Etmp,EEtmp,gaptmp,env = run_dmrg(mpo,
                                 mbd=mbd,
                                 fname=fname+'s0',
                                 nStates=2,
                                 alg=alg,
                                 returnEnv=True,
                                 calcLeftState=leftState)
if leftState:
    EE = np.append(EE,EEtmp[0])
    EEl= np.append(EEl,EEtmp[1])
    # Calculate Current
    currMPO = curr_mpo((Nx,Ny),hamParams,periodicy=False,periodicx=False)
    opCurr = contract(Nx*Ny,mbd,
                        mpo = currMPO,
                        mps = fname+'s0'+'_mbd0',
                        lmps= fname+'s0'+'_mbd0_left')
    opNorm = contract(Nx*Ny,mbd,
                        mps = fname+'s0'+'_mbd0',
                        lmps= fname+'s0'+'_mbd0_left')
    curr=np.append(curr,opCurr/opNorm)
    print('Current = {}'.format(curr[-1]))
else:
    EE = np.append(EE,EEtmp)
E = np.append(E,Etmp)
gap = np.append(gap,gaptmp)
sVec = np.append(sVec,s0)

# Run Calculations
sCurr = s0
orthonormalize=False
dsInd = 0
while sCurr <= sF:
    sCurr += ds0[dsInd]
    # Run Calculation
    print('Running s = {}'.format(sCurr))
    mpo = return_mpo_asep2D((Nx,Ny), (0.5,  # jr
                                    0.5,  # jl
                                    p,    # ju
                                    1.-p, # jd
                                    0.,   # cr
                                    0.,   # cl
                                    0.5,  # cu
                                    0.5,  # cd
                                    0.,   # dr
                                    0.,   # dl
                                    0.5,  # du
                                    0.5,  # dd
                                    0.,   # sx
                                    sCurr),   # xy
                                    periodicy=False,
                                    periodicx=False)
    Etmp,EEtmp,gaptmp,env = run_dmrg(mpo,
                                     initEnv=env,
                                     initGuess=fname+'s'+str(len(sVec)-1),
                                     mbd=mbd,
                                     fname=fname+'s'+str(len(sVec)),
                                     nStates=2,
                                     alg=alg,
                                     returnEnv=True,
                                     preserveState=False,
                                     calcLeftState=leftState,
                                     orthonormalize=orthonormalize)
    if leftState: EErtmp = EEtmp[0]
    else: EErtmp = EEtmp
    if (sCurr > s_thresh) and (EErtmp < 0.99):
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
            currMPO = curr_mpo((Nx,Ny),hamParams,periodicy=False,periodicx=False)
            opCurr = contract(Nx*Ny,mbd,
                                mpo = currMPO,
                                mps = fname+'s'+str(len(sVec))+'_mbd0',
                                lmps= fname+'s'+str(len(sVec))+'_mbd0_left')
            opNorm = contract(Nx*Ny,mbd,
                                mps = fname+'s'+str(len(sVec))+'_mbd0',
                                lmps= fname+'s'+str(len(sVec))+'_mbd0_left')
            curr=np.append(curr,opCurr/opNorm)
            print('Current = {}'.format(curr[-1]))
        else:
            EE = np.append(EE,EEtmp)
        E = np.append(E,Etmp)
        gap = np.append(gap,gaptmp)
        sVec = np.append(sVec,s0)
    if sCurr >= ds_change[dsInd]:
        dsInd += 1
    # Create Plots
    if make_plt:
        if len(sVec) > 1:
            currCalc = np.gradient(E,sVec)#(E[:-1]-E[1:])/(sVec[:-1]-sVec[1:])
            ax1.clear()
            ax1.plot(sVec,currCalc,'b.')
            if leftState: ax1.plot(sVec,curr,'r.')
            ax2.clear()
            ax2.plot(sVec,EE,'b.')
            if leftState: ax2.plot(sVec,EEl,'r.')
            ax3.clear()
            suscCalc = np.gradient(currCalc,sVec)
            if leftState: susc = np.gradient(curr,sVec)
            ax3.plot(sVec,suscCalc,'b.')
            if leftState: ax3.plot(sVec,susc,'r.')
            ax4.clear()
            ax4.semilogy(sVec,gap,'b.')
            plt.pause(0.01)
    # Save Results
    if leftState:
        np.savez('results/asep2D_stateMatching_psweep_Ny'+str(Ny)+'_Nx'+str(Nx)+'_mbd'+str(mbd),Nx=Nx,Ny=Ny,p=p,mbd=mbd,s=sVec,E=E,EE=EE,EEl=EEl,gap=gap)
        np.savez(path+'results',Nx=Nx,Ny=Ny,p=p,mbd=mbd,s=sVec,E=E,EE=EE,EEl=EEl,gap=gap)
    else:
        np.savez('results/asep2D_stateMatching_psweep_Ny'+str(Ny)+'_Nx'+str(Nx)+'_mbd'+str(mbd),Nx=Nx,Ny=Ny,p=p,mbd=mbd,s=sVec,E=E,EE=EE,gap=gap)
        np.savez(path+'results',Nx=Nx,Ny=Ny,p=p,mbd=mbd,s=sVec,E=E,EE=EE,gap=gap)
if make_plt:
    plt.show()
