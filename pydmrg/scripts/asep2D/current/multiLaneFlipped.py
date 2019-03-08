from dmrg import *
from mpo.asep2D import return_mpo as return_mpo_asep2D
from tools.contract import full_contract as contract
import time
from sys import argv
import os
from mpo.asep2D import curr_mpo,act_mpo

# Collect inputs
Ny = int(argv[1])  # System size y-dir 
Nx = int(argv[2])  # System size x-dir
mbd = int(argv[3]) # Maximum Bond Dimension
bcs = str(argv[4]) # Boundary Condition (periodic, closed, open) x-direction

# Set Calculation Parameters
p = 0.1 
s_symm = -(Ny-1.)/(2.*(Ny+1.))*np.log(p/(1.-p))
s0 = -0.32
sF = -0.28#s_symm 
ds0 = [0.01]
ds_change = [10]
s_thresh = sF+10
make_plt = False
alg = 'davidson'
leftState = True

# If periodic, then say so
if bcs == 'periodic':
    periodicy = True
    periodicx = False
else:
    periodicy = False
    periodicx = False

# Allocate Memory for results
E   = np.array([])
EE  = np.array([])
if leftState: 
    EEl = np.array([])
    curr = np.array([])
    act = np.array([])
gap = np.array([])
sVec = np.array([])

# Create Directory for saving states
dirid = str(int(time.time()))
path = 'saved_states/MCy_'+bcs+'_multiLaneFlipped_'+'Nx'+str(Nx)+'Ny'+str(Ny)+'mbd'+str(mbd)+'_'+dirid+'/'
os.mkdir(path)
fname = path+'MPS_'

# Set up Plotting Stuff
if make_plt:
    import matplotlib.pyplot as plt
    f = plt.figure()
    ax1 = f.add_subplot(231)
    ax2 = f.add_subplot(232)
    ax3 = f.add_subplot(233)
    ax4 = f.add_subplot(234)
    ax5 = f.add_subplot(235)
    ax6 = f.add_subplot(236)

# Run initial Calculation
print(s0)
if bcs == 'closed':
    #                     jr,  jl, ju, jd, cr, cl, cu, cd, dr, dl, du, dd,sx,sy
    hamParams = np.array([p ,1.-p,0.5,0.5,0.5,0.5,0.0,0.0,0.5,0.5,0.0,0.0,s0,0.])
    mpo = return_mpo_asep2D((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)
elif bcs == 'open':
    #                     jr,  jl, ju, jd, cr, cl, cu, cd, dr, dl, du, dd,sx,sy
    hamParams = np.array([p ,1.-p,0.5,0.5,0.5,0.5,0.9,0.1,0.5,0.5,0.9,0.1,s0,0.])
    mpo = return_mpo_asep2D((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)
elif bcs == 'periodic':
    hamParams = np.array([p,1.-p,0.5,0.5,0.5,0.5,0.0,0.0,0.5,0.5,0.0,0.0,s0,0.])
    mpo = return_mpo_asep2D((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)
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
    currMPO = curr_mpo((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)
    opCurr = contract(mpo = currMPO,
                        mps = fname+'s0'+'_mbd0',
                        lmps= fname+'s0'+'_mbd0_left')
    opNorm = contract(mps = fname+'s0'+'_mbd0',
                        lmps= fname+'s0'+'_mbd0_left')
    curr=np.append(curr,opCurr/opNorm)
    print('Current = {}'.format(curr[-1]))
    actMPO = act_mpo((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx,includex=True,includey=False)
    opAct = contract(mpo = actMPO,
                        mps = fname+'s'+str(len(sVec))+'_mbd0',
                        lmps= fname+'s'+str(len(sVec))+'_mbd0_left')
    act=np.append(act,opAct/opNorm)
    print('Activity = {}'.format(act[-1]))
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
    if bcs == 'closed':
        hamParams = np.array([p,1.-p,0.5,0.5,0.5,0.5,0.0,0.0,0.5,0.5,0.0,0.0,sCurr,0.])
        mpo = return_mpo_asep2D((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)
    elif bcs == 'open':
        #hamParams = np.array([p,1.-p,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,sCurr,0.])
        hamParams = np.array([p ,1.-p,0.5,0.5,0.5,0.5,0.9,0.1,0.5,0.5,0.9,0.1,sCurr,0.])
        mpo = return_mpo_asep2D((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)
    elif bcs == 'periodic':
        hamParams = np.array([p,1.-p,0.5,0.5,0.5,0.5,0.0,0.0,0.5,0.5,0.0,0.0,sCurr,0.])
        mpo = return_mpo_asep2D((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)
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
            sCurr -= ds0[dsInd]
            # Start to use orhogonalization
            orthonormalize=True
    if leftState:
        EE = np.append(EE,EEtmp[0])
        EEl= np.append(EEl,EEtmp[1])
        # Calculate Current
        currMPO = curr_mpo((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)
        opCurr = contract(mpo = currMPO,
                            mps = fname+'s'+str(len(sVec))+'_mbd0',
                            lmps= fname+'s'+str(len(sVec))+'_mbd0_left')
        opNorm = contract(mps = fname+'s'+str(len(sVec))+'_mbd0',
                            lmps= fname+'s'+str(len(sVec))+'_mbd0_left')
        curr=np.append(curr,opCurr/opNorm)
        print('Current = {}'.format(curr[-1]))
        actMPO = act_mpo((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx,includex=True,includey=False)
        opAct = contract(mpo = actMPO,
                            mps = fname+'s'+str(len(sVec))+'_mbd0',
                            lmps= fname+'s'+str(len(sVec))+'_mbd0_left')
        act=np.append(act,opAct/opNorm)
        print('Activity = {}'.format(act[-1]))
    else:
        EE = np.append(EE,EEtmp)
    E = np.append(E,Etmp)
    gap = np.append(gap,gaptmp)
    sVec = np.append(sVec,sCurr)
    if sCurr >= ds_change[dsInd]:
        dsInd += 1
    # Create Plots
    if make_plt:
        if len(sVec) > 1:
            ax1.clear()
            ax1.plot(sVec,E,'r.')
            currCalc = np.gradient(E,sVec)#(E[:-1]-E[1:])/(sVec[:-1]-sVec[1:])
            ax2.clear()
            ax2.plot(sVec,currCalc,'b.')
            if leftState: ax2.plot(sVec,curr,'r.')
            ax3.clear()
            ax3.plot(sVec,EE,'b.')
            if leftState: ax3.plot(sVec,EEl,'r.')
            ax4.clear()
            suscCalc = np.gradient(currCalc,sVec)
            if leftState: susc = np.gradient(curr,sVec)
            ax4.plot(sVec,suscCalc,'b.')
            if leftState: ax4.plot(sVec,susc,'r.')
            ax5.clear()
            ax5.semilogy(sVec,gap,'b.')
            ax6.clear()
            ax6.plot(sVec,act,'r.')
            plt.pause(0.01)
    # Save Results
    if leftState:
        np.savez(path+'results',Nx=Nx,Ny=Ny,p=p,mbd=mbd,s=sVec,E=E,EE=EE,EEl=EEl,gap=gap,act=act,hamParams=hamParams)
    else:
        np.savez(path+'results',Nx=Nx,Ny=Ny,p=p,mbd=mbd,s=sVec,E=E,EE=EE,gap=gap,hamParams=hamParams)
if make_plt:
    plt.show()
