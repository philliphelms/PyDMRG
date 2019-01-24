from dmrg import *
from mpo.asep2D import return_mpo as return_mpo_asep2D
import time
from sys import argv
import os

# Set Calculation Parameters
Ny = int(argv[1])
Nx = int(argv[2])
p = 0.1 
mbd = int(argv[3]) # Can only be a single value currently
ds0 = [0.05,0.001,0.01]
ds_change = [0.,0.1,10]
s_symm = -(Ny-1.)/(2.*(Ny+1.))*np.log(p/(1.-p))
s0 = -0.5
sF = s_symm #+ (s_symm - s0)
make_plt = False
alg = 'davidson'
s_thresh = s_symm
if Ny >= 20:
    s_thresh = 0.1
if Ny >= 30:
    s_thresh = 0.05
if Ny >= 50:
    s_thresh = 0.01

# Allocate Memory for results
E   = np.array([])
EE  = np.array([])
EEl = np.array([])
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
                                s0),   # xy
                                periodicy=False,
                                periodicx=False)
Etmp,EEtmp,gaptmp,env = run_dmrg(mpo,
                                 mbd=mbd,
                                 fname=fname+'s0',
                                 nStates=2,
                                 alg=alg,
                                 returnEnv=True,
                                 calcLeftState=True)
E = np.append(E,Etmp)
EE = np.append(EE,EEtmp[0])
EEl = np.append(EE,EEtmp[1])
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
                                     calcLeftState=True,
                                     orthonormalize=orthonormalize)
    print(sCurr,EEtmp[0],sCurr>s_thresh,EEtmp[0]<0.99)
    if (sCurr > s_thresh) and (EEtmp[0] < 0.99):
        if not orthonormalize:
            # Redo previous calculation
            sCurr -= ds0
            # Start to use orhogonalization
            orthonormalize=True
    else:
        E = np.append(E,Etmp)
        EE = np.append(EE,EEtmp[0])
        EEl = np.append(EEl,EEtmp[1])
        gap = np.append(gap,gaptmp)
        sVec = np.append(sVec,sCurr)
    if sCurr >= ds_change[dsInd]:
        dsInd += 1
    # Create Plots
    if make_plt:
        if len(sVec) > 1:
            curr = np.gradient(E,sVec)#(E[:-1]-E[1:])/(sVec[:-1]-sVec[1:])
            ax1.clear()
            ax1.plot(sVec,curr,'b.')
            ax2.clear()
            ax2.plot(sVec,EE,'b.')
            ax3.clear()
            susc = np.gradient(curr,sVec)
            ax3.plot(sVec,susc,'b.')
            ax4.clear()
            ax4.semilogy(sVec,gap,'b.')
            plt.pause(0.01)
    # Save Results
    np.savez('results/asep2D_stateMatching_psweep_Ny'+str(Ny)+'_Nx'+str(Nx)+'_mbd'+str(mbd),Nx=Nx,Ny=Ny,p=p,mbd=mbd,s=sVec,E=E,EE=EE,EEl=EEl,gap=gap)
    np.savez(path+'results',Nx=Nx,Ny=Ny,p=p,mbd=mbd,s=sVec,E=E,EE=EE,EEl=EEl,gap=gap)
if make_plt:
    plt.show()
