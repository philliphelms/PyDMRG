from dmrg import *
from mpo.asep import return_mpo
import time
from sys import argv

# Set Calculation Parameters
N = int(argv[1])
p = 0.1 
mbd = 10 # Can only be a single value currently
ds0 = 0.001
s_symm = -(N-1.)/(2.*(N+1.))*np.log(p/(1.-p))
s0 = -0.1
sF = 2*N*s_symm #+ (s_symm - s0)
make_plt = True
alg = 'davidson'
s_thresh = s_symm
if N > 20:
    s_thresh = 0.1
if N > 30:
    s_thresh = 0.05
if N > 50:
    s_thresh = 0.001

# Allocate Memory for results
E   = np.array([])
EE  = np.array([])
gap = np.array([])
sVec = np.array([])
fname = 'saved_states/stateMatchingMPS_N'+str(N)+'_id'+str(int(time.time()))

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
mpo = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,s0))
Etmp,EEtmp,gaptmp,env = run_dmrg(mpo,
                                 mbd=mbd,
                                 fname=fname,
                                 nStates=2,
                                 alg=alg,
                                 returnEnv=True)
E = np.append(E,Etmp)
EE = np.append(EE,EEtmp)
gap = np.append(gap,gaptmp)
sVec = np.append(sVec,s0)

# Run Calculations
sCurr = s0
orthonormalize=False
while sCurr <= sF:
    sCurr += ds0
    # Run Calculation
    print('Running s = {}'.format(sCurr))
    mpo = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,sCurr))
    Etmp,EEtmp,gaptmp,env = run_dmrg(mpo,initEnv=env,
                                     mbd=mbd,
                                     initGuess=fname,
                                     fname=fname,
                                     alg=alg,
                                     nStates=2,
                                     preserveState=False,
                                     returnEnv=True,
                                     orthonormalize=orthonormalize)
    if (sCurr > s_thresh) and (EEtmp < 0.99):
        if not orthonormalize:
            # Redo previous calculation
            sCurr -= ds0
            # Start to use orhogonalization
            orthonormalize=True
    else:
        E = np.append(E,Etmp)
        EE = np.append(EE,EEtmp)
        gap = np.append(gap,gaptmp)
        sVec = np.append(sVec,sCurr)
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
    np.savez('results/asep_stateMatching_psweep_N'+str(N)+'_Np1_Ns'+str(len(sVec)),N=N,p=p,mbd=mbd,s=sVec,E=E,EE=EE,gap=gap)
if make_plt:
    plt.show()
