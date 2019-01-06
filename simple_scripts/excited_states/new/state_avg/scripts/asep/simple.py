from dmrg import *
from mpo.asep import return_mpo
import time
from sys import argv
import matplotlib.pyplot as plt

# Set Calculation Parameters
N = 6
p = 0.1 
mbd = 10#np.array([2,4,8,16,32,48,64])
sVec = np.linspace(0.7847,0.78475,20)

# Allocate Memory for results
E   = np.zeros((len(sVec)))
EE  = np.zeros((len(sVec)))
gap = np.zeros((len(sVec)))
fname = 'saved_states/simpleMPS_N'+str(N)+'_id'+str(int(time.time()))


f = plt.figure()
ax1 = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)
# Run Calculations
for sind,s in enumerate(sVec):
    if sind == 0:
        print(s)
        mpo = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,s))
        Etmp,EEtmp,gaptmp = run_dmrg(mpo,
                                     mbd=mbd,
                                     fname=fname,
                                     nStates=2,
                                     alg='exact')
        print(Etmp)
        print(EEtmp)
        print(gaptmp)
        E[sind] = Etmp
        EE[sind] = EEtmp
        gap[sind] = gaptmp
    else:
        print(s)
        mpo = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,s))
        Etmp,EEtmp,gaptmp = run_dmrg(mpo,
                                     mbd=mbd,
                                     initGuess=fname,
                                     fname=fname,
                                     nStates=2,
                                     alg='exact',
                                     preserveState=True)
        E[sind] = Etmp
        EE[sind] = EEtmp
        gap[sind] = gaptmp
    # Create Plots
    curr = (E[:-1]-E[1:])/(sVec[0]-sVec[1])
    splt_curr = sVec[:-1]+0.5*(sVec[1]-sVec[0])
    ax1.clear()
    #ax1.plot(splt_curr[:sind],curr[:sind],'.')
    ax1.plot(sVec[:sind],E[:sind])
    ax2.clear()
    ax2.plot(sVec[:sind],EE[:sind],'.')
    susc = (curr[:-1]-curr[1:])/(sVec[0]-sVec[1])
    splt_susc = splt_curr[:-1]+0.5*(sVec[1]-sVec[0])
    ax3.clear()
    ax3.plot(splt_susc[:sind-1],susc[:sind-1],'.')
    ax4.clear()
    ax4.semilogy(sVec[:sind],gap[:sind],'.')
    plt.pause(0.01)
    # Save Results
    np.savez('results/asep_simple_psweep_N'+str(N)+'_Np1_Ns'+str(len(sVec)),N=N,p=p,mbd=mbd,s=sVec,E=E,EE=EE,gap=gap)
plt.show()
