from dmrg import *
from mpo.asep import return_mpo
import time
from sys import argv

# Set Calculation Parameters
N = 30
p = 0.1 
mbd = 10#np.array([2,4,8,16,32,48,64])
sVec = np.linspace(-0.5,0.5,10)

# Allocate Memory for results
E   = np.zeros((len(sVec)))
EE  = np.zeros((len(sVec)))
gap = np.zeros((len(sVec)))
fname = 'mps/simpleMPS_N'+str(N)+'_id'+str(int(time.time()))

# Run Calculations
for sind,s in enumerate(sVec):
    if sind == 0:
        print(s)
        mpo = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,s))
        Etmp,EEtmp,gaptmp = run_dmrg(mpo,mbd=mbd,fname=fname,nStates=2,alg='davidson')
        print(Etmp)
        print(EEtmp)
        print(gaptmp)
        E[sind] = Etmp
        EE[sind] = EEtmp
        gap[sind] = gaptmp
    else:
        print(s)
        mpo = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,s))
        Etmp,EEtmp,gaptmp = run_dmrg(mpo,mbd=mbd,initGuess=fname,fname=fname,nStates=2,alg='davidson')
        E[sind] = Etmp
        EE[sind] = EEtmp
        gap[sind] = gaptmp
    # Save Results
    np.savez('results/asep_simple_psweep_N'+str(N)+'_Np1_Ns'+str(len(sVec)),N=N,p=p,mbd=mbd,s=sVec,E=E,EE=EE,gap=gap)
