from dmrg import *
from mpo.asep import return_mpo
import time
from sys import argv

# Set Calculation Parameters
N = int(argv[1])
p = 0.1 
mbd = int(argv[2])#np.array([2,4,8,16,32,48,64])
sVec = np.array([-0.05,-0.04,-0.03,-0.02,-0.01,0.,0.01,0.02,0.03,0.04,0.05,
                 -0.045,-0.035,-0.025,-0.015,-0.005,0.005,0.015,0.025,0.035,0.045,
                 -0.0475,-0.0425,-0.0375,-0.0325,-0.0275,-0.0225,-0.0175,-0.0125,-0.0075,-0.0025,
                 0.0025,0.0075,0.0125,0.0175,0.0225,0.0275,0.0325,0.0375,0.0425,0.0475,
                 -0.04825,-0.04675,-0.04375,-0.04175,-0.03825,-0.03675,-0.03375,-0.03175,
                 -0.02825,-0.02675,-0.02375,-0.02175,-0.01825,-0.01675,-0.01375,-0.01175,
                 -0.00825,-0.00675,-0.00375,-0.00175,0.00175,0.00375,0.00675,0.00825,
                 0.01175,0.01375,0.01675,0.01825,0.02175,0.02375,0.02675,0.02825,
                 0.03175,0.03375,0.03675,0.03825,0.04175,0.04375,0.04675,0.04825])
alg = 'davidson'

# Allocate Memory for results
E   = np.zeros((len(sVec)))
EE  = np.zeros((len(sVec)))
gap = np.zeros((len(sVec)))
fname = 'saved_states/simpleMPS_currentBiased_N'+str(N)+'_id'+str(int(time.time()))

# Run Calculations
for sind,s in enumerate(sVec):
    if sind == 0:
        print(s)
        mpo = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,s))
        Etmp,EEtmp,gaptmp = run_dmrg(mpo,
                                     mbd=mbd,
                                     fname=fname,
                                     nStates=2,
                                     alg=alg)
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
                                     alg=alg,
                                     preserveState=False)
        E[sind] = Etmp
        EE[sind] = EEtmp
        gap[sind] = gaptmp
    # Save Results
    np.savez('results/asep_currentBiased_simple_psweep_N'+str(N)+'_Np1_Ns'+str(len(sVec)),N=N,p=p,mbd=mbd,s=sVec,E=E,EE=EE,gap=gap)
