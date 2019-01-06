from dmrg import *
from mpo.asep import return_mpo
import time
from sys import argv
from os import rename

# 10 - 0.22
# 20 - 0.05
# 30 - 0.03

# Set Calculation Parameters
N = 20
p = 0.1 
mbd = 10 # Can only be a single value currently
ds0 = 0.01
ds_min = 1e-5
s_symm = -(N-1.)/(2.*(N+1.))*np.log(p/(1.-p))
s0 = 0.
sF = s_symm #+ (s_symm - s0)
s_threshold = 0.05
ee_threshold = 0.99
ovlp_tol = 0.999
make_plt = True

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
                                 alg='exact',
                                 returnEnv=True)
E = np.append(E,Etmp)
EE = np.append(EE,EEtmp)
gap = np.append(gap,gaptmp)
sVec = np.append(sVec,s0)

# Run Calculations
sCurr = s0
while sCurr <= sF:
    sCurr += ds0
    # Check Overlap from previous s point
    passed = False
    dsi = ds0
    bestEE = 0
    best_ds = None
    while not passed:
        print('Trying s = {}, ds = {}'.format(sCurr,dsi))
        site = int(N/2)
        mpo = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,sCurr))
        Etmp,EEtmp,gaptmp,envtmp = run_dmrg(mpo,env=env,
                                             mbd=mbd,
                                             initGuess=fname,
                                             fname=fname+'tmp',
                                             alg='exact',
                                             nStates=2,
                                             preserveState=False,
                                             minIter = 3,
                                             maxIter = 5,
                                             returnEnv=True)
        if (sCurr < s_threshold) or (EEtmp > ee_threshold) or (forcedPass):
            passed = True
            env = envtmp
            rename(fname+'tmp_mbd0.npz',fname+'_mbd0.npz')
        else:
            # The state does not overlap, we need a smaller step size
            sCurr -= dsi
            if dsi < ds_min:
                sCurr += ds_min
                passed = True
                print('Failed to Match State')
                if bestEE > EEtmp:
                    dsi = best_ds
                    sCurr += dsi
                    passed = False
                    forcedPass = True
            else:
                if EEtmp > bestEE:
                    bestEE = EEtmp
                    best_ds = dsi
                dsi /= 2.
                sCurr += dsi
    E = np.append(E,Etmp)
    EE = np.append(EE,EEtmp)
    gap = np.append(gap,gaptmp)
    sVec = np.append(sVec,sCurr)
    # Create Plots
    if make_plt:
        #ax1.clear()
        #ax1.plot(sVec,E,'b.')
        curr = np.gradient(E,sVec)#(E[:-1]-E[1:])/(sVec[:-1]-sVec[1:])
        #splt_curr = sVec[:-1]+0.5*(sVec[:-1]-sVec[1:])
        ax1.clear()
        ax1.plot(sVec,curr,'b.')
        ax2.clear()
        ax2.plot(sVec,EE,'b.')
        ax3.clear()
        susc = np.gradient(curr,sVec)
        ax3.plot(sVec,susc,'b.')
        ax4.clear()
        ax4.semilogy(sVec,gap,'b.')
        # Plot around symmetric point
        #ax1.plot(s_symm + (s_symm - sVec),E,'b.')
        #ax1.plot(s_symm + (s_symm - sVec) - dsi,-curr,'b.')
        #ax2.plot(s_symm + (s_symm - sVec),EE,'b.')
        #ax3.plot(s_symm + (s_symm - sVec),susc,'b.')
        #ax4.semilogy(s_symm + (s_symm - sVec),gap,'b.')
        #ax2.clear()
        #ax2.plot(sVec[:sind],EE[:sind],'.')
        #susc = (curr[:-1]-curr[1:])/(sVec[0]-sVec[1])
        #splt_susc = splt_curr[:-1]+0.5*(sVec[1]-sVec[0])
        #ax3.clear()
        #ax3.plot(splt_susc[:sind-1],susc[:sind-1],'.')
        #ax4.clear()
        #ax4.semilogy(sVec[:sind],gap[:sind],'.')
        plt.pause(0.01)
    # Save Results
    np.savez('results/asep_stateMatching_psweep_N'+str(N)+'_Np1_Ns'+str(len(sVec)),N=N,p=p,mbd=mbd,s=sVec,E=E,EE=EE,gap=gap)
if make_plt:
    plt.show()
