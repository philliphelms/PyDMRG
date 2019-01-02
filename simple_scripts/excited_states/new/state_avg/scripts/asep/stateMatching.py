from dmrg import *
from mpo.asep import return_mpo
import time
from sys import argv

# Set Calculation Parameters
N = 50
p = 0.1 
mbd = 10 # Can only be a single value currently
s0 = 0.0
ds0 = 0.01
sF = 1.0
ovlp_tol = 0.99
make_plt = False

# Allocate Memory for results
E   = np.array([])
EE  = np.array([])
gap = np.array([])
sVec = np.array([])
fname = 'mps/stateMatchingMPS_N'+str(N)+'_id'+str(int(time.time()))

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
Etmp,EEtmp,gaptmp = run_dmrg(mpo,
                             mbd=mbd,
                             fname=fname,
                             nStates=2,
                             alg='exact')
print(len(E))
print(len(sVec))
E = np.append(E,Etmp)
EE = np.append(EE,EEtmp)
gap = np.append(gap,gaptmp)
sVec = np.append(sVec,s0)
print(len(E))
print(len(sVec))
# Run Calculations
sCurr = s0
while sCurr <= sF:
    sCurr += ds0
    # Check Overlap from previous s point
    passed = False
    dsi = ds0
    while not passed:
        print('Trying s = {}'.format(sCurr))
        site = int(N/2)
        mpo = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,sCurr))
        mps,gSite = load_mps(N,fname+'_mbd0')
        env = calc_env(mps,mpo,mbd,gaugeSite=gSite)
        _,v,ovlp = calc_eigs(mps,mpo,env,site,2,alg='exact',preserveState=False)
        if ovlp > ovlp_tol:
            # The state is similar and we can keep going in the sweep
            passed = True
        else:
            # The state does not overlap, we need a smaller step size
            sCurr -= dsi
            dsi /= 2.
            sCurr += dsi
    # Run Actual Calculation
    mpo = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,sCurr))
    Etmp,EEtmp,gaptmp = run_dmrg(mpo,
                                 mbd=mbd,
                                 initGuess=fname,
                                 fname=fname,
                                 nStates=2,
                                 alg='exact',
                                 preserveState=True)
    print(len(E))
    print(len(sVec))
    E = np.append(E,Etmp)
    EE = np.append(EE,EEtmp)
    gap = np.append(gap,gaptmp)
    sVec = np.append(sVec,sCurr)
    # Create Plots
    if make_plt:
        ax1.clear()
        ax1.plot(sVec,E)
        ax2.clear()
        ax2.plot(sVec,EE)
        ax3.clear()
        ax3.semilogy(sVec,gap)
        #curr = (E[:-1]-E[1:])/(sVec[0]-sVec[1])
        #splt_curr = sVec[:-1]+0.5*(sVec[1]-sVec[0])
        #ax1.clear()
        #ax1.plot(splt_curr[:sind],curr[:sind],'.')
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
