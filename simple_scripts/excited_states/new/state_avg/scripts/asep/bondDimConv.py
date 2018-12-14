from dmrg import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpo.asep import return_mpo

# Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
plt.style.use('fivethirtyeight') #'fivethirtyeight'
colormap = cm.plasma #coolwarm, inferno, viridis

# Set Calculation Parameters
N = 10
p = 0.1 
mbd = np.array([2,4,6,8,10])
sVec = np.linspace(-0.5,0.5,100)

# Allocate Memory for results
E   = np.zeros((len(sVec),len(mbd)))
EE  = np.zeros((len(sVec),len(mbd)))
gap = np.zeros((len(sVec),len(mbd)))

# Set up Figures
f = plt.figure()
ax1 = f.add_subplot(161)
ax2 = f.add_subplot(162)
ax3 = f.add_subplot(163)
ax4 = f.add_subplot(164)
ax5 = f.add_subplot(165)
ax6 = f.add_subplot(166)

# Run Calculations
for sind,s in enumerate(sVec):
    if sind == 0:
        print(s)
        mpo = return_mpo(N,(0.5,0.5,p,1-p,0.5,0.5,s))
        E[sind,:],EE[sind,:],gap[sind,:] = run_dmrg(mpo,mbd=mbd,fname='mps/myMPS_N'+str(N),nStates=2)
    else:
        print(s)
        mpo = return_mpo(N,(0.5,0.5,p,1-p,0.5,0.5,s))
        E[sind,:],EE[sind,:],gap[sind,:] = run_dmrg(mpo,mbd=mbd,initGuess='mps/myMPS_N'+str(N),fname='mps/myMPS_N'+str(N),nStates=2)
    # Plot Results
    ax1.clear()
    for i in range(len(mbd)):
        ax1.plot(sVec[:sind],E[:sind,i])
    ax2.clear()
    for i in range(len(mbd)):
        ax2.semilogy(sVec[:sind],np.abs(E[:sind,i]-E[:sind,-1]))
    curr = (E[0:-1,:]-E[1:,:])/(sVec[0]-sVec[1])
    splt = sVec[1:]
    ax3.clear()
    for i in range(len(mbd)):
        ax3.plot(splt[:sind],curr[:sind,i])
    susc = (curr[0:-1,:]-curr[1:,:])/(sVec[0]-sVec[1])
    splt = sVec[1:-1]
    ax4.clear()
    for i in range(len(mbd)):
        ax4.plot(splt[:sind-1],susc[:sind-1,i])
    ax5.clear()
    for i in range(len(mbd)):
        ax5.plot(sVec[:sind],EE[:sind,i])
    ax6.clear()
    for i in range(len(mbd)):
        ax6.semilogy(sVec[:sind],gap[:sind,i])
    plt.pause(0.01)
    # Save Results
    np.savez('results/asep_psweep_N'+str(N)+'_Np1_Ns'+str(len(sVec)),N=N,p=p,mbd=mbd,s=sVec,E=E,EE=EE,gap=gap)
plt.show()

