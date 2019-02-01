from dmrg import *
from mpo.asep import return_mpo
import time
from sys import argv
import matplotlib.pyplot as plt
from tools.mpo_tools import mpo2mat
np.set_printoptions(linewidth=1000)

# Set Calculation Parameters
N = int(argv[1])
p = 0.1 
mbd = int(argv[2])#np.array([2,4,8,16,32,48,64])
s_symm = -(N-1.)/(2.*(N+1.))*np.log(p/(1.-p))
print('symmetric point = {}'.format(s_symm))
print(s_symm)
print(np.linspace(-1,s_symm,30))
print(np.linspace(s_symm,2.*s_symm-1.,30))
s = np.concatenate((np.linspace(-1,s_symm,30),np.linspace(s_symm,2.*s_symm+1.,30)))
E = np.zeros(len(s))
EEl = np.zeros(len(s))
EEr = np.zeros(len(s))
alg = 'exact'

# Set up figures
f = plt.figure()
ax1 = f.add_subplot(211)
ax2 = f.add_subplot(212)
ax1.plot(np.array([s_symm,s_symm]),np.array([0,0.2]),'k:')
ax2.plot(np.array([s_symm,s_symm]),np.array([0,2]),'k:')

# Do this stuff at center site
for sInd in range(len(s)):
    print('s = {}'.format(s[sInd]))
    #mpo1 = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,s[sInd]*(N+1)/(N-1)))
    mpo1 = return_mpo(N,(0.5,0.5,p,1.-p,0.5,0.5,s[sInd]))
    Etmp,EEtmp,gap = run_dmrg(mpo1,
                        mbd=mbd,
                        fname='saved_states/entanglement_symmetry',
                        nStates=2,
                        alg=alg,
                        calcLeftState=True)
    print(Etmp)
    print(E)
    E[sInd] = Etmp
    EEr[sInd] = EEtmp[0]
    EEl[sInd] = EEtmp[1]
    ax1.clear()
    ax1.plot(s,EEr,'b.')
    ax1.plot(s,EEl,'r.')
    ax2.clear()
    ax2.plot(s,E,'k.')
    plt.pause(0.0001)
np.savez('Entanglement_withGC_N4.npz',s=s,E=E,EEr=EEr,EEl=EEl)
plt.show()
