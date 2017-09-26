import numpy as np
import matplotlib.pyplot as plt

Nvec = np.array([10,20,30,40,50,60])#,20,30,40,50])
Svec = np.linspace(-1,1,20)
allE = [None]*len(Nvec)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.style.use('ggplot')

pltstyle = 'b:o'
plt.figure(1)
plt.figure(2)
for i in range(len(Nvec)):
    N = Nvec[i]
    E = np.zeros_like(Svec)
    for j in range(len(Svec)):
        data = np.load('Results_'+str(N)+'_'+str(j)+'.npz')
        E[j] = data['E'][-1]
    allE[i] = E
    Ediff = E[1:]-E[:len(E)-1]
    Evec_adj = E/(N+1)
    Sdiff = Svec[1:]-Svec[:len(Svec)-1]
    slope = -Ediff/(Sdiff*(N+1))
    Evec = E
    plt.figure(1)
    plt.plot(Svec,Evec,pltstyle)
    plt.hold(True)
    plt.xlabel('$s$',fontsize=20)
    plt.ylabel('$\mu$',fontsize=20)
    plt.figure(2)
    plt.plot(Svec,Evec_adj,pltstyle)
    plt.hold(True)
    plt.xlabel('$s$',fontsize=20)
    plt.ylabel('$\mu/(N+1)$',fontsize=20)
    plt.ylim((-0.1,0.6))
    plt.xlim((-1,1))
    plt.figure(3)
    plt.plot(Svec[1:],slope,pltstyle)
    plt.xlabel('$s$',fontsize=20)
    plt.ylabel('$\partial_s\mu/(N+1)$',fontsize=20)
    plt.hold(True)
    plt.ylim((0,1))
    plt.xlim((-1,1))
plt.show()
