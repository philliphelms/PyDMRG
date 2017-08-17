import numpy as np
import matplotlib.pyplot as plt

Nvec = np.array([2,4,6,8,10])#,20,30,40,50])
Svec = np.linspace(-1,1,100)
allE = [None]*len(Nvec)
plt.figure(1)
plt.figure(2)
for i in range(len(Nvec)):
    N = Nvec[i]
    E = np.zeros_like(Svec)
    for j in range(len(Svec)):
        data = np.load('oldData/Results_'+str(N)+'_'+str(j)+'.npz')
        E[j] = data['E'][-1]
    allE[i] = E
    plt.figure(1)
    plt.plot(Svec,E,':o')
    plt.hold(True)
    Ediff = E[1:]-E[:len(E)-1]
    Sdiff = Svec[1:]-Svec[:len(Svec)-1]
    slope = -Ediff/(Sdiff*(N+1))
    plt.figure(2)
    plt.plot(Svec[1:],slope,':o')
    plt.hold(True)
plt.show()
