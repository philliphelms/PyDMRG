import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# For the TASEP model, this script calculations the current and cumulant 
# generating function as a function of s. This is for a single value of alpha
# and beta, which are specified on line 39.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

# Run TASEP Current Calculations
N_vec = np.array([30])#,20,30,40,50,60,70,80,90,100])
s_vec = np.linspace(-1,1,100)
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
fig5 = plt.figure()
col_vec = ['r','r','y','g','b','c','k','m']
for j in range(len(N_vec)):
    N = N_vec[j]
    print('Running Calcs for N={}'.format(N))
    Evec = np.zeros(s_vec.shape)
    Evec_adj = np.zeros(s_vec.shape)
    EE = np.zeros(s_vec.shape)
    for i in range(len(s_vec)):
        print('\tRunning Calcs for s={}'.format(s_vec[i]))
        x = mps_opt.MPS_OPT(N =int(N),
                              hamType = "tasep",
                              plotExpVals = True,
                              hamParams = (0.35,s_vec[i],2/3))
        Evec[i] = x.kernel()
        Evec_adj[i] = Evec[i]/(N+1)
        print(x.entanglement_entropy)
        EE[i] = x.entanglement_entropy[int(N/2)]
    Ediff = Evec[1:]-Evec[:len(Evec)-1]
    Sdiff = s_vec[1:]-s_vec[:len(s_vec)-1]
    slope = -Ediff/(Sdiff)
    plt.figure(fig1.number)
    plt.plot(s_vec,Evec,col_vec[j]+'-',linewidth=3)
    plt.xlabel('$s$',fontsize=20)
    plt.ylabel('$\mu$',fontsize=20)
    plt.figure(fig2.number)
    plt.plot(s_vec,Evec_adj,col_vec[j]+'-',linewidth=3)
    plt.xlabel('$s$',fontsize=20)
    plt.ylabel('$\mu/(N+1)$',fontsize=20)
    plt.figure(fig3.number)
    plt.plot(s_vec[1:],slope,col_vec[j]+'-',linewidth=3)
    plt.xlabel('$s$',fontsize=20)
    plt.ylabel('$\partial_s\mu=J$',fontsize=20)
    plt.figure(fig4.number)
    plt.plot(s_vec[1:],slope/(N+1),col_vec[j]+'-',linewidth=3)
    plt.xlabel('$s$',fontsize=20)
    plt.ylabel('$\partial_s\mu/(N+1)$',fontsize=20)
    plt.figure(fig5.number)
    plt.plot(s_vec,EE,col_vec[j]+'-',linewidth=3)
    plt.xlabel('$s$',fontsize=20)
    plt.ylabel('Entanglemente Entropy',fontsize=20)
fig1.savefig('varyS_CGF.pdf')
fig2.savefig('varyS_scaledCGF.pdf')
fig3.savefig('varyS_current.pdf')
fig4.savefig('varyS_scaledCurrent.pdf')
fig5.savefig('varyS_entanglementEntropy.pdf')
