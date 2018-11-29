import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm # Colormaps
from sys import argv


# Specify Filename
filename = argv[1]#'1543442402_asep_bond_dim_check_N20_data_p1_s10_bd3.npz'

# Load Data
npzfile = np.load(filename)
print(npzfile.keys())
s = npzfile['s']
p = npzfile['p']
bd= npzfile['bd']
E = npzfile['CGF']
EE= npzfile['EE']
EEs=npzfile['EEs']

# Convert to real
E = np.real(E)
EE= np.real(EE)
EEs=np.real(EEs)

# Convert to 1D arrays
E = E[0,:,:]
EE= EE[0,:,:]
print(EEs.shape)
EEs=EEs[0,:,:,:]

# Specify plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
plt.style.use('seaborn-bright') #'fivethirtyeight'
colormap = cm.plasma #coolwarm, inferno, viridis

# CGF Converges Graph (s on x-axis)
if True:
    fig = plt.figure()
    ax = fig.gca()
    ns,nBD = E.shape
    for i in range(nBD):
        ax.semilogy(s,np.abs((E[:,i]-E[:,-1])/E[:,-1]),'o-',label='$M='+str(bd[i])+'$',color=colormap(i/nBD))
    ax.legend()

# CGF Convergence Graph (M on x-axis)
if True:
    fig = plt.figure()
    ax = fig.gca()
    ns,nBD = E.shape
    for i in range(ns):
        ax.semilogy(bd,np.abs((E[i,:]-E[i,-1])/E[i,-1]),'o-',label='$s='+str(s[i])+'$',color=colormap(i/ns))
    ax.legend()

# CGF Graph (using Largest M)
if True:
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(s,E[:,-1])

# Current Graph (using largest M)
if True:
    fig = plt.figure()
    ax = fig.gca()
    CGF = E[:,-1]
    curr = (CGF[1:]-CGF[:-1])/(s[0]-s[1])
    splt = s[1:]
    ax.plot(splt,curr,color=colormap(0.5))

# Susceptibility Graph (using largest M)
if True:
    fig = plt.figure()
    ax = fig.gca()
    CGF = E[:,-1]
    curr = (CGF[1:]-CGF[:-1])/(s[0]-s[1])
    splt = s[1:]
    susc = (curr[1:]-curr[:-1])/(s[0]-s[1])
    splt = splt[1:]
    ax.plot(splt,susc,color=colormap(0.5))

# Entanglement Entropy Graph (Using Largest M)
if True:
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(s,EE[:,-1],color=colormap(0.5))

# Entanglement Spectrum Graph (Using Largest M)
if True:
    fig = plt.figure()
    ax = fig.gca()
    EEs = EEs[:,-1,:]
    ns,nEEs = EEs.shape
    for i in range(nEEs):
        ax.plot(s,EEs[:,i],label='Schmidt Val '+str(i),color=colormap(i/nEEs))
    print(EEs)
    ax.legend()

plt.show()
