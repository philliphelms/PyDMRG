import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm # Colormaps

filename = 'N10_data_p1s500.npz'
filename = '10x2_data_p1_s1pts_periodic.npz'
#filename = 'asep_psweep_N10_data_p1s100.npz'

npzfile = np.load(filename)
print(npzfile.files)

s = npzfile['s']
p = npzfile['p']
CGF = npzfile['CGF']
EE = npzfile['EE']
nPart = npzfile['nPart']
density = npzfile['density']
current = npzfile['current']

# Convert to real
CGF = np.real(CGF)
EE = np.real(EE)
nPart = np.real(nPart)
density = np.real(density)
current = np.real(current)

# Convert to 1D arrays
CGF = CGF[0,:]
EE = EE[0,:]
nPart = nPart[0,:]
density = density[0,:,:]
current = current[0,:]

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
plt.style.use('fivethirtyeight') #'fivethirtyeight'
colormap = cm.plasma #coolwarm, inferno, viridis

# CGF Graph
if True:
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.plot(s,CGF)
    plt.show()

# Current Plot
if True:
    Current = (CGF[1:]-CGF[0:-1])/(s[0]-s[1])
    s_plt = s[0:-1]
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.plot(s_plt,Current)
    ax.plot(s,current)
    plt.show()

# Susceptibility Plot
if True:
    Current = (CGF[1:]-CGF[0:-1])/(s[0]-s[1])
    s_plt = s[0:-1]
    Susceptibility = (Current[1:]-Current[0:-1])/(s_plt[0]-s_plt[1])
    s_plt = s_plt[0:-1]
    fig = plt.figure()
    ax = fig.gca()
    start = 20
    end = 80
    surf = ax.plot(s_plt[start:end],Susceptibility[start:end])
    plt.show()

# Particle Count Plot
if True:
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.plot(s,nPart)
    plt.show()

# Density Profiles Plot
if True:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(len(s)):
        if not i%3:
            ax.plot(np.arange(len(density[i,:])),s[i]*np.ones(len(density[i,:])),density[i,:],
                    'k-o',linewidth=1)
    #ax.set_xlabel('Site')
    #ax.set_ylabel('$p$')
    #ax.set_zlabel('$\\rho$')
    ax.set_zlim(0,1)
    plt.show()

# ee plot
if True:
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.plot(s,EE)
    #ax.set_xlabel('$\lambda$')
    #ax.set_ylabel('$p$')
    #ax.set_zlabel('$s_{entanglement}$', rotation=0)
    plt.show()
