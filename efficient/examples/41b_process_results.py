import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm # Colormaps

filename = '10_10_data.npz'

npzfile = np.load(filename)
s = npzfile['s']
p = npzfile['p']
CGF = npzfile['CGF']
EE = npzfile['EE']
nPart = npzfile['nPart']
CGF_ed = npzfile['CGF_ed']
nPart_ed = npzfile['nPart_ed']
density = npzfile['density']
density_ed = npzfile['density_ed']

# Convert to real
s = np.real(s)
p = np.real(p)
CGF = np.real(CGF)
EE = np.real(EE)
nPart = np.real(nPart)
CGF_ed = np.real(CGF_ed)
nPart_ed = np.real(nPart_ed)
density = np.real(density)
density_ed = np.real(density_ed)

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
plt.style.use('fivethirtyeight') #'fivethirtyeight'

# CGF Plot
if True:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.zaxis.set_rotate_label(False)
    sM,pM = np.meshgrid(s,p)
    surf = ax.plot_surface(sM,pM,CGF,cmap=cm.coolwarm,linewidth=0,antialiased=False)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$p$')
    ax.set_zlabel('$\psi$', rotation=0)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()

# Current Plot
if True:
    Current = (CGF[:,1:]-CGF[:,0:-1])/(s[0]-s[1])
    s_plt = s[0:-1]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.zaxis.set_rotate_label(False)
    sM,pM = np.meshgrid(s_plt,p)
    surf = ax.plot_surface(sM,pM,Current,cmap=cm.coolwarm,linewidth=0,antialiased=False)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$p$')
    ax.set_zlabel('$\partial\lambda(\psi)=J$')#, rotation=0)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()

# Particle Count Plot
if False:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sM,pM = np.meshgrid(s,p)
    surf = ax.plot_surface(sM,pM,nPart_ed,cmap=cm.coolwarm,linewidth=0,antialiased=False)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$p$')
    ax.set_zlabel('Particles')#, rotation=0)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()

# Density Profiles Plot
if True:
    # Trajectory 1
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    p_ind = 1
    for i in range(len(s)):
        ax.plot(np.arange(len(density_ed[p_ind,i,:])),s[i]*np.ones(len(density_ed[p_ind,i,:])),density_ed[p_ind,i,:],
                'k-o',linewidth=1)
    ax.set_xlabel('Site')
    ax.set_ylabel('$\lambda$')
    ax.set_zlabel('$\\rho$')
    plt.show()
    # Trajectory 2
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    p_ind = 5
    for i in range(len(s)):
        ax.plot(np.arange(len(density_ed[p_ind,i,:])),s[i]*np.ones(len(density_ed[p_ind,i,:])),density_ed[p_ind,i,:],
                'k-o',linewidth=1)
    ax.set_xlabel('Site')
    ax.set_ylabel('$\lambda$')
    ax.set_zlabel('$\\rho$')
    plt.show()
    # Trajectory 2
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    s_ind = 9
    for i in range(len(p)):
        ax.plot(np.arange(len(density_ed[i,s_ind,:])),p[i]*np.ones(len(density_ed[i,s_ind,:])),density_ed[i,s_ind,:],
                'k-o',linewidth=1)
    ax.set_xlabel('Site')
    ax.set_ylabel('$p$')
    ax.set_zlabel('$\\rho$')
    plt.show()

# EE Plot
if True:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.zaxis.set_rotate_label(False)
    sM,pM = np.meshgrid(s,p)
    surf = ax.plot_surface(sM,pM,EE,cmap=cm.coolwarm,linewidth=0,antialiased=False)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$p$')
    ax.set_zlabel('$S_{Entanglement}$', rotation=0)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()
