import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm # Colormaps

filename = 'results/1534482104_30x2_data_p1_s1pts_periodic.npz'
filename = 'results/1534564705_10x1_data_p1_s1pts_periodic.npz'

Nx = 10
Ny = 2

npzfile = np.load(filename)
print(npzfile.files)

sx = npzfile['sx']
sy = npzfile['sy']
px = npzfile['px']
py = npzfile['py']
CGF = npzfile['CGF']
EE = npzfile['EE']
nPart = npzfile['nPart']
density = npzfile['density']

# Convert to real
CGF = np.real(CGF)
EE = np.real(EE)
nPart = np.real(nPart)
density = np.real(density)

# Convert to 1D arrays
CGF = CGF[0,:,0,0]
EE = EE[0,:,0,0]
nPart = nPart[0,:,0,0]
print(density.shape)
density = density[0,:,0,0,:]

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
    surf = ax.plot(sx,CGF)

# Current Plot
if True:
    Current = (CGF[1:]-CGF[0:-1])/(sx[0]-sx[1])
    s_plt = sx[0:-1]
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.plot(s_plt,Current)

# Susceptibility Plot
if True:
    Current = (CGF[1:]-CGF[0:-1])/(sx[0]-sx[1])
    s_plt = sx[0:-1]
    Susceptibility = (Current[1:]-Current[0:-1])/(s_plt[0]-s_plt[1])
    s_plt = s_plt[0:-1]
    fig = plt.figure()
    ax = fig.gca()
    start = 20
    end = 80
    surf = ax.plot(s_plt[start:end],Susceptibility[start:end])

# Particle Count Plot
if False:
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.plot(sx,nPart)

# Density Profiles Plot
if True:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(len(sx)):
        if not i%1:
            ax.plot(np.arange(len(density[i,:])),sx[i]*np.ones(len(density[i,:])),density[i,:],'k-o',linewidth=1)
    #ax.set_xlabel('Site')
    #ax.set_ylabel('$p$')
    #ax.set_zlabel('$\\rho$')
    ax.set_zlim(0,1)

if True:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x,y = np.meshgrid(np.arange(Nx),sx)
    ax.plot_surface(x,y,density[:,::Ny],cmap=colormap)
    ax.set_zlim(0,1)

# ee plot
if True:
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.plot(sx,EE)
    #ax.set_xlabel('$\lambda$')
    #ax.set_ylabel('$p$')
    #ax.set_zlabel('$s_{entanglement}$', rotation=0)

plt.show()
