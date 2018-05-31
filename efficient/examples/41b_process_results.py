import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm # Colormaps

#filename = 'asep_10x10_data_spacing50.npz'
#filename = 'asep_10x10_data_spacing200.npz'
filename = 'asep_10x10_data_spacing100.npz'
#filename = '10_10_data_p0.npz'
#filename = '10_10_data_p0_forJ.npz'

npzfile = np.load(filename)
#s = npzfile['s']
#p = npzfile['p']
print(npzfile.files)
CGF = npzfile['CGF']
EE = npzfile['EE']
#nPart = npzfile['nPart']
CGF_ed = npzfile['CGF_ed']
nPart_ed = npzfile['nPart_ed']
density = npzfile['density']
density_ed = npzfile['density_ed']
s = np.linspace(-5,5,len(CGF))
p = np.linspace(0,1,len(CGF))

# Convert to real
s = np.real(s)
p = np.real(p)
CGF = np.real(CGF)
EE = np.real(EE)
#nPart = np.real(nPart)
CGF_ed = np.real(CGF_ed)
nPart_ed = np.real(nPart_ed)
density = np.real(density)
density_ed = np.real(density_ed)

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
plt.style.use('fivethirtyeight') #'fivethirtyeight'
colormap = cm.plasma #coolwarm, inferno, viridis

# CGF Plot
if False:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.zaxis.set_rotate_label(False)
    sM,pM = np.meshgrid(s,p)
    surf = ax.plot_surface(sM,pM,CGF,cmap=colormap,linewidth=0,antialiased=False)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$p$')
    ax.set_zlabel('$\psi$', rotation=0)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()

# CGF Image
if False:
    fig = plt.figure()
    ax = fig.gca()
    sM,pM = np.meshgrid(s,p)
    surf = plt.pcolormesh(s,p,CGF,cmap=colormap,linewidth=0)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$p$')
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()

# Current Plot
if True:
    Current = (CGF_ed[:,1:]-CGF_ed[:,0:-1])/(s[0]-s[1])
    s_plt = s[0:-1]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.zaxis.set_rotate_label(False)
    sM,pM = np.meshgrid(s_plt,p)
    surf = ax.plot_surface(sM,pM,Current,cmap=colormap,linewidth=0,antialiased=False)
    #ax.set_xlabel('$\lambda$')
    #ax.set_ylabel('$p$')
    #ax.set_zlabel('$\partial\lambda(\psi)=J$')#, rotation=0)
    #fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()

# Current Plot (1d)
if False:
    Current = (CGF_ed[:,1:]-CGF_ed[:,0:-1])/(s[0]-s[1])
    s_plt = s[0:-1]
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.plot(p,Current[:,0])
    ax.set_xlabel('$p$')
    ax.set_ylabel('$\partial\lambda(\psi)=J$')#, rotation=0)
    plt.show()

# Current Image
if False:
    Current = (CGF_ed[:,1:]-CGF_ed[:,0:-1])/(s[0]-s[1])
    s_plt = s[0:-1]
    fig = plt.figure()
    ax = fig.gca()
    sM,pM = np.meshgrid(s_plt,p)
    surf = ax.pcolormesh(sM,pM,Current,cmap=colormap)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$p$')
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()

# Susceptibility Plot
if False:
    Current = (CGF_ed[:,1:]-CGF_ed[:,0:-1])/(s[0]-s[1])
    s_plt = s[0:-1]
    Susceptibility = (Current[:,1:]-Current[:,0:-1])/(s_plt[0]-s_plt[1])
    s_plt = s_plt[0:-1]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.zaxis.set_rotate_label(False)
    sM,pM = np.meshgrid(s_plt,p)
    start = 20
    end = 80
    surf = ax.plot_surface(sM[10:90,start:end],pM[10:90,start:end],Susceptibility[10:90,start:end],cmap=colormap,linewidth=0,antialiased=False)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$p$')
    ax.set_zlabel('$\partial^2\lambda(\psi)=\chi$')#, rotation=0)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()

# Susceptibility Image
if True:
    Current = (CGF_ed[:,1:]-CGF_ed[:,0:-1])/(s[0]-s[1])
    s_plt = s[0:-1]
    Susceptibility = (Current[:,1:]-Current[:,0:-1])/(s_plt[0]-s_plt[1])
    s_plt = s_plt[0:-1]
    fig = plt.figure()
    ax = fig.gca()
    sM,pM = np.meshgrid(s_plt,p)
    start = 20
    end = 80
    surf = ax.pcolormesh(sM[10:90,start:end],pM[10:90,start:end],Susceptibility[10:90,start:end],cmap=colormap,linewidth=0,antialiased=False)
    #ax.set_xlabel('$\lambda$')
    #ax.set_ylabel('$p$')
    #fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()


# Particle Count Plot
if False:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sM,pM = np.meshgrid(s,p)
    surf = ax.plot_surface(sM,pM,nPart_ed,cmap=colormap,linewidth=0,antialiased=False)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$p$')
    ax.set_zlabel('Particles')#, rotation=0)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()

# Density Profiles Plot
if False:
    len(p)
    print(density_ed.shape)
    for p_ind in range(len(p)):
        print('p = {}'.format(p_ind))
        # Trajectory 1
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #p_ind = 1
        for i in range(len(s)):
            if not i%4:
                ax.plot(np.arange(len(density_ed[p_ind,i,:])),s[i]*np.ones(len(density_ed[p_ind,i,:])),density_ed[p_ind,i,:],
                        'k-o',linewidth=1)
        #ax.set_xlabel('Site')
        #ax.set_ylabel('$\lambda$')
        #ax.set_zlabel('$\\rho$')
        ax.set_zlim(0,1)
        plt.show()
    for s_ind in range(len(s)):
        print('s = {}'.format(s_ind))
        # Trajectory 2
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #s_ind = 9
        for i in range(len(p)):
            if not i%5:
                ax.plot(np.arange(len(density_ed[i,s_ind,:])),p[i]*np.ones(len(density_ed[i,s_ind,:])),density_ed[i,s_ind,:],
                        'k-o',linewidth=1)
        #ax.set_xlabel('Site')
        #ax.set_ylabel('$p$')
        #ax.set_zlabel('$\\rho$')
        ax.set_zlim(0,1)
        plt.show()

# ee plot
if False:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.zaxis.set_rotate_label(False)
    sm,pm = np.meshgrid(s,p)
    surf = ax.plot_surface(sm,pm,EE,cmap=colormap,linewidth=0,antialiased=False)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$p$')
    ax.set_zlabel('$s_{entanglement}$', rotation=0)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()

# ee plot
if False:
    fig = plt.figure()
    ax = fig.gca()
    sm,pm = np.meshgrid(s,p)
    surf = ax.pcolormesh(sm,pm,EE,cmap=colormap)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$p$')
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()

# MatPlotLib Movie
if False:
    import os
    import subprocess
    files = []
    print(len(p))
    for p_ind in range(5,len(p)-5):
        fig = plt.figure(figsize=(10,5))
        # EE Image
        ax1 = fig.add_subplot(121)
        sm,pm = np.meshgrid(s,p)
        surf = ax1.pcolormesh(sm,pm,EE,cmap=colormap)
        ax1.plot(np.array([-5,5]),np.array([p[p_ind],p[p_ind]]),'r-',linewidth=1)
        ax1.set_xlabel('$\lambda$')
        ax1.set_ylabel('$p$')
        #fig.colorbar(surf,shrink=0.5,aspect=5)
        ax2 = fig.add_subplot(122,projection='3d')
        for i in range(len(s)):
            if not i%3:
                ax2.plot(np.arange(len(density[p_ind,i,:])),s[i]*np.ones(len(density[p_ind,i,:])),density[p_ind,i,:],
                        'k-o',linewidth=1)
        #ax2.set_xlabel('Site')
        #ax2.set_ylabel('$\lambda$')
        ax2.set_zlim(0,1)
        #ax2.set_zlabel('$\\rho$')
        fname = '_tmp%03d.png' % p_ind
        plt.savefig(fname)
        files.append(fname)

    subprocess.call("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc "
                     "-lavcopts vcodec=wmv2 -oac copy -o p_looped_animation.mpg", shell=True)
    for fname in files:
        os.remove(fname)

# MatPlotLib Movie
if True:
    import os
    import subprocess
    files = []
    for s_ind in range(len(s)):
        fig = plt.figure(figsize=(10,5))
        # EE Image
        ax1 = fig.add_subplot(121)
        sm,pm = np.meshgrid(s,p)
        surf = ax1.pcolormesh(sm,pm,EE,cmap=colormap)
        ax1.plot(np.array([s[s_ind],s[s_ind]]),np.array([0,1]),'r-',linewidth=1)
        #ax1.set_xlabel('$\lambda$')
        #ax1.set_ylabel('$p$')
        #fig.colorbar(surf,shrink=0.5,aspect=5)
        ax2 = fig.add_subplot(122,projection='3d')
        for i in range(2,len(p)-2):
            if not i%3:
                ax2.plot(np.arange(len(density[i,s_ind,:])),s[i]*np.ones(len(density[i,s_ind,:])),density[i,s_ind,:],
                        'k-o',linewidth=1)
        #ax2.set_xlabel('Site')
        #ax2.set_ylabel('$\lambda$')
        ax2.set_zlim(0.3,0.7)
        #ax2.set_zlabel('$\\rho$')
        fname = '_tmp%03d.png' % s_ind
        plt.savefig(fname)
        files.append(fname)

    subprocess.call("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc "
                     "-lavcopts vcodec=wmv2 -oac copy -o s_looped_animation.mpg", shell=True)
    for fname in files:
        os.remove(fname)
