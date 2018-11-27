import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sys import argv

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=10)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'
colormap = cm.plasma


filename = argv[1]
npzfile = np.load(filename)
print(npzfile.files)
sVec = npzfile['s']
pVec = npzfile['p']
CGF_ed = npzfile['CGF_ed']
density_ed = npzfile['density_ed']
eigenSpec_ed = npzfile['eigenSpec_ed']
rightEE_ed = npzfile['rightEE_ed']
leftEE_ed = npzfile['leftEE_ed']
print(eigenSpec_ed.shape)
gap_ed = eigenSpec_ed[:,:,0]-eigenSpec_ed[:,:,1]
print(gap_ed)

if len(pVec) == 1:
    if True:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(sVec,CGF_ed[0,:],'.')
        plt.savefig('cgf_line.pdf')

    if True:
        fig = plt.figure()
        ax = fig.gca()
        current = (CGF_ed[:,1:]-CGF_ed[:,0:-1])/(sVec[0]-sVec[1])
        s_plt = sVec[0:-1]
        plt.plot(s_plt,current[0,:],'.')
        plt.savefig('current_line.pdf')

    if True:
        Current = (CGF_ed[:,1:]-CGF_ed[:,0:-1])/(sVec[0]-sVec[1])
        s_plt = sVec[0:-1]
        Susceptibility = (Current[:,1:]-Current[:,0:-1])/(s_plt[0]-s_plt[1])
        s_plt = s_plt[0:-1]
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(s_plt,Susceptibility[0,:],'.')
        plt.savefig('susc_line.pdf')

    if True:
        plt.figure()
        gap = eigenSpec_ed[0,:,0]-eigenSpec_ed[0,:,1]
        plt.semilogy(sVec,gap,'.')
        plt.savefig('gap_line.pdf')
        plt.figure()
        gap = eigenSpec_ed[0,:,0]-eigenSpec_ed[0,:,1]
        gap_der = (gap[1:]-gap[0:-1])/(sVec[0]-sVec[1])
        s_plt = sVec[0:-1]
        plt.plot(s_plt,gap_der,'.')
if len(sVec) == 1:
    print('Not implemented for p sweep yet')
else:
    # CGF Surface Plot
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca(projection='3d')
    ax.zaxis.set_rotate_label(False)
    sM,pM = np.meshgrid(sVec,pVec)
    surf = ax.plot_surface(sM,pM,np.real(CGF_ed),cmap=colormap,linewidth=0,antialiased=False)
    plt.savefig('cgf_surf.pdf')

    # Current Surface Plot
    Current = (CGF_ed[:,1:]-CGF_ed[:,0:-1])/(sVec[0]-sVec[1])
    s_plt = sVec[0:-1]
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca(projection='3d')
    sM,pM = np.meshgrid(s_plt,pVec)
    surf = ax.plot_surface(sM,pM,np.real(Current),cmap=colormap,linewidth=0,antialiased=False)
    plt.savefig('current_surf.pdf')

    # Susceptibility Surface Plot
    Current = (CGF_ed[:,1:]-CGF_ed[:,0:-1])/(sVec[0]-sVec[1])
    s_plt = sVec[0:-1]
    Susceptibility = (Current[:,1:]-Current[:,0:-1])/(s_plt[0]-s_plt[1])
    s_plt = s_plt[0:-1]
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    sM,pM = np.meshgrid(s_plt,pVec)
    surf = ax.pcolormesh(sM,pM,np.real(Susceptibility),cmap=colormap,linewidth=0,antialiased=False,vmin=0, vmax=10)
    ax.set_ylim(0,1)
    ax.set_xlim(-5,5)
    fig.colorbar(surf)
    plt.savefig('susc_surf.pdf')
    
    # Gap Surface Plot
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    sm,pm = np.meshgrid(sVec,pVec)
    surf = ax.pcolormesh(sm,pm,np.log(np.real(gap_ed)),cmap=colormap,vmin=-5,vmax=1)
    ax.set_ylim(0,1)
    ax.set_xlim(-5,5)
    fig.colorbar(surf)
    plt.savefig('gap_surf.pdf')

    # Right Entanglement Entropy Surface Plot
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    sm,pm = np.meshgrid(sVec,pVec)
    surf = ax.pcolormesh(sm,pm,np.real(rightEE_ed),cmap=colormap,vmin=0,vmax=1)
    ax.set_ylim(0,1)
    ax.set_xlim(-5,5)
    fig.colorbar(surf)
    plt.savefig('eer_surf.pdf')

    # Left Entanglement Entropy Surface Plot
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    sm,pm = np.meshgrid(sVec,pVec)
    surf = ax.pcolormesh(sm,pm,np.real(leftEE_ed),cmap=colormap,vmin=0,vmax=1)
    ax.set_ylim(0,1)
    ax.set_xlim(-5,5)
    fig.colorbar(surf)
    plt.savefig('eel_surf.pdf')


plt.show()
