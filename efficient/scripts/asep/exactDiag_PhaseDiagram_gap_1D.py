import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sys import argv
from matplotlib import cm

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=10)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'
colormap = cm.plasma

N = 10
Np = 1
Ns = 10
rho_r = 0.5
rho_l = 0.5
pVec = np.linspace(0.,1.,Np)
pVec = np.array([0.1])
sVec = np.linspace(-1,0,Ns)

CGF_ed = np.zeros((len(pVec),len(sVec)),dtype=np.complex128)    
nPart_ed = np.zeros((len(pVec),len(sVec)),dtype=np.complex128)
density_ed = np.zeros((len(pVec),len(sVec),N),dtype=np.complex128)
eigenSpec_ed = np.zeros((len(pVec),len(sVec),2**N),dtype=np.complex128)

for i,p in enumerate(pVec):
    for j,s in enumerate(sVec):
        print('s,p = {},{}'.format(s,p))
        x = mps_opt.MPS_OPT(N=N,
                            hamType = "sep",
                            hamParams = (rho_l,1-rho_l,p,1-p,1-rho_r,rho_r,s))
        CGF_ed[i,j] = x.exact_diag()
        nPart_ed[i,j] = np.sum(x.ed.nv)
        density_ed[i,j,:] = x.ed.nv
        print(x.ed.eigSpec)
        print(x.ed.eigVecs.shape)
        # Renormalize
        _,n_renorm = x.ed.eigVecs.shape
        for i in range(n_renorm):
            x.ed.eigVecs[:,i] /= np.sum(x.ed.eigVecs[:,i])
        print(np.sum(np.abs(x.ed.eigVecs[:,-1]-x.ed.eigVecs[:,-2])))
        plt.plot(x.ed.eigVecs[:,-1]-x.ed.eigVecs[:,-2])
        if np.isclose(x.ed.eigSpec[-1],x.ed.eigSpec[-2]):
            for cnt in range(len(x.ed.eigVecs[:,-1])):
                print('{}\t{}\t{}'.format(np.abs(x.ed.eigVecs[cnt,-1]-x.ed.eigVecs[cnt,-2]),x.ed.eigVecs[cnt,-1],x.ed.eigVecs[cnt,-2]))
        plt.pause(0.1)
        eigenSpec_ed[i,j,:] = x.ed.eigSpec[::-1]
        fname = 'GapPhasDiagram_N'+str(N)+'_data_p'+str(Np)+'s'+str(Ns)
        np.savez(fname,
                 s=sVec,
                 p=pVec,
                 CGF_ed=CGF_ed,
                 nPart_ed=nPart_ed,
                 density_ed=density_ed,
                 eigenSpec_ed = eigenSpec_ed)



# Create plots
if True:
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    ax.plot(sVec,CGF_ed[0,:])
    plt.savefig('cgf_line.pdf')
if True:
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    curr = (CGF_ed[0,1:]-CGF_ed[0,0:-1])/(sVec[0]-sVec[1])
    ax.plot(sVec[0:-1],curr)
    plt.savefig('curr_line.pdf')
if True:
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    curr = (CGF_ed[0,1:]-CGF_ed[0,0:-1])/(sVec[0]-sVec[1])
    susc = (curr[1:]-curr[0:-1])/(sVec[0]-sVec[1])
    ax.plot(sVec[0:-1][0:-1],susc)
    plt.savefig('susc_line.pdf')
if True:
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    # Resize some of the data
    for i in range(2**N):
        ax.plot(sVec,eigenSpec_ed[0,:,i],'-o')
    plt.savefig('EigenSpec.pdf')
    


if False:
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca(projection='3d')
    ax.zaxis.set_rotate_label(False)
    sM,pM = np.meshgrid(sVec,pVec)
    surf = ax.plot_surface(sM,pM,np.real(CGF_ed),cmap=colormap,linewidth=0,antialiased=False)
    plt.savefig('cgf_surf.pdf')
    
if False:
    #CGFt = CGF_ed[::2,::2]
    #st = sVec[::2]
    #pt = pVec[::2]
    Current = (CGF_ed[:,1:]-CGF_ed[:,0:-1])/(sVec[0]-sVec[1])
    s_plt = sVec[0:-1]
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca(projection='3d')
    sM,pM = np.meshgrid(s_plt,pVec)
    surf = ax.plot_surface(sM,pM,np.real(Current),cmap=colormap,linewidth=0,antialiased=False)
    plt.savefig('current_surf.pdf')

if False:
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

if False:
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    sm,pm = np.meshgrid(sVec,pVec)
    surf = ax.pcolormesh(sm,pm,np.real(gap_ed),cmap=colormap)
    ax.set_ylim(0,1)
    ax.set_xlim(-5,5)
    fig.colorbar(surf)
    plt.savefig('gap_surf.pdf')
