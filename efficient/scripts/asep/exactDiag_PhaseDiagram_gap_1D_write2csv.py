import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sys import argv
from matplotlib import cm
import xlsxwriter


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

N = 6
Np = 1
Ns = 30
rho_r = 0.5
rho_l = 0.5
if Np == 1:
    pVec = np.array([0.1])
else:
    pVec = np.linspace(0.,1.,Np)
if Ns == 1:
    sVec = np.array([-1])
else:
    sVec = np.linspace(-1,2,Ns) # Center ~= -0.854477
    sVec = np.linspace(-0.8545,1,Ns)
    sVec = np.array([-3,-1.56944612666,-0.854477])

CGF_ed = np.zeros((len(pVec),len(sVec)),dtype=np.complex128)    
nPart_ed = np.zeros((len(pVec),len(sVec)),dtype=np.complex128)
density_ed = np.zeros((len(pVec),len(sVec),N),dtype=np.complex128)
eigenSpec_ed = np.zeros((len(pVec),len(sVec),2**N),dtype=np.complex128)
leigVec_ed = np.zeros((len(pVec),len(sVec),2**N),dtype=np.complex128)
reigVec_ed = np.zeros((len(pVec),len(sVec),2**N),dtype=np.complex128)


# Get worksheet started
workbook = xlsxwriter.Workbook('States6_productState.xlsx')
worksheet = workbook.add_worksheet('Ground State')
worksheet2= workbook.add_worksheet('First Excited State')
worksheet3= workbook.add_worksheet('Second Excited State')
for i in range(2**N):
    occ = list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:]))
    for j in range(len(occ)):
        worksheet.write(i+3,j+1,occ[j])
        worksheet2.write(i+3,j+1,occ[j])
        worksheet3.write(i+3,j+1,occ[j])

for i,p in enumerate(pVec):
    for j,s in enumerate(sVec):
        print('s,p = {},{}'.format(s,p))
        x = mps_opt.MPS_OPT(N=N,
                            hamType = "sep",
                            hamParams = (rho_l,1-rho_l,p,1-p,1-rho_r,rho_r,s))
        CGF_ed[i,j] = x.exact_diag()
        nPart_ed[i,j] = np.sum(x.ed.nv)
        density_ed[i,j,:] = x.ed.nv
        worksheet.write(0,0,'Energy')
        worksheet.write(0,1,np.real(x.ed.eigSpec[0]))
        worksheet.write(0,2,np.imag(x.ed.eigSpec[0]))
        worksheet.write(2,2*j+len(occ)+1,s)
        worksheet2.write(0,0,'Energy')
        worksheet2.write(0,1,np.real(x.ed.eigSpec[1]))
        worksheet2.write(0,2,np.imag(x.ed.eigSpec[1]))
        worksheet2.write(2,2*j+len(occ)+1,s)
        worksheet3.write(0,0,'Energy')
        worksheet3.write(0,1,np.real(x.ed.eigSpec[2]))
        worksheet3.write(0,2,np.imag(x.ed.eigSpec[2]))
        worksheet3.write(2,2*j+len(occ)+1,s)
        print('This is the one',np.sum(x.ed.eigVecs[:,0]))
        x.ed.eigVecs[:,0] /= np.sqrt(np.sum(x.ed.eigVecs[:,0]**2.))
        for k in range(2**N):
            worksheet.write(k+3,2*j+len(occ)+1,x.ed.eigVecs[k,0])
            worksheet.write(k+3,2*j+len(occ)+2,x.ed.eigVecsL[k,0]**2)
            worksheet2.write(k+3,2*j+len(occ)+1,x.ed.eigVecs[k,1])
            worksheet2.write(k+3,2*j+len(occ)+2,x.ed.eigVecsL[k,1]**2)
            worksheet3.write(k+3,2*j+len(occ)+1,x.ed.eigVecs[k,2])
            worksheet3.write(k+3,2*j+len(occ)+2,x.ed.eigVecsL[k,2]**2)
        eigenSpec_ed[i,j,:] = x.ed.eigSpec[::-1]
        fname = 'GapPhasDiagram_N'+str(N)+'_data_p'+str(Np)+'s'+str(Ns)
        np.savez(fname,
                 s=sVec,
                 p=pVec,
                 CGF_ed=CGF_ed,
                 nPart_ed=nPart_ed,
                 density_ed=density_ed,
                 eigenSpec_ed = eigenSpec_ed)

workbook.close()

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
if False:
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    # Resize some of the data
    for i in range(2**N):
        ax.plot(sVec,eigenSpec_ed[0,:,i],'-o')
    plt.savefig('EigenSpec.pdf')
if True:
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    ax.semilogy(sVec,eigenSpec_ed[0,:,0]-eigenSpec_ed[0,:,1])
    plt.savefig('gap_line.pdf')
    plt.show()

    


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
