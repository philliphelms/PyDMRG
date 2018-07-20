import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm # Colormaps

filename = 'results/asep_2d_PBC_bond_dim_check_Nx10_Ny2_data_p1s10.npz'
tol = 1e-8
end_bd = 19
filename = 'results/asep_2d_PBC_bond_dim_check_Nx10_Ny3_data_p1s10.npz'
tol = 1e-8
end_bd = 17
filename = 'results/asep_2d_PBC_bond_dim_check_Nx10_Ny4_data_p1s10.npz'
tol = 1e-8
end_bd = 12
filename = 'results/asep_2d_PBC_bond_dim_check_Nx20_Ny2_data_p1s10.npz'
tol = 1e-8
end_bd = 17
filename = 'results/asep_2d_PBC_bond_dim_check_Nx20_Ny3_data_p1s10.npz'
tol = 1e-8
end_bd = 11
filename = 'results/asep_2d_PBC_bond_dim_check_Nx20_Ny4_data_p1s10.npz'
tol = 1e-8
end_bd = 7
filename = 'results/asep_2d_PBC_bond_dim_check_Nx30_Ny2_data_p1s10.npz'
tol = 1e-8
end_bd = 13

bd = np.array([2,4,6,8,10,20,30,40,50,60,70,80,90,100,125,150,175,200,250,300,350,400,450,500,600,700,800,900,1000])
npzfile = np.load(filename)
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
plt.style.use('seaborn-bright') #'fivethirtyeight'
colormap = cm.plasma #coolwarm, inferno, viridis

# CGF Graph
if True:
    fig = plt.figure()
    ax = fig.gca()
    ns,nBD = CGF.shape
    for i in range(end_bd):
        ax.semilogy(s,np.abs((CGF[:,i]-CGF[:,end_bd])/CGF[:,end_bd])+tol*1e-1,'o-',label='$M='+str(bd[i])+'$',color=colormap(i/end_bd))
    ax.semilogy(np.array([s[0],s[-1]]),np.array([tol,tol]),'k:',label='Tolerance')
    ax.legend()

# 2nd CGF Graph (BD as abscissa)
if True:
    fig = plt.figure()
    ax = fig.gca()
    ns,nBD = CGF.shape
    for i in range(ns):
        ax.semilogy(bd[:end_bd],np.abs((CGF[i,:end_bd]-CGF[i,end_bd])/CGF[i,end_bd])+tol*1e-1,'o-',label='$s='+str(s[i])+'$',color=colormap(i/ns))
    ax.semilogy(np.array([bd[0],bd[end_bd-1]]),np.array([tol,tol]),'k:',label='Tolerance',lw=3)
    ax.legend(loc=1)

# Fraction of points below tolerance
if True:
    fig = plt.figure()
    ax = fig.gca()
    below_tol_cnt = np.zeros(bd.shape)
    for i in range(ns):
        diff = np.abs((CGF[i,:end_bd]-CGF[i,end_bd])/CGF[i,end_bd])
        for j in range(len(diff)):
            if diff[j] > tol:
                below_tol_cnt[j] += 1
    below_tol_cnt /= ns
    ax.plot(bd,below_tol_cnt)
    ax.legend(loc=1)



plt.show()
