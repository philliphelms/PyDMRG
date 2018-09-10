from mpi4py import MPI
import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sys import argv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('On rank {}'.format(rank))

#########################################
# Calculation Parameters:
N = int(argv[1])
nproc = int(argv[2])
rho_r = 0.5
rho_l = 0.5
s = np.linspace(-2,-0.5,nproc-1)
p = np.array([0.2])
bd = [2,4,6,8,10,20,30,40,50,60,70,80,90,100,125,150,175,200,250,300,350,400,450,500,600,700,800,900,1000]
maxIter = [5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2]
#########################################
fname = str(int(time.time()))+'_'+'asep_bond_dim_check_N'+str(N)+'_data_p'+str(len(p))+'s'+str(len(s))+'bd'+str(len(bd))

if rank == 0:
    CGF = np.zeros((len(p),len(s),len(bd)),dtype=np.complex128)
    EE = np.zeros((len(p),len(s),len(bd)),dtype=np.complex128)
    np.savez(fname,s=s,p=p,bd=np.array(bd),maxIter=np.array(maxIter),CGF=CGF,EE=EE)
    for i in range(1, size):
        sendMsg = 'Save rank {}'.format(i)
        comm.send(sendMsg, dest=i)
        recMsg = comm.recv(source=i)
        print(recMsg)
else:
    i = 0
    j = rank - 1
    print('s = {}'.format(s[j]))
    print('p = {}'.format(p[i]))
    x = mps_opt.MPS_OPT(N=N,
                        maxBondDim = bd,
                        maxIter = maxIter,
                        hamType = "sep",
                        outputFile = '1D_bondDimConvergence_s'+str(j)+'.log',
                        hamParams = (rho_l,1-rho_l,p[i],1-p[i],1-rho_r,rho_r,s[j]))
    x.kernel()
    recvMsg = comm.recv(source=0)
    print(recvMsg)
    print("Loading {}".format(fname))
    npzfile = np.load(fname+'.npz')
    print('Loaded {}'.format(fname))
    CGF = npzfile['CGF']
    EE = npzfile['EE']
    CGF[i,j,:] = x.bondDimEnergies
    EE[i,j,:] = x.bondDimEntanglement
    print('Saving {}'.format(fname))
    np.savez(fname,s=s,p=p,bd=np.array(bd),maxIter=np.array(maxIter),CGF=CGF,EE=EE)
    print('Saved {}'.format(fname))
    comm.send('Completed saving rank {}'.format(rank),dest=0)
