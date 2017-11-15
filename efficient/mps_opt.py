import numpy as np
import matplotlib.pyplot as plt
import time
import mpo
import warnings
from scipy.linalg import eig as fullEig
from scipy.sparse.linalg import eigs as arnoldiEig
from mpl_toolkits.mplot3d import axes3d
from numpy import ma

class MPS_OPT:

    def __init__(self, N=10, d=2, maxBondDim=8, tol=1e-5, maxIter=10,\
                 hamType='tasep', hamParams=(0.35,-1,2/3),\
                 plotExpVals=False, plotConv=False,\
                 eigMethod='full'):
        # Import parameters
        self.N = N
        self.d = d
        self.maxBondDim = maxBondDim
        self.tol = tol
        self.maxIter = maxIter
        self.hamType = hamType
        self.hamParams = hamParams
        self.plotExpVals = plotExpVals
        self.plotConv = plotConv
        self.eigMethod = eigMethod
        self.conv_figure = False
        self.exp_val_figure = False
     
        self.calc_spin_x = [0]*self.N
        self.calc_spin_y = [0]*self.N 
        self.calc_spin_z = [0]*self.N
        self.calc_empty = [0]*self.N
        self.calc_occ = [0]*self.N

    def generate_mps(self):
        self.M = []
        for i in range(int(self.N/2)):
            self.M.insert(len(self.M),np.ones((self.d,min(self.d**(i),self.maxBondDim),min(self.d**(i+1),self.maxBondDim))))
        for i in range(int(self.N/2))[::-1]:
            self.M.insert(len(self.M),np.ones((self.d,min(self.d**(i+1),self.maxBondDim),min(self.d**i,self.maxBondDim))))

    def generate_mpo(self):
        self.mpo = mpo.MPO(self.hamType,self.hamParams,self.N)

    def right_canonicalize_mps(self):
        for i in range(1,len(self.M))[::-1]:
            self.normalize(i,'left')

    def generate_f(self):
        self.F = []
        self.F.insert(len(self.F),np.array([[[1]]]))
        for i in range(int(self.N/2)):
            self.F.insert(len(self.F),np.zeros((min(self.d**(i+1),self.maxBondDim),4,min(self.d**(i+1),self.maxBondDim))))
        for i in range(int(self.N/2)-1,0,-1):
            self.F.insert(len(self.F),np.zeros((min(self.d**(i),self.maxBondDim),4,min(self.d**i,self.maxBondDim))))
        self.F.insert(len(self.F),np.array([[[1]]]))

    def normalize(self,i,direction):
        if direction is 'right':
            (n1,n2,n3) = self.M[i].shape
            M_reshape = np.reshape(self.M[i],(n1*n2,n3))
            (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
            self.M[i] = np.reshape(U,(n1,n2,n3))
            self.M[i+1] = np.einsum('i,ij,kjl->kil',s,V,self.M[i+1])
        elif direction is 'left':
            M_reshape = np.swapaxes(self.M[i],0,1)
            (n1,n2,n3) = M_reshape.shape
            M_reshape = np.reshape(M_reshape,(n1,n2*n3))
            (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
            M_reshape = np.reshape(V,(n1,n2,n3))
            self.M[i] = np.swapaxes(M_reshape,0,1)
            self.M[i-1] = np.einsum('klj,ji,i->kli',self.M[i-1],U,s)
        else:
            raise NameError('Direction must be left or right')

    def calc_initial_f(self):
        for i in range(int(self.N)-1,0,-1):
            print(i)
            self.F[i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(self.M[i]),self.mpo.W[i],self.M[i],self.F[i+1])

    def local_optimization(self,i):
        H = np.einsum('jlp,lmin,kmq->ijknpq',self.F[i],self.mpo.W[i],self.F[i+1])
        (n1,n2,n3,n4,n5,n6) = H.shape
        H = np.reshape(H,(n1*n2*n3,n4*n5*n6))
        #u,v = arnoldiEig(H,1,which='LR')
        u,v = np.linalg.eig(H)
        if (self.hamType is "tasep") or (self.hamType is "sep"):
            ind = np.argsort(u)[-1]
        else:
            ind = np.argsort(u)[0]
        E = u[ind]
        v = v[:,ind]
        print('\tEnergy at site {}= {}'.format(i,E))
        self.M[i] = np.reshape(v,(n1,n2,n3))
        return E

    def update_f(self,i,direction):
        if direction is 'right':
            self.F[i+1] = np.einsum('jlp,ijk,lmin,npq->kmq',self.F[i],np.conj(self.M[i]),self.mpo.W[i],self.M[i])
        elif direction is 'left':
            self.F[i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(self.M[i]),self.mpo.W[i],self.M[i],self.F[i+1])
        else:
            raise NameError('Direction must be left or right')

    def calc_observables(self,site):
        self.energy_calc = np.einsum('ijk,jlmn,olp,mio,nkp->',\
                self.F[site],self.mpo.W[site],self.F[site+1],np.conjugate(self.M[site]),self.M[site])
        if (self.hamType is "heis") or (self.hamType is "heis_2d"):
            self.calc_spin_x[site] = np.einsum('ijk,il,ljk->',np.conj(self.M[site]),self.mpo.Sx,self.M[site])
            self.calc_spin_y[site] = np.einsum('ijk,il,ljk->',np.conj(self.M[site]),self.mpo.Sy,self.M[site])
            self.calc_spin_z[site] = np.einsum('ijk,il,ljk->',np.conj(self.M[site]),self.mpo.Sz,self.M[site])
        elif (self.hamType is "tasep") or (self.hamType is "sep"):
            self.calc_empty[site] = np.einsum('ijk,il,ljk->',np.conj(self.M[site]),self.mpo.v,self.M[site])
            self.calc_occ[site] = np.einsum('ijk,il,ljk->',np.conj(self.M[site]),self.mpo.n,self.M[site])
        return self.energy_calc

    def plot_observables(self):
        if self.plotExpVals:
            plt.ion()
            if not self.exp_val_figure:
                self.exp_val_figure = plt.figure()
                self.angle = 0
            else:
                plt.figure(self.exp_val_figure.number)
            plt.cla()
            if (self.hamType is "tasep") or (self.hamType is "sep"):
                plt.plot(range(0,int(self.N-1)),self.calc_occ[0:int(self.N-1)],linewidth=3)
                plt.ylabel('Average Occupation',fontsize=20)
                plt.xlabel('Site',fontsize=20)
            elif self.hamType is "heis":
                ax = self.exp_val_figure.gca(projection='3d')
                x = np.arange(self.N)
                y = np.zeros(self.N)
                z = np.zeros(self.N)
                ax.scatter(x,y,z,color='k')
                plt.quiver(x,y,z,self.calc_spin_x,self.calc_spin_y,self.calc_spin_z,pivot='tail')
                ax.set_zlim((min(self.calc_spin_z),max(self.calc_spin_z)))
                ax.set_ylim((min(self.calc_spin_y),max(self.calc_spin_y)))
                plt.ylabel('y',fontsize=20)
                plt.xlabel('x',fontsize=20)
                ax.set_zlabel('z',fontsize=20)    
                self.angle += 3
                ax.view_init(30, self.angle)
                plt.draw()
            elif self.hamType is "heis_2d":
                ax = self.exp_val_figure.gca(projection='3d')
                x, y = np.meshgrid(np.arange((-self.mpo.N2d+1)/2,(self.mpo.N2d-1)/2+1),
                                   np.arange((-self.mpo.N2d+1)/2,(self.mpo.N2d-1)/2+1))
                ax.scatter(x,y,np.zeros((self.mpo.N2d,self.mpo.N2d)),color='k')
                plt.quiver(x,y,np.zeros((self.mpo.N2d,self.mpo.N2d)),
                           np.reshape(self.calc_spin_x,x.shape),
                           np.reshape(self.calc_spin_y,x.shape),
                           np.reshape(self.calc_spin_z,x.shape),
                           pivot='tail')
                ax.plot_surface(x, y, np.zeros((self.mpo.N2d,self.mpo.N2d)), alpha=0.2)
                ax.set_zlim((min(self.calc_spin_z),max(self.calc_spin_z)))
                plt.ylabel('y',fontsize=20)
                plt.xlabel('x',fontsize=20)
                ax.set_zlabel('z',fontsize=20)
                self.angle += 3
                ax.view_init(30, self.angle)
                plt.draw()
            else:
                raise ValueError("Plotting of expectation values is not implemented for the given hamiltonian type")
            plt.hold(False)
            plt.pause(0.0001)

    def plot_convergence(self,i):
        if self.plotConv:
            plt.ion()
            if not self.conv_figure:
                self.conv_figure = plt.figure()
                self.y_vec = [self.energy_calc]
                self.x_vec = [i]
            else:
                plt.figure(self.conv_figure.number)
                self.y_vec.insert(-1,self.energy_calc)
                self.x_vec.insert(-1,i)
            plt.cla()
            if len(self.y_vec) > 3:
                plt.plot(self.x_vec[:-2],self.y_vec[:-2],'r-',linewidth=2)
            plt.ylabel('Energy',fontsize=20)
            plt.xlabel('Site',fontsize=20)
            plt.hold(False)
            plt.pause(0.0001)

    def kernel(self):
        self.generate_mps()
        self.right_canonicalize_mps()
        self.generate_mpo()
        self.generate_f()
        self.calc_initial_f()
        converged = False
        iterCnt = 0
        E_prev = self.calc_observables(0)
        while not converged:
            # Right Sweep --------------------------
            print('Right Sweep {}'.format(iterCnt))
            for i in range(int(self.N-1)):
                E = self.local_optimization(i)
                self.calc_observables(i)
                self.normalize(i,'right')
                self.update_f(i,'right')
                self.plot_observables()
                self.plot_convergence(i)
            # Left Sweep ---------------------------
            print('Left Sweep {}'.format(iterCnt))
            for i in range(int(self.N-1),0,-1):
                E = self.local_optimization(i)
                self.calc_observables(i)
                self.normalize(i,'left')
                self.update_f(i,'left')
                self.plot_observables()
                self.plot_convergence(i)
            # Check Convergence --------------------
            if np.abs(E-E_prev) < self.tol:
                print('#'*75+'\nConverged at E = {}'.format(E)+'\n'+'#'*75)
                self.finalEnergy = E
                converged = True
            elif iterCnt >= self.maxIter:
                print('!'*75+'\nConvergence not acheived\n'+'!'*75)
                self.finalEnergy = E
                converged = True
            else:
                E_prev = E
                iterCnt += 1
        return self.finalEnergy