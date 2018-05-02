import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import mpo
import warnings
from mpl_toolkits.mplot3d import axes3d
import scipy as sp
import pickle

class MPS_OPT:

    def __init__(self, N=10, d=2, maxBondDim=100, tol=1e-5, maxIter=5,\
                 hamType='tasep', hamParams=(0.35,-1,2/3),\
                 plotExpVals=False, plotConv=False,\
                 initialGuess=0.001,ed_limit=12,max_eig_iter=50,\
                 periodic_x=False,periodic_y=False,add_noise=False,\
                 saveResults=True,dataFolder='data/',verbose=3,nroots=1):
        # Import parameters
        self.N = N
        self.N_mpo = N
        self.d = d
        self.maxBondDimInd = 0
        if isinstance(maxBondDim, list):
            self.maxBondDim = maxBondDim
        else:
            self.maxBondDim = [maxBondDim]
        self.maxBondDimCurr = self.maxBondDim[self.maxBondDimInd]
        if isinstance(tol,list):
            self.tol = tol
        else:
            self.tol = [tol]*len(self.maxBondDim)
        if isinstance(maxIter,list):
            self.maxIter = maxIter
        else:
            self.maxIter = [maxIter]*len(self.maxBondDim)
        assert(len(self.maxIter) is len(self.maxBondDim))
        self.hamType = hamType
        self.hamParams = hamParams
        self.plotExpVals = plotExpVals
        self.plotConv = plotConv
        self.saveResults = saveResults
        self.dataFolder = dataFolder
        self.verbose = verbose
        self.nroots = nroots
        self.curr_root = 0
        self.initialGuess = initialGuess
        self.ed_limit = ed_limit
        self.max_eig_iter = max_eig_iter
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.add_noise = add_noise

    def initialize_containers(self):
        if type(self.N) is not int:
            self.N = self.N[0]*self.N[1]
        self.inside_iter_time = [np.zeros(len(self.maxBondDim))]*self.nroots
        self.inside_iter_cnt = [np.zeros(len(self.maxBondDim))]*self.nroots
        self.outside_iter_time = [np.zeros(len(self.maxBondDim))]*self.nroots
        self.outside_iter_cnt = [np.zeros(len(self.maxBondDim))]*self.nroots
        self.time_total = time.time()
        self.exp_val_figure= [False]*self.nroots
        self.conv_figure= [False]*self.nroots
        self.calc_spin_x = [[0]*self.N]*self.nroots
        self.calc_spin_y = [[0]*self.N]*self.nroots
        self.calc_spin_z = [[0]*self.N]*self.nroots
        self.calc_empty = [[0]*self.N]*self.nroots
        self.calc_occ = [[0]*self.N]*self.nroots
        self.bondDimEnergies = [np.zeros(len(self.maxBondDim))]*self.nroots
        import lib.linalg_helper
        import lib.numpy_helper
        self.einsum = lib.numpy_helper.einsum
        if (self.hamType is "heis") or (self.hamType is "heis_2d") or (self.hamType is 'ising'):
            self.eig = lib.linalg_helper.eigh
        else:
            self.eig = lib.linalg_helper.eig

    def generate_mps(self):
        if self.verbose > 4:
            print('\t'*2+'Generating MPS')
        M = []
        base = np.array([[-1/np.sqrt(2),-1/np.sqrt(2)],[-1/np.sqrt(2),1/np.sqrt(2)]])
        for i in range(int(self.N/2)):
            if self.initialGuess is "zeros":
                M.insert(len(M),np.zeros((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
            elif self.initialGuess is "ones":
                M.insert(len(M),np.ones((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
            elif self.initialGuess is "rand":
                M.insert(len(M),np.random.rand(self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))) 
            else:
                M.insert(len(M),self.initialGuess*np.ones((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
        if self.N%2 is 1: # Check if system size is odd
            if self.initialGuess is "zeros":
                M.insert(len(self.N),np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
            elif self.initialGuess is "ones":
                M.insert(len(M),np.ones((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
            elif self.initialGuess is "rand":
                M.insert(len(M),np.random.rand(self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr)))
            else:
                M.insert(len(M),self.initialGuess*np.ones((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
        for i in range(int(self.N/2))[::-1]:
            if self.initialGuess is "zeros":
                M.insert(len(M),np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr))))
            elif self.initialGuess is "ones":
                M.insert(len(M),np.ones((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr))))
            elif self.initialGuess is "rand":
                M.insert(len(M),np.random.rand(self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr)))
            else:
                M.insert(len(M),self.initialGuess*np.ones((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr))))
        self.M = []
        for i in range(self.nroots):
            if i is 0:
                self.M.insert(len(self.M),M)
            else:
                self.M.insert(len(self.M),copy.deepcopy(M))

    def generate_mpo(self):
        if self.verbose > 4:
            print('\t'*2+'Generating MPO')
        self.mpo = mpo.MPO(self.hamType,self.hamParams,self.N_mpo,periodic_x=self.periodic_x,periodic_y=self.periodic_y)

    def right_canonicalize_mps(self):
        if self.verbose > 4:
            print('\t'*2+'Performing Right Canonicalization')
        for i in range(1,len(self.M))[::-1]:
            self.normalize(i,'left')
            self.calc_observables(i)

    def generate_f(self):
        if self.verbose > 4:
            print('\t'*2+'Generating initial F arrays')
        F = []
        for i in range(self.mpo.nops):
            F_tmp = []
            F_tmp.insert(len(F_tmp),np.array([[[1]]]))
            for j in range(int(self.N/2)):
                F_tmp.insert(len(F_tmp),np.zeros((min(self.d**(j+1),self.maxBondDimCurr),2,min(self.d**(j+1),self.maxBondDimCurr))))
            if self.N%2 is 1:
                F_tmp.insert(len(F_tmp),np.zeros((min(self.d**(j+2),self.maxBondDimCurr),2,min(self.d**(j+2),self.maxBondDimCurr))))
            for j in range(int(self.N/2)-1,0,-1):
                F_tmp.insert(len(F_tmp),np.zeros((min(self.d**(j),self.maxBondDimCurr),2,min(self.d**j,self.maxBondDimCurr))))
            F_tmp.insert(len(F_tmp),np.array([[[1]]]))
            F.insert(len(F),F_tmp)
        self.F = []
        for i in range(self.nroots):
            if i is 0:
                self.F.insert(len(self.F),F)
            else:
                self.F.insert(len(self.F),copy.deepcopy(F))

    def normalize(self,i,direction):
        if self.verbose > 4:
            print('\t'*2+'Normalization at site {} in direction: {}'.format(i,direction))
        for root_ind in range(self.curr_root+1):
            if direction is 'right':
                (n1,n2,n3) = self.M[root_ind][i].shape
                M_reshape = np.reshape(self.M[root_ind][i],(n1*n2,n3))
                (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
                self.M[root_ind][i] = np.reshape(U,(n1,n2,n3))
                self.M[root_ind][i+1] = self.einsum('i,ij,kjl->kil',s,V,self.M[root_ind][i+1])
            elif direction is 'left':
                M_reshape = np.swapaxes(self.M[root_ind][i],0,1)
                (n1,n2,n3) = M_reshape.shape
                M_reshape = np.reshape(M_reshape,(n1,n2*n3))
                (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
                M_reshape = np.reshape(V,(n1,n2,n3))
                self.M[root_ind][i] = np.swapaxes(M_reshape,0,1)
                self.M[root_ind][i-1] = self.einsum('klj,ji,i->kli',self.M[root_ind][i-1],U,s)
            else:
                raise NameError('Direction must be left or right')

    def increaseBondDim(self):
        if self.verbose > 3:
            print('\t'*2+'Increasing Bond Dimensions from {} to {}'.format(self.maxBondDim[self.maxBondDimInd-1],self.maxBondDimCurr))
        Mnew = []
        for i in range(int(self.N/2)):
            Mnew.insert(len(Mnew),np.zeros((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
        if self.N%2 is 1:
            Mnew.insert(len(Mnew),np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
        for i in range(int(self.N/2))[::-1]:
            Mnew.insert(len(Mnew),np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr))))
        for i in range(len(Mnew)):
            nx,ny,nz = self.M[self.curr_root][i].shape
            Mnew[i][:nx,:ny,:nz] = self.M[self.curr_root][i]
            self.M[self.curr_root][i] = Mnew[i]

    def calc_initial_f(self):
        if self.verbose > 3:
            print('\t'*2+'Calculating all entries in F')
        self.generate_f()
        for i in range(self.mpo.nops):
            if self.verbose > 4:
                print('\t'*3+'For Operator {}/{}'.format(i,self.mpo.nops))
            for j in range(int(self.N)-1,0,-1):
                if self.verbose > 5:
                    print('\t'*3+'at site {}'.format(j))
                if self.mpo.ops[i][j] is None:
                    tmp_sum1 = self.einsum('cdf,eaf->acde',self.F[self.curr_root][i][j+1],self.M[self.curr_root][j])
                    self.F[self.curr_root][i][j] = self.einsum('bxc,acyb->xya',np.conj(self.M[self.curr_root][j]),tmp_sum1)
                else:
                    tmp_sum1 = self.einsum('cdf,eaf->acde',self.F[self.curr_root][i][j+1],self.M[self.curr_root][j])
                    tmp_sum2 = self.einsum('ydbe,acde->abcy',self.mpo.ops[i][j],tmp_sum1)
                    self.F[self.curr_root][i][j] = self.einsum('bxc,abcy->xya',np.conj(self.M[self.curr_root][j]),tmp_sum2)

    def update_f(self,j,direction):
        if self.verbose > 4:
            print('\t'*2+'Updating F at site {}'.format(j))
        for root_ind in range(self.curr_root+1):
            if direction is 'right':
                for i in range(self.mpo.nops):
                    if self.mpo.ops[i][j] is None:
                        tmp_sum1 = self.einsum('jlp,ijk->iklp',self.F[root_ind][i][j],np.conj(self.M[root_ind][j]))
                        self.F[root_ind][i][j+1] = self.einsum('npq,nkmp->kmq',self.M[root_ind][j],tmp_sum1)
                    else:
                        tmp_sum1 = self.einsum('jlp,ijk->iklp',self.F[root_ind][i][j],np.conj(self.M[root_ind][j]))
                        tmp_sum2 = self.einsum('lmin,iklp->kmnp',self.mpo.ops[i][j],tmp_sum1)
                        self.F[root_ind][i][j+1] = self.einsum('npq,kmnp->kmq',self.M[root_ind][j],tmp_sum2)
            elif direction is 'left':
                for i in range(self.mpo.nops):
                    if self.mpo.ops[i][j] is None:
                        tmp_sum1 = self.einsum('cdf,eaf->acde',self.F[root_ind][i][j+1],self.M[root_ind][j])
                        self.F[root_ind][i][j] = self.einsum('bxc,acyb->xya',np.conj(self.M[root_ind][j]),tmp_sum1)
                    else:
                        tmp_sum1 = self.einsum('cdf,eaf->acde',self.F[root_ind][i][j+1],self.M[root_ind][j])
                        tmp_sum2 = self.einsum('ydbe,acde->abcy',self.mpo.ops[i][j],tmp_sum1)
                        self.F[root_ind][i][j] = self.einsum('bxc,abcy->xya',np.conj(self.M[root_ind][j]),tmp_sum2)
            else:
                raise NameError('Direction must be left or right')

    def add_noise_func(self,i):
        if self.add_noise:
            if self.verbose > 6:
                print('\t\tAdding Noise')
            max_noise = np.amax(self.M[self.curr_root][i])*(10**(-(self.currIterCnt-1)/2))
            (n1,n2,n3) = self.M[self.curr_root][i].shape
            noise = np.random.rand(n1,n2,n3)*max_noise
            self.M[self.curr_root][i] += noise

    def local_optimization(self,j):
        if self.verbose > 4:
            print('\t'*2+'Local Optimization at site {}'.format(j))
        sgn = 1.0
        if (self.hamType is "tasep") or (self.hamType is "sep") or (self.hamType is "sep_2d"): sgn = -1.0
        (n1,n2,n3) = self.M[self.curr_root][j].shape
        self.num_opt_fun_calls = 0
        def opt_fun(x):
            self.num_opt_fun_calls += 1
            if self.verbose > 6:
                print('\t'*5+'Eigenvalue Iteration')
            x_reshape = np.reshape(x,(n1,n2,n3))
            #fin_sum = np.zeros(x_reshape.shape)
            fin_sum_td = np.zeros(x_reshape.shape)
            for i in range(self.mpo.nops):
                if self.mpo.ops[i][j] is None:
                    in_sum1_td = np.tensordot(self.F[self.curr_root][i][j+1],x_reshape,axes=([2],[2]))
                    fin_sum_td += sgn*np.swapaxes(np.swapaxes(np.tensordot(self.F[self.curr_root][i][j],in_sum1_td,axes=([1,2],[1,3])),1,2),0,1)
                else:
                    in_sum1_td = np.tensordot(self.F[self.curr_root][i][j+1],x_reshape,axes=([2],[2]))
                    in_sum2_td = np.tensordot(self.mpo.ops[i][j],in_sum1_td,axes=([1,3],[1,2]))
                    fin_sum_td += sgn*np.swapaxes(np.tensordot(self.F[self.curr_root][i][j],in_sum2_td,axes=([1,2],[0,3])),0,1)
            return np.reshape(fin_sum_td,-1)
        def precond(dx,e,x0):
            # function(dx, e, x0) => array_like_dx
            return dx
        self.add_noise_func(j)
        if self.curr_root > 0:
            init_guess = []
            for root_ind in range(self.curr_root+1):
                init_guess.insert(len(init_guess),np.reshape(self.M[root_ind][j],-1))
        else:
            init_guess = np.reshape(self.M[self.curr_root][j],-1)
        #print(len(init_guess)-1)
        E,v = self.eig(opt_fun,init_guess,precond,max_cycle=self.max_eig_iter,nroots=self.curr_root+1)
        if self.curr_root > 0:
            print('E = {}'.format(E))
            E = E[self.curr_root]
            v = v[self.curr_root]
        self.M[self.curr_root][j] = np.reshape(v,(n1,n2,n3))
        if self.verbose > 3:
            print('\t'+'Optimization Complete at {}\n\t\tEnergy = {}'.format(j,sgn*E))
            if self.verbose > 4:
                print('\t\t\t'+'Number of optimization function calls = {}'.format(self.num_opt_fun_calls))
        return sgn*E

    def calc_observables(self,site):
        if self.verbose > 5:
            print('\t'*2+'Calculating Observables')
        if (self.hamType is "heis") or (self.hamType is "heis_2d") or (self.hamType is 'ising'):
            tmp_tens = np.tensordot(self.mpo.Sx,self.M[self.curr_root][site],axes=([1],[0]))
            self.calc_spin_x[self.curr_root][site] = np.tensordot(np.conj(self.M[self.curr_root][site]),tmp_tens,axes=([0,1,2],[0,1,2]))
            tmp_tens = np.tensordot(self.mpo.Sy,self.M[self.curr_root][site],axes=([1],[0]))
            self.calc_spin_y[self.curr_root][site] = np.tensordot(np.conj(self.M[self.curr_root][site]),tmp_tens,axes=([0,1,2],[0,1,2]))
            tmp_tens = np.tensordot(self.mpo.Sz,self.M[self.curr_root][site],axes=([1],[0]))
            self.calc_spin_y[self.curr_root][site] = np.tensordot(np.conj(self.M[self.curr_root][site]),tmp_tens,axes=([0,1,2],[0,1,2]))
        elif (self.hamType is "tasep") or (self.hamType is "sep") or (self.hamType is "sep_2d"):
            tmp_tens = np.tensordot(self.mpo.v,self.M[self.curr_root][site],axes=([1],[0]))
            self.calc_empty[self.curr_root][site] = np.tensordot(np.conj(self.M[self.curr_root][site]),tmp_tens,axes=([0,1,2],[0,1,2]))
            tmp_tens = np.tensordot(self.mpo.n,self.M[self.curr_root][site],axes=([1],[0]))
            self.calc_occ[self.curr_root][site] = np.tensordot(np.conj(self.M[self.curr_root][site]),tmp_tens,axes=([0,1,2],[0,1,2]))
        if self.verbose > 4:
            print('\t'*2+'Total Number of particles: {}'.format(np.sum(self.calc_occ[self.curr_root])))

    def energy_contraction(self,j):
        E = 0
        for i in range(self.mpo.nops):
            if self.mpo.ops[i][j] is None:
                E += self.einsum('ijk,olp,mio,nkp->',self.F[self.curr_root][i][j],self.F[self.curr_root][i][j+1],np.conjugate(self.M[self.curr_root][j]),self.M[self.curr_root][j])
            else:
                E += self.einsum('ijk,jlmn,olp,mio,nkp->',self.F[self.curr_root][i][j],self.mpo.ops[i][j],self.F[self.curr_root][i][j+1],np.conjugate(self.M[self.curr_root][j]),self.M[self.curr_root][j])
        return E

    def plot_observables(self):
        if self.plotExpVals:
            plt.ion()
            if not self.exp_val_figure[self.curr_root]:
                self.exp_val_figure[self.curr_root] = plt.figure()
                self.angle = 0
            else:
                plt.figure(self.exp_val_figure[self.curr_root].number)
            plt.cla()
            if (self.hamType is "tasep") or (self.hamType is "sep"):
                plt.plot(range(0,int(self.N)),self.calc_occ[self.curr_root],linewidth=3)
                plt.ylabel('Average Occupation',fontsize=20)
                plt.xlabel('Site',fontsize=20)
            elif (self.hamType is "sep_2d"):
                plt.clf()
                x,y = (np.arange(self.mpo.Nx),np.arange(self.mpo.Ny))
                currPlot = plt.imshow(np.flipud(np.real(np.reshape(self.calc_occ[self.curr_root],(self.mpo.Nx,self.mpo.Ny))).transpose()),origin='lower')
                plt.colorbar(currPlot)
                #plt.clim(0,1)
                plt.gca().set_xticks(range(len(x)))
                plt.gca().set_yticks(range(len(y)))
                plt.gca().set_xticklabels(x)
                plt.gca().set_yticklabels(y)
                plt.gca().grid(False)
            elif (self.hamType is "heis")  or (self.hamType is 'ising'):
                ax = self.exp_val_figure[self.curr_root].gca(projection='3d')
                x = np.arange(self.N)
                y = np.zeros(self.N)
                z = np.zeros(self.N)
                ax.scatter(x,y,z,color='k')
                plt.quiver(x,y,z,self.calc_spin_x[self.curr_root],self.calc_spin_y[self.curr_root],self.calc_spin_z[self.curr_root],pivot='tail')
                ax.set_zlim((np.min((-np.abs(np.min(self.calc_spin_z[self.curr_root])),-np.abs(np.max(self.calc_spin_z[self.curr_root])))),
                             np.max(( np.abs(np.max(self.calc_spin_z[self.curr_root])) , np.abs(np.min(self.calc_spin_z[self.curr_root]))))))
                ax.set_ylim((np.min((-np.abs(np.min(self.calc_spin_y[self.curr_root])),-np.abs(np.max(self.calc_spin_y[self.curr_root])))),
                             np.max(( np.abs(np.max(self.calc_spin_y[self.curr_root])), np.abs(np.min(self.calc_spin_y[self.curr_root]))))))
                plt.ylabel('y',fontsize=20)
                plt.xlabel('x',fontsize=20)
                ax.set_zlabel('z',fontsize=20)    
                self.angle += 3
                ax.view_init(30, self.angle)
                plt.draw()
            elif self.hamType is "heis_2d":
                ax = self.exp_val_figure[self.curr_root].gca(projection='3d')
                x, y = np.meshgrid(np.arange((-self.mpo.Ny+1)/2,(self.mpo.Ny-1)/2+1),
                                   np.arange((-self.mpo.Nx+1)/2,(self.mpo.Nx-1)/2+1))
                ax.scatter(x,y,np.zeros((self.mpo.Nx,self.mpo.Ny)),color='k')
                plt.quiver(x,y,np.zeros((self.mpo.Nx,self.mpo.Ny)),
                           np.reshape(self.calc_spin_x[self.curr_root],x.shape),
                           np.reshape(self.calc_spin_y[self.curr_root],x.shape),
                           np.reshape(self.calc_spin_z[self.curr_root],x.shape),
                           pivot='tail')
                ax.plot_surface(x, y, np.zeros((self.mpo.Nx,self.mpo.Ny)), alpha=0.2)
                ax.set_zlim((min(self.calc_spin_z[self.curr_root]),max(self.calc_spin_z[self.curr_root])))
                plt.ylabel('y',fontsize=20)
                plt.xlabel('x',fontsize=20)
                ax.set_zlabel('z',fontsize=20)
                self.angle += 3
                ax.view_init(30, self.angle)
                plt.draw()
            else:
                raise ValueError("Plotting of expectation values is not implemented for the given hamiltonian type")
            plt.pause(0.0001)

    def plot_convergence(self,i):
        if self.plotConv:
            plt.ion()
            if not self.conv_figure[self.curr_root]:
                self.conv_figure[self.curr_root] = plt.figure()
                self.y_vec = [self.E]
                self.x_vec = [i]
            else:
                plt.figure(self.conv_figure[self.curr_root].number)
                self.y_vec.insert(-1,self.E)
                self.x_vec.insert(-1,i)
            plt.cla()
            if len(self.y_vec) > 3:
                plt.plot(self.x_vec[:-2],self.y_vec[:-2],'r-',linewidth=2)
            plt.ylabel('Energy',fontsize=20)
            plt.xlabel('Site',fontsize=20)
            plt.pause(0.0001)

    def saveFinalResults(self,calcType):
        if self.verbose > 5:
            print('\t'*2+'Writing final results to output file')
        if self.saveResults:
            # Create Filename:
            filename = 'results_'+self.hamType+'_N'+str(self.N)+'_M'+str(self.maxBondDim[-1])+'_time_'+str(int(time.time()*10))
            #for i in range(len(self.hamParams)):
            #    filename += ('_'+str(self.hamParams[i]))
            if calcType is 'dmrg':
                if self.hamType is "sep_2d":
                    np.savez(self.dataFolder+'dmrg/'+filename,
                             N = self.N,
                             M = self.maxBondDim,
                             #MPS = self.M,
                             periodic_x = self.periodic_x,
                             periodic_y = self.periodic_y,
                             all_energies = self.bondDimEnergies,
                             hamParams = self.hamParams[:len(self.hamParams)-1],
                             s = self.hamParams[-1],
                             dmrg_energy = self.finalEnergy,
                             calc_empty = self.calc_empty,
                             calc_occ = self.calc_occ,
                             calc_spin_x = self.calc_spin_x,
                             calc_spin_y = self.calc_spin_y,
                             calc_spin_z = self.calc_spin_z)
                else:
                    np.savez(self.dataFolder+'dmrg/'+filename,
                             N=self.N,
                             M=self.maxBondDim,
                             #MPS = self.M,
                             hamParams = self.hamParams,
                             periodic_x = self.periodic_x,
                             periodic_y = self.periodic_y,
                             all_energies = self.bondDimEnergies,
                             dmrg_energy = self.finalEnergy,
                             calc_empty = self.calc_empty,
                             calc_occ = self.calc_occ,
                             calc_spin_x = self.calc_spin_x,
                             calc_spin_y = self.calc_spin_y,
                             calc_spin_z = self.calc_spin_z)
            elif calcType is 'mf':
                np.savez(self.dataFolder+'mf/'+filename,
                         E_mf = self.E_mf)
            elif calcType is 'ed':
                np.savez(self.dataFolder+'ed/'+filename,
                         E_ed = self.E_ed)

    def kernel(self):
        if self.verbose > 1:
            print('Beginning DMRG Ground State Calculation')
        self.t0 = time.time()
        if self.curr_root is 0:
            self.initialize_containers()
            self.generate_mps()
            self.generate_mpo()
        self.right_canonicalize_mps()
        self.calc_initial_f()
        converged = False
        self.currIterCnt = 0
        self.totIterCnt = 0
        self.calc_observables(0)
        E_prev = 0#self.energy_contraction(0)
        self.E = E_prev
        while not converged:
            # Right Sweep --------------------------
            if self.verbose > 2:
                print('\t'*0+'Right Sweep {}, E = {}'.format(self.totIterCnt,self.E))
            for i in range(int(self.N-1)):
                inside_t1 = time.time()
                self.E = self.local_optimization(i)
                self.calc_observables(i)
                self.normalize(i,'right')
                self.update_f(i,'right')
                self.plot_observables()
                self.plot_convergence(i)
                inside_t2 = time.time()
                self.inside_iter_time[self.curr_root][self.maxBondDimInd] += inside_t2-inside_t1
                self.inside_iter_cnt[self.curr_root][self.maxBondDimInd] += 1
            # Left Sweep ---------------------------
            if self.verbose > 2:
                print('\t'*0+'Left Sweep  {}, E = {}'.format(self.totIterCnt,self.E))
            for i in range(int(self.N-1),0,-1):
                inside_t1 = time.time()
                self.E = self.local_optimization(i)
                self.calc_observables(i)
                self.normalize(i,'left')
                self.update_f(i,'left')
                self.plot_observables()
                self.plot_convergence(i)
                inside_t2 = time.time()
                self.inside_iter_time[self.curr_root][self.maxBondDimInd] += inside_t2-inside_t1
                self.inside_iter_cnt[self.curr_root][self.maxBondDimInd] += 1
            # Check Convergence --------------------
            self.tf = time.time()
            self.outside_iter_time[self.curr_root][self.maxBondDimInd] += self.tf-self.t0
            self.outside_iter_cnt[self.curr_root][self.maxBondDimInd] += 1
            self.t0 = time.time()
            if np.abs(self.E-E_prev) < self.tol[self.maxBondDimInd]:
                if self.maxBondDimInd is (len(self.maxBondDim)-1):
                    self.finalEnergy = self.E
                    self.bondDimEnergies[self.curr_root][self.maxBondDimInd] = self.E
                    self.time_total = time.time() - self.time_total
                    converged = True
                    if self.verbose > 0:
                        print('\n'+'#'*75)
                        print('Converged at E = {}'.format(self.finalEnergy))
                        if self.verbose > 1:
                            print('  Final Bond Dimension = {}'.format(self.maxBondDimCurr))
                            print('  Avg time per iter for final M = {} s'.format(self.inside_iter_time[self.curr_root][self.maxBondDimInd]/\
                                                                                  self.inside_iter_cnt [self.curr_root][self.maxBondDimInd]))
                            print('  Total Time = {} s'.format(self.time_total))
                        print('#'*75+'\n')
                else:
                    if self.verbose > 1:
                        print('\n'+'-'*45)
                        print('Converged at E = {}'.format(self.E))
                        if self.verbose > 2:
                            print('  Current Bond Dimension = {}'.format(self.maxBondDimCurr))
                            print('  Avg time per inner iter = {} s'.format(self.inside_iter_time[self.curr_root][self.maxBondDimInd]/\
                                                                            self.inside_iter_cnt [self.curr_root][self.maxBondDimInd]))
                            print('  Total time for M({}) = {} s'.format(self.maxBondDimCurr,self.outside_iter_time[self.curr_root][self.maxBondDimInd]))
                            print('  Required number of iters = {}'.format(self.outside_iter_cnt[self.curr_root][self.maxBondDimInd]))
                        print('-'*45+'\n')
                    self.bondDimEnergies[self.curr_root][self.maxBondDimInd] = self.E
                    self.maxBondDimInd += 1
                    self.maxBondDimCurr = self.maxBondDim[self.maxBondDimInd]
                    self.increaseBondDim()
                    self.generate_f()
                    self.calc_initial_f()
                    self.totIterCnt += 1
                    self.currIterCnt = 0
            elif self.currIterCnt >= self.maxIter[self.maxBondDimInd]-1:
                if self.maxBondDimInd is (len(self.maxBondDim)-1):
                    self.bondDimEnergies[self.curr_root][self.maxBondDimInd] = self.E
                    self.finalEnergy = self.E
                    converged = True
                    self.time_total = time.time() - self.time_total
                    if self.verbose > 0:
                        print('\n'+'!'*75)
                        print('Not Converged at E = {}'.format(self.finalEnergy))
                        if self.verbose > 1:
                            print('  Final Bond Dimension = {}'.format(self.maxBondDimCurr))
                            print('  Avg time per iter for final M = {} s'.format(self.inside_iter_time[self.curr_root][self.maxBondDimInd]/\
                                                                                  self.inside_iter_cnt [self.curr_root][self.maxBondDimInd]))
                            print('  Total Time = {} s'.format(self.time_total))
                        print('!'*75+'\n')
                else:
                    if self.verbose > 1:
                        print('\n'+'-'*45)
                        print('Not Converged at E = {}'.format(self.E))
                        if self.verbose > 2:
                            print('  Current Bond Dimension = {}'.format(self.maxBondDimCurr))
                            print('  Avg time per inner iter = {} s'.format(self.inside_iter_time[self.curr_root][self.maxBondDimInd]/\
                                                                            self.inside_iter_cnt [self.curr_root][self.maxBondDimInd]))
                            print('  Total time for M({}) = {} s'.format(self.maxBondDimCurr,self.outside_iter_time[self.curr_root][self.maxBondDimInd]))
                            print('  Required number of iters = {}'.format(self.outside_iter_cnt[self.curr_root][self.maxBondDimInd]))
                        print('-'*45+'\n')
                    self.bondDimEnergies[self.curr_root][self.maxBondDimInd] = self.E
                    self.maxBondDimInd += 1
                    self.maxBondDimCurr = self.maxBondDim[self.maxBondDimInd]
                    self.increaseBondDim()
                    self.generate_f()
                    self.calc_initial_f()
                    self.totIterCnt += 1
                    self.currIterCnt = 0
            else:
                if self.verbose > 3:
                    print('\t'*1+'Energy Change {}\nNeeded <{}'.format(np.abs(self.E-E_prev),self.tol[self.maxBondDimInd]))
                E_prev = self.E
                self.currIterCnt += 1
                self.totIterCnt += 1
        # Check if we want multiple roots
        if self.curr_root < self.nroots-1:
            print(self.curr_root)
            self.curr_root += 1
            self.kernel()
        self.saveFinalResults('dmrg')
        return self.finalEnergy


    # ADD THE ABILITY TO DO OTHER TYPES OF CALCULATIONS FROM THE MPS OBJECT
    def exact_diag(self,maxIter=10000,tol=1e-10):
        if self.N > self.ed_limit:
            print('!'*50+'\nExact Diagonalization limited to systems of 12 or fewer sites\n'+'!'*50)
            return 0
        if not hasattr(self,'mpo'):
            self.initialize_containers()
            self.generate_mpo()
        import exactDiag_meanField
        if self.hamType is 'tasep':
            x = exactDiag_meanField.exactDiag(L=self.N,
                                              clumpSize=self.N,
                                              alpha=self.hamParams[0],
                                              gamma=0,
                                              beta=0,
                                              delta=self.hamParams[2],
                                              s=self.hamParams[1],
                                              p=1,
                                              q=0,
                                              maxIter=maxIter,
                                              tol=tol)
        elif self.hamType is 'sep':
            x = exactDiag_meanField.exactDiag(L=self.N,
                                              clumpSize=self.N,
                                              alpha=self.hamParams[0],
                                              gamma=self.hamParams[1],
                                              beta=self.hamParams[4],
                                              delta=self.hamParams[5],
                                              s=self.hamParams[6],
                                              p=self.hamParams[2],
                                              q=self.hamParams[3],
                                              maxIter=maxIter,
                                              tol=tol)
        else:
            raise ValueError("Only 1D SEP and TASEP are supported for Exact Diagonalization")
        self.E_ed = x.kernel()
        self.saveFinalResults('ed')
        return(self.E_ed)

    def mean_field(self,maxIter=10000,tol=1e-10,clumpSize=2):
        if not hasattr(self,'mpo'):
            self.initialize_containers()
            self.generate_mpo()
        import exactDiag_meanField
        if self.hamType is 'tasep':
            x = exactDiag_meanField.exactDiag(L=self.N,
                                              clumpSize=clumpSize,
                                              alpha=self.hamParams[0],
                                              gamma=0,
                                              beta=0,
                                              delta=self.hamParams[2],
                                              s=self.hamParams[1],
                                              p=1,
                                              q=0,
                                              maxIter=maxIter,
                                              tol=tol)
        elif self.hamType is 'sep':
            x = exactDiag_meanField.exactDiag(L=self.N,
                                              clumpSize=clumpSize,
                                              alpha=self.hamParams[0],
                                              gamma=self.hamParams[1],
                                              beta=self.hamParams[4],
                                              delta=self.hamParams[5],
                                              s=self.hamParams[6],
                                              p=self.hamParams[2],
                                              q=self.hamParams[3],
                                              maxIter=maxIter,
                                              tol=tol)
        else:
            raise ValueError("Only 1D SEP and TASEP are supported for Mean Field")
        self.E_mf = x.kernel()
        self.saveFinalResults('mf')
        return(self.E_mf)
