import numpy as np
import time
import mpo
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
#from scipy.sparse.linalg import eigs, LinearOperator

class MPS_OPT:

    def __init__(self, N=10, d=2, maxBondDim=100, tol=1e-5, maxIter=10,\
                 hamType='tasep', hamParams=(0.35,-1,2/3),target_state=0,\
                 plotExpVals=False, plotConv=False,leftMPS=False,calc_psi=True,\
                 usePyscf=True,initialGuess=0.01,ed_limit=12,max_eig_iter=1000,\
                 periodic_x=False,periodic_y=False,add_noise=False,\
                 saveResults=True,dataFolder='data/',verbose=3):
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
        self.target_state = target_state
        self.plotExpVals = plotExpVals
        self.plotConv = plotConv
        self.saveResults = saveResults
        self.dataFolder = dataFolder
        self.verbose = verbose
        if usePyscf:
            from pyscf.lib import einsum, eig
            self.einsum = einsum
            self.eig = eig
        else:
            self.einsum = np.einsum
            self.eig = np.eig
        self.usePyscf = usePyscf
        self.initialGuess = initialGuess
        self.ed_limit = ed_limit
        self.max_eig_iter = max_eig_iter
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.add_noise = add_noise
        self.leftMPS = leftMPS
        self.calc_psi = calc_psi

    def initialize_containers(self):
        if type(self.N) is not int:
            self.N = self.N[0]*self.N[1]
        self.inside_iter_time = np.zeros(len(self.maxBondDim))
        self.inside_iter_cnt = np.zeros(len(self.maxBondDim))
        self.outside_iter_time = np.zeros(len(self.maxBondDim))
        self.outside_iter_cnt = np.zeros(len(self.maxBondDim))
        self.time_total = time.time()
        self.exp_val_figure=False
        self.conv_figure=False
        self.calc_spin_x = [0]*self.N
        self.calc_spin_y = [0]*self.N 
        self.calc_spin_z = [0]*self.N
        self.calc_empty = [0]*self.N
        self.calc_occ = [0]*self.N
        self.bondDimEnergies = np.zeros(len(self.maxBondDim),dtype=np.complex128)
        self.entanglement_spectrum = [0]*self.N
        self.entanglement_entropy = [0]*self.N
        self.final_convergence = None
        self.current = None

    def generate_mps(self):
        if self.verbose > 4:
            print('\t'*2+'Generating MPS')
        self.M = []
        for i in range(int(self.N/2)):
            if self.initialGuess is "zeros":
                self.M.insert(len(self.M),np.zeros((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
            elif self.initialGuess is "ones":
                self.M.insert(len(self.M),np.ones((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
            elif self.initialGuess is "rand":
                self.M.insert(len(self.M),np.random.rand(self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))) 
            else:
                self.M.insert(len(self.M),self.initialGuess*np.ones((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
        if self.N%2 is 1: # Check if system size is odd
            if self.initialGuess is "zeros":
                self.M.insert(len(self.N),np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
            elif self.initialGuess is "ones":
                self.M.insert(len(self.M),np.ones((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
            elif self.initialGuess is "rand":
                self.M.insert(len(self.M),np.random.rand(self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr)))
            else:
                self.M.insert(len(self.M),self.initialGuess*np.ones((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
        for i in range(int(self.N/2))[::-1]:
            if self.initialGuess is "zeros":
                self.M.insert(len(self.M),np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr))))
            elif self.initialGuess is "ones":
                self.M.insert(len(self.M),np.ones((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr))))
            elif self.initialGuess is "rand":
                self.M.insert(len(self.M),np.random.rand(self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr)))
            else:
                self.M.insert(len(self.M),self.initialGuess*np.ones((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr))))
        if self.leftMPS:
            self.Ml = []
            for i in range(int(self.N/2)):
                if self.initialGuess is "zeros":
                    self.Ml.insert(len(self.Ml),np.zeros((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
                elif self.initialGuess is "ones":
                    self.Ml.insert(len(self.Ml),np.ones((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
                elif self.initialGuess is "rand":
                    self.Ml.insert(len(self.Ml),np.random.rand(self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))) 
                else:
                    self.Ml.insert(len(self.Ml),self.initialGuess*np.ones((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
            if self.N%2 is 1: # Check if system size is odd
                if self.initialGuess is "zeros":
                    self.Ml.insert(len(self.Ml),np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
                elif self.initialGuess is "ones":
                    self.Ml.insert(len(self.Ml),np.ones((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
                elif self.initialGuess is "rand":
                    self.Ml.insert(len(self.Ml),np.random.rand(self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr)))
                else:
                    self.Ml.insert(len(self.Ml),self.initialGuess*np.ones((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
            for i in range(int(self.N/2))[::-1]:
                if self.initialGuess is "zeros":
                    self.Ml.insert(len(self.Ml),np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr))))
                elif self.initialGuess is "ones":
                    self.Ml.insert(len(self.Ml),np.ones((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr))))
                elif self.initialGuess is "rand":
                    self.Ml.insert(len(self.Ml),np.random.rand(self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr)))
                else:
                    self.Ml.insert(len(self.Ml),self.initialGuess*np.ones((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr))))

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
        self.M[0] = np.swapaxes(self.M[-1],1,2)
        if self.leftMPS:
            self.Ml[0] = np.swapaxes(self.Ml[-1],1,2)

    def generate_f(self):
        if self.verbose > 4:
            print('\t'*2+'Generating initial F arrays')
        self.F = []
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
            self.F.insert(len(self.F),F_tmp)
        if self.leftMPS:
            if self.verbose > 4:
                print('\t'*2+'Generating initial left MPS F arrays')
            self.Fl = []
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
                self.Fl.insert(len(self.Fl),F_tmp)

    def normalize(self,i,direction):
        if self.verbose > 4:
            print('\t'*2+'Normalization at site {} in direction: {}'.format(i,direction))
        if direction is 'right':
            (n1,n2,n3) = self.M[i].shape
            M_reshape = np.reshape(self.M[i],(n1*n2,n3))
            (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
            self.M[i] = np.reshape(U,(n1,n2,n3))
            self.M[i+1] = self.einsum('i,ij,kjl->kil',s,V,self.M[i+1])
            # Calculate Entanglement Entropy
            self.entanglement_spectrum[i] = -s**2*np.log(s**2)
            if np.isnan(np.sum(self.entanglement_spectrum[i])):
                self.entanglement_spectrum[i][np.isnan(self.entanglement_spectrum[i])] = 0
            self.entanglement_entropy[i] = np.sum(self.entanglement_spectrum[i])
            if self.verbose > 3:
                print('\t\tEntanglement Entropy = {}'.format(self.entanglement_entropy[i]))
                if self.verbose > 5:
                    print('\t\t\tEntanglement Spectrum:\n')
                    for j in range(len(self.entanglement_spectrum[i])):
                        print('\t\t\t\t{}'.format(self.entanglement_spectrum[i]))
        elif direction is 'left':
            M_reshape = np.swapaxes(self.M[i],0,1)
            (n1,n2,n3) = M_reshape.shape
            M_reshape = np.reshape(M_reshape,(n1,n2*n3))
            (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
            M_reshape = np.reshape(V,(n1,n2,n3))
            self.M[i] = np.swapaxes(M_reshape,0,1)
            self.M[i-1] = self.einsum('klj,ji,i->kli',self.M[i-1],U,s)
            # Calculate Entanglement Entropy
            self.entanglement_spectrum[i] = -s**2*np.log(s**2)
            if np.isnan(np.sum(self.entanglement_spectrum[i])):
                self.entanglement_spectrum[i][np.isnan(self.entanglement_spectrum[i])] = 0
            self.entanglement_entropy[i] = np.sum(self.entanglement_spectrum[i])
            if self.verbose > 3:
                print('\t\tEntanglement Entropy = {}'.format(self.entanglement_entropy[i-1]))
                if self.verbose > 5:
                    print('\t\t\tEntanglement Spectrum:\n')
                    for j in range(len(self.entanglement_spectrum[i-1])):
                        print('\t\t\t\t{}'.format(self.entanglement_spectrum[i-1]))
        else:
            raise NameError('Direction must be left or right')
        if self.leftMPS:
            if self.verbose > 4:
                print('\t'*2+'Normalization of left MPS at site {} in direction: {}'.format(i,direction))
            if direction is 'right':
                (n1,n2,n3) = self.Ml[i].shape
                M_reshape = np.reshape(self.Ml[i],(n1*n2,n3))
                (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
                self.Ml[i] = np.reshape(U,(n1,n2,n3))
                self.Ml[i+1] = self.einsum('i,ij,kjl->kil',s,V,self.Ml[i+1])
            elif direction is 'left':
                M_reshape = np.swapaxes(self.Ml[i],0,1)
                (n1,n2,n3) = M_reshape.shape
                M_reshape = np.reshape(M_reshape,(n1,n2*n3))
                (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
                M_reshape = np.reshape(V,(n1,n2,n3))
                self.Ml[i] = np.swapaxes(M_reshape,0,1)
                self.Ml[i-1] = self.einsum('klj,ji,i->kli',self.Ml[i-1],U,s)

    def increaseBondDim(self):
        self.return_psi()
        old_psi = self.rpsi.copy()
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
            nx,ny,nz = self.M[i].shape
            Mnew[i][:nx,:ny,:nz] = self.M[i]
            self.M[i] = Mnew[i]
        if self.leftMPS:
            if self.verbose > 3:
                print('\t'*2+'Increasing left MPS Bond Dimensions from {} to {}'.format(self.maxBondDim[self.maxBondDimInd-1],self.maxBondDimCurr))
            Mnew = []
            for i in range(int(self.N/2)):
                Mnew.insert(len(Mnew),np.zeros((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
            if self.N%2 is 1:
                Mnew.insert(len(Mnew),np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr))))
            for i in range(int(self.N/2))[::-1]:
                Mnew.insert(len(Mnew),np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**i,self.maxBondDimCurr))))
            for i in range(len(Mnew)):
                nx,ny,nz = self.Ml[i].shape
                Mnew[i][:nx,:ny,:nz] = self.Ml[i]
                self.Ml[i] = Mnew[i]
        self.return_psi()
        print(np.sum(np.abs(old_psi-self.rpsi)))

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
                    tmp_sum1 = self.einsum('cdf,eaf->acde',self.F[i][j+1],self.M[j])
                    self.F[i][j] = self.einsum('bxc,acyb->xya',np.conj(self.M[j]),tmp_sum1)
                else:
                    tmp_sum1 = self.einsum('cdf,eaf->acde',self.F[i][j+1],self.M[j])
                    tmp_sum2 = self.einsum('ydbe,acde->abcy',self.mpo.ops[i][j],tmp_sum1)
                    self.F[i][j] = self.einsum('bxc,abcy->xya',np.conj(self.M[j]),tmp_sum2)
        if self.leftMPS:
            if self.verbose > 3:
                print('\t'*2+'Calculating all entries for left MPS in F')
            for i in range(self.mpo.nops):
                if self.verbose > 4:
                    print('\t'*3+'For Operator {}/{}'.format(i,self.mpo.nops))
                for j in range(int(self.N)-1,0,-1):
                    if self.verbose > 5:
                        print('\t'*3+'at site {}'.format(j))
                    if self.mpo.ops[i][j] is None:
                        tmp_sum1 = self.einsum('cdf,eaf->acde',self.Fl[i][j+1],self.Ml[j])
                        self.Fl[i][j] = self.einsum('bxc,acyb->xya',np.conj(self.Ml[j]),tmp_sum1)
                    else:
                        tmp_sum1 = self.einsum('cdf,eaf->acde',self.Fl[i][j+1],self.Ml[j])
                        tmp_sum2 = self.einsum('ydbe,acde->abcy',self.mpo.ops[i][j],tmp_sum1)
                        self.Fl[i][j] = self.einsum('bxc,abcy->xya',np.conj(self.Ml[j]),tmp_sum2)

    def update_f(self,j,direction):
        if self.verbose > 4:
            print('\t'*2+'Updating F at site {}'.format(j))
        if direction is 'right':
            for i in range(self.mpo.nops):
                if self.mpo.ops[i][j] is None:
                    tmp_sum1 = self.einsum('jlp,ijk->iklp',self.F[i][j],np.conj(self.M[j]))
                    self.F[i][j+1] = self.einsum('npq,nkmp->kmq',self.M[j],tmp_sum1)
                else:
                    tmp_sum1 = self.einsum('jlp,ijk->iklp',self.F[i][j],np.conj(self.M[j]))
                    tmp_sum2 = self.einsum('lmin,iklp->kmnp',self.mpo.ops[i][j],tmp_sum1)
                    self.F[i][j+1] = self.einsum('npq,kmnp->kmq',self.M[j],tmp_sum2)
        elif direction is 'left':
            for i in range(self.mpo.nops):
                if self.mpo.ops[i][j] is None:
                    tmp_sum1 = self.einsum('cdf,eaf->acde',self.F[i][j+1],self.M[j])
                    self.F[i][j] = self.einsum('bxc,acyb->xya',np.conj(self.M[j]),tmp_sum1)
                else:
                    tmp_sum1 = self.einsum('cdf,eaf->acde',self.F[i][j+1],self.M[j])
                    tmp_sum2 = self.einsum('ydbe,acde->abcy',self.mpo.ops[i][j],tmp_sum1)
                    self.F[i][j] = self.einsum('bxc,abcy->xya',np.conj(self.M[j]),tmp_sum2)
        else:
            raise NameError('Direction must be left or right')
        if self.leftMPS:
            if self.verbose > 4:
                print('\t'*2+'Updating left MPS F at site {}'.format(j))
            if direction is 'right':
                for i in range(self.mpo.nops):
                    if self.mpo.ops[i][j] is None:
                        tmp_sum1 = self.einsum('jlp,ijk->iklp',self.Fl[i][j],np.conj(self.Ml[j]))
                        self.Fl[i][j+1] = self.einsum('npq,nkmp->kmq',self.Ml[j],tmp_sum1)
                    else:
                        tmp_sum1 = self.einsum('jlp,ijk->iklp',self.Fl[i][j],np.conj(self.Ml[j]))
                        tmp_sum2 = self.einsum('lmin,iklp->kmnp',self.mpo.ops[i][j],tmp_sum1)
                        self.Fl[i][j+1] = self.einsum('npq,kmnp->kmq',self.Ml[j],tmp_sum2)
            elif direction is 'left':
                for i in range(self.mpo.nops):
                    if self.mpo.ops[i][j] is None:
                        tmp_sum1 = self.einsum('cdf,eaf->acde',self.Fl[i][j+1],self.Ml[j])
                        self.Fl[i][j] = self.einsum('bxc,acyb->xya',np.conj(self.Ml[j]),tmp_sum1)
                    else:
                        tmp_sum1 = self.einsum('cdf,eaf->acde',self.Fl[i][j+1],self.Ml[j])
                        tmp_sum2 = self.einsum('ydbe,acde->abcy',self.mpo.ops[i][j],tmp_sum1)
                        self.Fl[i][j] = self.einsum('bxc,abcy->xya',np.conj(self.Ml[j]),tmp_sum2)

    def local_optimization(self,i):
        if self.verbose > 4:
            print('\t'*2+'Local optimization at site {}'.format(i))
        if self.usePyscf:
            return self.pyscf_optimization(i)
        else:
            return self.slow_optimization(i)

    def add_noise_func(self,j):
        if self.add_noise:
            if self.verbose > 6:
                print('\t\tAdding Noise')
            max_noise = np.amax(self.M[j])*(10**(-(self.currIterCnt-1)/2))
            (n1,n2,n3) = self.M[j].shape
            noise = np.random.rand(n1,n2,n3)*max_noise
            self.M[j] += noise

    def pyscf_optimization(self,j):
        if self.verbose > 5:
            print('\t'*3+'Using Pyscf optimization routine')
        sgn = 1.0
        if (self.hamType is "tasep") or (self.hamType is "sep") or (self.hamType is "sep_2d"): sgn = -1.0
        (n1,n2,n3) = self.M[j].shape
        self.num_opt_fun_calls = 0
        def opt_fun(x):
            self.num_opt_fun_calls += 1
            if self.verbose > 6:
                print('\t'*5+'Eigenvalue Iteration')
            x_reshape = np.reshape(x,(n1,n2,n3))
            fin_sum = np.zeros(x_reshape.shape,dtype=np.complex128)
            for i in range(self.mpo.nops):
                if self.mpo.ops[i][j] is None:
                    in_sum1 =  self.einsum('ijk,lmk->ijlm',self.F[i][j+1],x_reshape)
                    fin_sum += sgn*self.einsum('pnm,inom->opi',self.F[i][j],in_sum1)
                else:
                    in_sum1 =  self.einsum('ijk,lmk->ijlm',self.F[i][j+1],x_reshape)
                    in_sum2 = self.einsum('njol,ijlm->noim',self.mpo.ops[i][j],in_sum1)
                    fin_sum += sgn*self.einsum('pnm,noim->opi',self.F[i][j],in_sum2)
            return np.reshape(fin_sum,-1)
        def precond(dx,e,x0):
            return dx
        def callback(envs_dict):
            self.davidson_conv = envs_dict['icyc']+2 < self.max_eig_iter
        init_guess = np.reshape(self.M[j],-1)
        if self.leftMPS:
            E,vl,vr = self.eig(opt_fun,init_guess,precond,
                               max_cycle = self.max_eig_iter,
                               pick = pick_eigs,
                               left = self.leftMPS,
                               follow_state = True,
                               tol = self.tol[self.maxBondDimInd],
                               callback = callback,
                               nroots=min(self.target_state+1,n1*n2*n3-1))
            # Make vl & vr biorthonormal:
            vl /= np.sum(vl*vr)
        else:
            E,vr = self.eig(opt_fun,init_guess,precond,
                                 max_cycle=self.max_eig_iter,
                                 pick = pick_eigs,
                                 follow_state = True,
                                 tol = self.tol[self.maxBondDimInd],
                                 callback = callback,
                                 nroots=min(self.target_state+1,n1*n2*n3-1))
        sort_inds = np.argsort(np.real(E))#[::-1]
        try:
            E = E[sort_inds[min(self.target_state,len(sort_inds)-1)]]
            if self.leftMPS:
                vl = vl[sort_inds[min(self.target_state,len(sort_inds)-1)]]
            vr = vr[sort_inds[min(self.target_state,len(sort_inds)-1)]]
        except:
            E = E
        if self.davidson_conv:
            self.M[j] = np.reshape(vr,(n1,n2,n3))
            if self.leftMPS:
                self.Ml[j] = np.reshape(vl,(n1,n2,n3))
        self.add_noise_func(j)
        if self.verbose > 3:
            if self.davidson_conv:
                print('\t'+'Optimization Converged at {}\n\t\tEnergy = {}'.format(j,sgn*E))
            else:
                print('\t'+'Optimization Not Converged at {}\n\t\tEnergy = {}'.format(j,self.E_curr))
            if self.verbose > 4:
                print('\t\t\t'+'Number of optimization function calls = {}'.format(self.num_opt_fun_calls))
        if self.davidson_conv:
            return sgn*E
        else:
            return self.E_curr

    def slow_optimization(self,i):
        if self.verbose > 5:
            print('\t'*4+'Using slow optimization routine')
        sgn = 1.0
        if (self.hamType is "tasep") or (self.hamType is "sep") or (self.hamType is "sep_2d"): sgn = -1.0
        for j in range(self.mpo.nops):
            if j is 0:
                if self.mpo.ops[j][i] is None:
                    H = sgn*self.einsum('jlp,kmq->ljkmpq',self.F[j][i],self.F[j][i+1])
                else:
                    H = sgn*self.einsum('jlp,lmin,kmq->ijknpq',self.F[j][i],self.mpo.ops[j][i],self.F[j][i+1])
            else:
                if self.mpo.ops[j][i] is None:
                    H += sgn*self.einsum('jlp,kmq->ljkmpq',self.F[j][i],self.F[j][i+1])
                else:
                    H += sgn*self.einsum('jlp,lmin,kmq->ijknpq',self.F[j][i],self.mpo.ops[j][i],self.F[j][i+1])
        (n1,n2,n3,n4,n5,n6) = H.shape
        H = np.reshape(H,(n1*n2*n3,n4*n5*n6))
        if (self.hamType is "tasep") or (self.hamType is "sep") or (self.hamType is "sep_2d"): H = -H
        u,v = self.eig(H)
        u_sort = u[np.argsort(u)]
        v = v[:,np.argsort(u)]
        ind = 0
        for j in range(len(u_sort)):
            if np.abs(np.imag(u_sort[j])) < 1e-8:
                ind = j
            break
        E = u_sort[ind]
        v = v[:,ind]
        self.M[i] = np.reshape(v,(n1,n2,n3))
        if self.verbose > 3:
            print('\t'+'Optimization Complete at {}\n\t\tEnergy = {}'.format(i,E))
        return E

    def calc_observables(self,site):
        if self.verbose > 5:
            print('\t'*2+'Calculating Observables')
        if (self.hamType is "heis") or (self.hamType is "heis_2d") or (self.hamType is 'ising'):
            if self.leftMPS:
                self.calc_spin_x[site] = self.einsum('ijk,il,ljk->',self.Ml[site],self.mpo.Sx,self.M[site])
                self.calc_spin_y[site] = self.einsum('ijk,il,ljk->',self.Ml[site],self.mpo.Sy,self.M[site])
                self.calc_spin_z[site] = self.einsum('ijk,il,ljk->',self.Ml[site],self.mpo.Sz,self.M[site])
            else:
                self.calc_spin_x[site] = self.einsum('ijk,il,ljk->',np.conj(self.M[site]),self.mpo.Sx,self.M[site])
                self.calc_spin_y[site] = self.einsum('ijk,il,ljk->',np.conj(self.M[site]),self.mpo.Sy,self.M[site])
                self.calc_spin_z[site] = self.einsum('ijk,il,ljk->',np.conj(self.M[site]),self.mpo.Sz,self.M[site])
        elif (self.hamType is "tasep") or (self.hamType is "sep") or (self.hamType is "sep_2d"):
            if self.leftMPS:
                self.calc_empty[site] = self.einsum('ijk,il,ljk->',self.Ml[site],self.mpo.v,self.M[site])
                self.calc_occ[site] = self.einsum('ijk,il,ljk->',self.Ml[site],self.mpo.n,self.M[site])
            else:
                self.calc_empty[site] = self.einsum('ijk,il,ljk->',np.conj(self.M[site]),self.mpo.v,self.M[site])
                self.calc_occ[site] = self.einsum('ijk,il,ljk->',np.conj(self.M[site]),self.mpo.n,self.M[site])
            # Try to calculate current
            if self.hamType is "sep":
                self.current = self.mpo.exp_p[-1]*self.calc_occ[-1]*self.mpo.exp_delta[-1]-self.mpo.exp_q[-1]*self.calc_empty[-1]*self.mpo.exp_beta[-1]
            if self.hamType is "tasep":
                self.current = self.calc_occ[-1]*self.mpo.beta
        if self.verbose > 4:
            print('\t'*2+'Total Number of particles: {}'.format(np.sum(self.calc_occ)))

    def energy_contraction(self,j):
        E = 0
        for i in range(self.mpo.nops):
            if self.mpo.ops[i][j] is None:
                if self.leftMPS:
                    E += self.einsum('ijk,olp,mio,nkp->',self.Fl[i][j],self.F[i][j+1],np.conj(self.Ml[j]),self.M[j])
                else:
                    E += self.einsum('ijk,olp,mio,nkp->',self.F[i][j],self.F[i][j+1],np.conj(self.M[j]),self.M[j])
            else:
                if self.leftMPS:
                    E += self.einsum('ijk,jlmn,olp,mio,nkp->',self.Fl[i][j],self.mpo.ops[i][j],self.F[i][j+1],np.conj(self.Ml[j]),self.M[j])
                else:
                    E += self.einsum('ijk,jlmn,olp,mio,nkp->',self.F[i][j],self.mpo.ops[i][j],self.F[i][j+1],np.conj(self.M[j]),self.M[j])
        return E

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
                plt.plot(range(0,int(self.N)),self.calc_occ,linewidth=3)
                plt.ylabel('Average Occupation',fontsize=20)
                plt.xlabel('Site',fontsize=20)
            elif (self.hamType is "sep_2d"):
                plt.clf()
                x,y = (np.arange(self.mpo.Nx),np.arange(self.mpo.Ny))
                currPlot = plt.imshow(np.flipud(np.real(np.reshape(self.calc_occ,(self.mpo.Nx,self.mpo.Ny))).transpose()),origin='lower')
                plt.colorbar(currPlot)
                #plt.clim(0,1)
                plt.gca().set_xticks(range(len(x)))
                plt.gca().set_yticks(range(len(y)))
                plt.gca().set_xticklabels(x)
                plt.gca().set_yticklabels(y)
                plt.gca().grid(False)
            elif (self.hamType is "heis")  or (self.hamType is 'ising'):
                ax = self.exp_val_figure.gca(projection='3d')
                x = np.arange(self.N)
                y = np.zeros(self.N)
                z = np.zeros(self.N)
                ax.scatter(x,y,z,color='k')
                plt.quiver(x,y,z,self.calc_spin_x,self.calc_spin_y,self.calc_spin_z,pivot='tail')
                ax.set_zlim((np.min((-np.abs(np.min(self.calc_spin_z)),-np.abs(np.max(self.calc_spin_z)))),
                             np.max(( np.abs(np.max(self.calc_spin_z)) , np.abs(np.min(self.calc_spin_z))))))
                ax.set_ylim((np.min((-np.abs(np.min(self.calc_spin_y)),-np.abs(np.max(self.calc_spin_y)))),
                             np.max(( np.abs(np.max(self.calc_spin_y)), np.abs(np.min(self.calc_spin_y))))))
                plt.ylabel('y',fontsize=20)
                plt.xlabel('x',fontsize=20)
                ax.set_zlabel('z',fontsize=20)    
                self.angle += 3
                ax.view_init(30, self.angle)
                plt.draw()
            elif self.hamType is "heis_2d":
                ax = self.exp_val_figure.gca(projection='3d')
                x, y = np.meshgrid(np.arange((-self.mpo.Ny+1)/2,(self.mpo.Ny-1)/2+1),
                                   np.arange((-self.mpo.Nx+1)/2,(self.mpo.Nx-1)/2+1))
                ax.scatter(x,y,np.zeros((self.mpo.Nx,self.mpo.Ny)),color='k')
                plt.quiver(x,y,np.zeros((self.mpo.Nx,self.mpo.Ny)),
                           np.reshape(self.calc_spin_x,x.shape),
                           np.reshape(self.calc_spin_y,x.shape),
                           np.reshape(self.calc_spin_z,x.shape),
                           pivot='tail')
                ax.plot_surface(x, y, np.zeros((self.mpo.Nx,self.mpo.Ny)), alpha=0.2)
                ax.set_zlim((min(self.calc_spin_z),max(self.calc_spin_z)))
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
            if not self.conv_figure:
                self.conv_figure = plt.figure()
                self.y_vec = [self.E_curr]
                self.x_vec = [i]
            else:
                plt.figure(self.conv_figure.number)
                self.y_vec.insert(-1,self.E_curr)
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

    def return_psi(self):
        if self.calc_psi:
            if self.N < 21:
                rpsi = np.zeros(2**self.N,dtype=np.complex128)
                if self.leftMPS:
                    lpsi = np.zeros(2**self.N,dtype=np.complex128)
                occ = np.zeros((2**self.N_mpo,self.N_mpo),dtype=int)
                sum_occ = np.zeros(2**self.N_mpo)
                for i in range(2**self.N):
                    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(self.N_mpo-len(bin(i)[2:]))+bin(i)[2:])))
                    sum_occ[i] = np.sum(occ[i,:])
                inds = np.argsort(sum_occ)
                sum_occ = sum_occ[inds]
                occ = occ[inds,:]
                for i in range(2**self.N):
                    for j in range(self.N):
                        if j is 0:
                            tmp_mat = self.M[j][occ[i,j],:,:]
                            if self.leftMPS:
                                tmp_mat_l = self.Ml[j][occ[i,j],:,:]
                        else:
                            tmp_mat = np.einsum('ij,jk->ik',tmp_mat,self.M[j][occ[i,j],:,:])
                            if self.leftMPS:
                                tmp_mat_l = np.einsum('ij,jk->ik',tmp_mat_l,self.Ml[j][occ[i,j],:,:])
                    rpsi[i] = tmp_mat[[0]]
                    if self.leftMPS:
                        lpsi[i] = tmp_mat[[0]]
                self.rpsi = rpsi
                #print('Right State:')
                #for i in range(len(self.rpsi)):
                #    print('{}'.format(np.real(self.rpsi[i])))
                #print(self.M)
                if self.leftMPS:
                    self.lpsi = lpsi
                    #print('lpsi = {}'.format(self.lpsi))
                    #print(self.Ml)
                else:
                    self.lpsi = None
            else:
                self.rpsi = None
                self.lpsi = None

    def kernel(self):
        if self.verbose > 1:
            print('Beginning DMRG Ground State Calculation')
        self.t0 = time.time()
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
        self.E_curr = E_prev
        self.E_conv = E_prev
        while not converged:
            # Right Sweep --------------------------
            if self.verbose > 2:
                print('\t'*0+'Right Sweep {}, E = {}'.format(self.totIterCnt,self.E_conv))
            for i in range(int(self.N-1)):
                inside_t1 = time.time()
                self.E_curr = self.local_optimization(i)
                self.calc_observables(i)
                self.normalize(i,'right')
                self.update_f(i,'right')
                self.plot_observables()
                self.plot_convergence(i)
                inside_t2 = time.time()
                self.inside_iter_time[self.maxBondDimInd] += inside_t2-inside_t1
                self.inside_iter_cnt[self.maxBondDimInd] += 1
                if i is int(self.N/2):
                    self.E_conv = self.E_curr
            # Left Sweep ---------------------------
            if self.verbose > 2:
                print('\t'*0+'Left Sweep  {}, E = {}'.format(self.totIterCnt,self.E_conv))
            for i in range(int(self.N-1),0,-1):
                inside_t1 = time.time()
                self.E_curr = self.local_optimization(i)
                self.calc_observables(i)
                self.normalize(i,'left')
                self.update_f(i,'left')
                self.plot_observables()
                self.plot_convergence(i)
                inside_t2 = time.time()
                self.inside_iter_time[self.maxBondDimInd] += inside_t2-inside_t1
                self.inside_iter_cnt[self.maxBondDimInd] += 1
                if i is int(self.N/2):
                    self.E_conv = self.E_curr
            # Check Convergence --------------------
            self.tf = time.time()
            self.outside_iter_time[self.maxBondDimInd] += self.tf-self.t0
            self.outside_iter_cnt[self.maxBondDimInd] += 1
            self.t0 = time.time()
            if np.abs((self.E_conv-E_prev)/E_prev) < self.tol[self.maxBondDimInd]:
                if self.maxBondDimInd is (len(self.maxBondDim)-1):
                    self.finalEnergy = self.E_conv
                    self.bondDimEnergies[self.maxBondDimInd] = self.E_conv
                    self.time_total = time.time() - self.time_total
                    converged = True
                    self.final_convergence = True
                    if self.verbose > 0:
                        print('\n'+'#'*75)
                        print('Converged at E = {}'.format(self.finalEnergy))
                        if self.verbose > 1:
                            print('  Final Bond Dimension = {}'.format(self.maxBondDimCurr))
                            print('  Total Current = {}'.format(self.current))
                            print('  Entanglement Entropy at center bond = {}'.format(self.entanglement_entropy[int(self.N/2)]))
                            print('  Total Number of particles: {}'.format(np.sum(self.calc_occ)))
                            if self.verbose > 4:
                                print('  Avg time per iter for final M = {} s'.format(self.inside_iter_time[self.maxBondDimInd]/\
                                                                                      self.inside_iter_cnt [self.maxBondDimInd]))
                                print('  Total Time = {} s'.format(self.time_total))
                                if self.verbose > 6:
                                    print('    Entanglement Spectrum at center bond = {}'.format(self.entanglement_spectrum[int(self.N/2)]))
                                    print('    Density = {}'.format(self.calc_occ))
                        print('#'*75+'\n')
                else:
                    if self.verbose > 1:
                        print('\n'+'-'*45)
                        print('Converged at E = {}'.format(self.E_conv))
                        if self.verbose > 2:
                            print('  Current Bond Dimension = {}'.format(self.maxBondDimCurr))
                            print('  Total Current = {}'.format(self.current))
                            print('  Entanglement Entropy at center bond = {}'.format(self.entanglement_entropy[int(self.N/2)]))
                            print('  Total Number of particles: {}'.format(np.sum(self.calc_occ)))
                            if self.verbose > 4:
                                print('  Avg time per inner iter = {} s'.format(self.inside_iter_time[self.maxBondDimInd]/\
                                                                                self.inside_iter_cnt [self.maxBondDimInd]))
                                print('  Total time for M({}) = {} s'.format(self.maxBondDimCurr,self.outside_iter_time[self.maxBondDimInd]))
                                print('  Required number of iters = {}'.format(self.outside_iter_cnt[self.maxBondDimInd]))
                                if self.verbose > 6:
                                    print('    Entanglement Spectrum at center bond = {}'.format(self.entanglement_spectrum[int(self.N/2)]))
                                    print('    Density = {}'.format(self.calc_occ))
                        print('-'*45+'\n')
                    self.bondDimEnergies[self.maxBondDimInd] = self.E_conv
                    self.maxBondDimInd += 1
                    self.maxBondDimCurr = self.maxBondDim[self.maxBondDimInd]
                    self.increaseBondDim()
                    self.generate_f()
                    self.calc_initial_f()
                    self.totIterCnt += 1
                    self.currIterCnt = 0
            elif self.currIterCnt >= self.maxIter[self.maxBondDimInd]-1:
                if self.maxBondDimInd is (len(self.maxBondDim)-1):
                    self.bondDimEnergies[self.maxBondDimInd] = self.E_conv
                    self.finalEnergy = self.E_conv
                    converged = True
                    self.final_convergence = False
                    print(self.final_convergence)
                    self.time_total = time.time() - self.time_total
                    if self.verbose > 0:
                        print('\n'+'!'*75)
                        print('Not Converged at E = {}'.format(self.finalEnergy))
                        if self.verbose > 1:
                            print('  Final Bond Dimension = {}'.format(self.maxBondDimCurr))
                            print('  Total Current = {}'.format(self.current))
                            print('  Entanglement Entropy at center bond = {}'.format(self.entanglement_entropy[int(self.N/2)]))
                            print('  Total Number of particles: {}'.format(np.sum(self.calc_occ)))
                            if self.verbose > 4:
                                print('  Avg time per iter for final M = {} s'.format(self.inside_iter_time[self.maxBondDimInd]/\
                                                                                      self.inside_iter_cnt [self.maxBondDimInd]))
                                print('  Total Time = {} s'.format(self.time_total))
                                if self.verbose > 6:
                                    print('    Entanglement Spectrum at center bond = {}'.format(self.entanglement_spectrum[int(self.N/2)]))
                                    print('    Density = {}'.format(self.calc_occ))
                        print('!'*75+'\n')
                else:
                    if self.verbose > 1:
                        print('\n'+'-'*45)
                        print('Not Converged at E = {}'.format(self.E_conv))
                        if self.verbose > 2:
                            print('  Current Bond Dimension = {}'.format(self.maxBondDimCurr))
                            print('  Total Current = {}'.format(self.current))
                            print('  Entanglement Entropy at center bond = {}'.format(self.entanglement_entropy[int(self.N/2)]))
                            print('  Total Number of particles: {}'.format(np.sum(self.calc_occ)))
                            if self.verbose > 4:
                                print('  Total time for M({}) = {} s'.format(self.maxBondDimCurr,self.outside_iter_time[self.maxBondDimInd]))
                                print('  Avg time per inner iter = {} s'.format(self.inside_iter_time[self.maxBondDimInd]/\
                                                                                self.inside_iter_cnt [self.maxBondDimInd]))
                                print('  Required number of iters = {}'.format(self.outside_iter_cnt[self.maxBondDimInd]))
                                if self.verbose > 6:
                                    print('    Entanglement Spectrum at center bond = {}'.format(self.entanglement_spectrum[int(self.N/2)]))
                                    print('    Density = {}'.format(self.calc_occ))
                        print('-'*45+'\n')
                    self.bondDimEnergies[self.maxBondDimInd] = self.E_conv
                    self.maxBondDimInd += 1
                    self.maxBondDimCurr = self.maxBondDim[self.maxBondDimInd]
                    self.increaseBondDim()
                    self.generate_f()
                    self.calc_initial_f()
                    self.totIterCnt += 1
                    self.currIterCnt = 0
            else:
                if self.verbose > 3:
                    print('\t'*1+'Energy Change {}\nNeeded <{}'.format(np.abs(self.E_conv-E_prev),self.tol[self.maxBondDimInd]))
                E_prev = self.E_conv
                self.currIterCnt += 1
                self.totIterCnt += 1
        self.saveFinalResults('dmrg')
        #self.return_psi()
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
            self.ed = exactDiag_meanField.exactDiag(L=self.N,
                                              clumpSize=self.N,
                                              alpha=self.mpo.a,
                                              gamma=0,
                                              beta=0,
                                              delta=self.mpo.b,
                                              s=self.mpo.s,
                                              p=1,
                                              q=0,
                                              maxIter=maxIter,
                                              tol=tol)
        elif self.hamType is 'sep':
            self.ed = exactDiag_meanField.exactDiag(L=self.N,
                                              clumpSize=self.N,
                                              alpha=self.mpo.a,
                                              gamma=self.mpo.g,
                                              beta=self.mpo.b,
                                              delta=self.mpo.d,
                                              s=self.mpo.s,
                                              p=self.mpo.p[1],
                                              q=self.mpo.q[1],
                                              maxIter=maxIter,
                                              tol=tol)
        else:
            raise ValueError("Only 1D SEP and TASEP are supported for Exact Diagonalization")
        self.E_ed = self.ed.kernel()
        self.saveFinalResults('ed')
        return(self.E_ed)

    def mean_field(self,maxIter=10000,tol=1e-10,clumpSize=2):
        if not hasattr(self,'mpo'):
            self.initialize_containers()
            self.generate_mpo()
        import exactDiag_meanField
        if self.hamType is 'tasep':
            self.mf = exactDiag_meanField.exactDiag(L=self.N,
                                              clumpSize=clumpSize,
                                              alpha=self.mpo.alpha,
                                              gamma=0,
                                              beta=0,
                                              delta=self.mpo.beta,
                                              s=self.mpo.s,
                                              p=1,
                                              q=0,
                                              maxIter=maxIter,
                                              tol=tol)
        elif self.hamType is 'sep':
            self.mf = exactDiag_meanField.exactDiag(L=self.N,
                                              clumpSize=clumpSize,
                                              alpha=self.mpo.alpha,
                                              gamma=self.mpo.gamma,
                                              beta=self.mpo.beta,
                                              delta=self.mpo.delta,
                                              s=self.mpo.s,
                                              p=self.mpo.p,
                                              q=self.mpo.q,
                                              maxIter=maxIter,
                                              tol=tol)
        else:
            raise ValueError("Only 1D SEP and TASEP are supported for Mean Field")
        self.E_mf = self.mf.kernel()
        self.saveFinalResults('mf')
        return(self.E_mf)

def pick_eigs(w, v, nroots, x0):
    abs_imag = abs(w.imag)
    max_imag_tol = max(1e-5,min(abs_imag)*1.1)
    realidx = np.where((abs_imag < max_imag_tol))[0]
    idx = realidx[w[realidx].real.argsort()]
    #print(w[idx][:5])
    return w[idx], v[:,idx], idx
