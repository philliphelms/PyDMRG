import numpy as np
import time
import mpo
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import lib.logger
import sys
from lib.storage import _Xlist
import copy

class MPS_OPT:

    def __init__(self, N=10, d=2, maxBondDim=20, tol=1e-6, maxIter=10,\
                 hamType='tasep', hamParams=(0.35,-1,2/3), target_state=0,\
                 plotExpVals=False, plotConv=False, leftMPS=False, calc_psi=False,\
                 usePyscf=True, initialGuess='rand', ed_limit=12, max_eig_iter=1000,\
                 periodic_x=False, periodic_y=False, add_noise=False, outputFile='default',\
                 saveResults=True, dataFolder='data/', verbose=3, imagTol=1e-8, 
                 incore=True, useNotConv=False):
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
            self.eig = np.linalg.eig
        self.usePyscf = usePyscf
        self.initialGuess = initialGuess
        self.ed_limit = ed_limit
        self.max_eig_iter = max_eig_iter
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.add_noise = add_noise
        self.leftMPS = leftMPS
        self.calc_psi = calc_psi
        self.imagTol = 1e-8
        if outputFile == 'default':
            self.outputFile = 'logs/dmrg_output_'+str(time.time())+'.log'
        else:
            self.outputFile = outputFile
        self.incore = incore
        self.useNotConv = useNotConv

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
        self.bondDimEntanglement = np.zeros(len(self.maxBondDim),dtype=np.complex128)
        self.entanglement_spectrum = [0]*self.N
        self.entanglement_entropy = [0]*self.N
        self.final_convergence = None
        self.current = None
        sys.stdout = lib.logger.Logger(self.outputFile)

    def generate_mps(self):
        if self.verbose > 4:
            print('\t'*2+'Generating MPS')
        if self.incore:
            self.Mr = []
        else:
            self.Mr = _Xlist()
        if self.leftMPS: 
            if self.incore:
                self.Ml = []
            else:
                self.Ml = _Xlist()
        for i in range(int(self.N/2)):
            self.Mr.append(np.zeros((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr)),dtype=np.complex_))
            if self.leftMPS: self.Ml.append(np.zeros((self.d,min(self.d**(i),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr)),dtype=np.complex_))
        if self.N%2 is 1:
            self.Mr.append(np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr)),dtype=np.complex_))
            if self.leftMPS: self.Ml.append(np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i+1),self.maxBondDimCurr)),dtype=np.complex_))
        for i in range(int(self.N/2))[::-1]:
            self.Mr.append(np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i),self.maxBondDimCurr)),dtype=np.complex_))
            if self.leftMPS: self.Ml.append(np.zeros((self.d,min(self.d**(i+1),self.maxBondDimCurr),min(self.d**(i),self.maxBondDimCurr)),dtype=np.complex_))

    def generate_mpo(self):
        if self.verbose > 4:
            print('\t'*2+'Generating MPO')
        self.mpo = mpo.MPO(self.hamType,self.hamParams,self.N_mpo,periodic_x=self.periodic_x,periodic_y=self.periodic_y)

    def set_initial_MPS(self,i):
        if self.initialGuess == "zeros":
            self.Mr[i] = np.zeros(self.Mr[i].shape,dtype=np.complex_)
            if self.leftMPS: self.Ml[i] = np.zeros(self.Ml[i].shape,dtype=np.complex_)
        elif self.initialGuess == "ones":
            self.Mr[i] = np.ones(self.Mr[i].shape,dtype=np.complex_)
            if self.leftMPS: self.Ml[i] = np.ones(self.Ml[i].shape,dtype=np.complex_)
        elif self.initialGuess == "rand":
            nx,ny,nz = self.Mr[i].shape
            self.Mr[i][0,0,0] = 100.
            tmpArr = np.random.rand(nx,ny,nz)+0j
            self.Mr[i] = tmpArr#np.random.rand(nx,ny,nz)
            if self.leftMPS: self.Ml[i] = np.random.rand(nx,ny,nz)+0j
        else:
            if self.initialGuess != 'prev':
                self.Mr[i] = self.initialGuess*np.ones(self.Mr[i].shape,dtype=np.complex_)
                if self.leftMPS: self.Ml[i] = self.initialGuess*np.ones(self.Ml[i].shape,dtype=np.complex_)

    def right_canonicalize_mps(self,initialSweep=False):
        if self.verbose > 4:
            print('\t'*2+'Performing Right Canonicalization')
        if initialSweep:
            for i in range(0,len(self.Mr))[::-1]:
                self.set_initial_MPS(i)
        for i in range(1,len(self.Mr))[::-1]:
            #if initialSweep:
            #    self.set_initial_MPS(i)
            self.canonicalize(i,'left')
            self.calc_observables(i)
        self.set_initial_MPS(0)
        self.initialGuess is 'prev'

    def calculate_entanglement(self,i,singVals):
        self.entanglement_spectrum[i] = -singVals**2*np.log(singVals**2)
        if np.isnan(np.sum(self.entanglement_spectrum[i])):
            self.entanglement_spectrum[i][np.isnan(self.entanglement_spectrum[i])] = 0
        self.entanglement_entropy[i] = np.sum(self.entanglement_spectrum[i])
        if self.verbose > 4:
            print('\t\tEntanglement Entropy = {}'.format(self.entanglement_entropy[i-1]))
            if self.verbose > 5:
                print('\t\t\tEntanglement Spectrum:\n')
                for j in range(self.N):
                    print('\t\t\t\t{}'.format(self.entanglement_spectrum[i-1]))

    def canonicalize(self,i,direction):
        if self.verbose > 4:
            print('\t'*2+'Normalization at site {} in direction: {}'.format(i,direction))
        if self.leftMPS:
            self.canonicalize_biorthonormal(i,direction)
        else:
            self.canonicalize_standard(i,direction)

    def canonicalize_biorthonormal(self,i,direction):
        if direction is 'right':
            (n1,n2,n3) = self.Mr[i].shape
            Mr_reshape = np.reshape(self.Mr[i],(n1*n2,n3))
            Ml_reshape = np.reshape(self.Ml[i],(n1*n2,n3))
            (ur,sr,vr) = np.linalg.svd(Mr_reshape,full_matrices=False)
            (ul,sl,vl) = np.linalg.svd(Ml_reshape,full_matrices=False)
            # Gauge Transform of Left State
            Xgauge = np.linalg.inv(np.einsum('ji,jk->ik',np.conj(ur),ul))
            ul = np.dot(ul,Xgauge)
            sl = self.einsum('ij,jk->ik',np.linalg.inv(Xgauge),np.diag(sl))
            self.Mr[i] = np.reshape(ur,(n1,n2,n3))
            self.Mr[i+1] = self.einsum('i,ij,kjl->kil',sr,vr,self.Mr[i+1])
            self.Ml[i] = np.reshape(ul,(n1,n2,n3))
            self.Ml[i+1] = self.einsum('ij,jk,lkm->lim',sl,vl,self.Ml[i+1])
        elif direction is 'left':
            (n1,n2,n3) = self.Mr[i].shape
            Mr_reshape = np.swapaxes(self.Mr[i],0,1)
            Mr_reshape = np.reshape(Mr_reshape,(n2,n1*n3))
            Ml_reshape = np.swapaxes(self.Ml[i],0,1)
            Ml_reshape = np.reshape(Ml_reshape,(n2,n1*n3))
            (ur,sr,vr) = np.linalg.svd(Mr_reshape,full_matrices=False)
            (ul,sl,vl) = np.linalg.svd(Ml_reshape,full_matrices=False)
            # Gauge Transform of Left State
            Xgauge = np.conj(np.linalg.inv(self.einsum('ij,kj->ki',vr,np.conj(vl))))
            vl = np.dot(Xgauge,vl)
            sl = self.einsum('ij,jk->ik',np.diag(sl),np.linalg.inv(Xgauge))
            Mr_reshape = np.reshape(vr,(n2,n1,n3))
            Ml_reshape = np.reshape(vl,(n2,n1,n3))
            self.Mr[i] = np.swapaxes(Mr_reshape,0,1)
            self.Ml[i] = np.swapaxes(Ml_reshape,0,1)
            self.Mr[i-1] = self.einsum('klj,ji,i->kli',self.Mr[i-1],ur,sr)
            self.Ml[i-1] = self.einsum('klj,ji,im->klm',self.Ml[i-1],ul,sl)
        else: 
            raise NameError('Direction must be left or right')
        self.calculate_entanglement(i,sr)

    def canonicalize_standard(self,i,direction):
        if direction is 'right':
            (n1,n2,n3) = self.Mr[i].shape
            M_reshape = np.reshape(self.Mr[i],(n1*n2,n3))
            (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
            self.Mr[i] = np.reshape(U,(n1,n2,n3))
            self.Mr[i+1] = self.einsum('i,ij,kjl->kil',s,V,self.Mr[i+1])
        elif direction is 'left':
            M_reshape = np.swapaxes(self.Mr[i],0,1)
            (n1,n2,n3) = M_reshape.shape
            M_reshape = np.reshape(M_reshape,(n1,n2*n3))
            (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
            M_reshape = np.reshape(V,(n1,n2,n3))
            self.Mr[i] = np.swapaxes(M_reshape,0,1)
            self.Mr[i-1] = self.einsum('klj,ji,i->kli',self.Mr[i-1],U,s)
        else: 
            raise NameError('Direction must be left or right')
        self.calculate_entanglement(i,s)

    def increaseBondDim(self):
        self.return_psi()
        if self.rpsi is not None:
            old_psi = self.rpsi.copy()
        if self.verbose > 3:
            print('\t'*2+'Increasing Bond Dimensions from {} to {}'.format(self.maxBondDim[self.maxBondDimInd-1],self.maxBondDimCurr))
        for i in range(int(self.N/2)):
            nx,ny,nz = self.Mr[i].shape
            sz1 = min(self.d**(i),self.maxBondDimCurr)
            sz2 = min(self.d**(i+1),self.maxBondDimCurr)
            self.Mr[i] = np.pad(self.Mr[i],((0,0),(0,sz1-ny),(0,sz2-nz)),'constant',constant_values=0j)
            if self.leftMPS: self.Ml[i] = np.pad(self.Ml[i],((0,0),(0,sz1-ny),(0,sz2-nz)),'constant',constant_values=0j)
        if self.N%2 is 1:
            i += 1
            nx,ny,nz = self.Mr[i].shape
            sz1 = min(self.d**(i),self.maxBondDimCurr)
            sz2 = min(self.d**(i),self.maxBondDimCurr)
            self.Mr[i] = np.pad(self.Mr[i],((0,0),(0,sz1-ny),(0,sz2-nz)),'constant',constant_values=0j)
            if self.leftMPS: self.Ml[i] = np.pad(self.Ml[i],((0,0),(0,sz1-ny),(0,sz2-nz)),'constant',constant_values=0j)
        for i in range(int(self.N/2))[::-1]:
            site = self.N-i-1
            nx,ny,nz = self.Mr[site].shape
            sz1 = min(self.d**(i+1),self.maxBondDimCurr)
            sz2 = min(self.d**(i),self.maxBondDimCurr)
            self.Mr[site] = np.pad(self.Mr[site],((0,0),(0,sz1-ny),(0,sz2-nz)),'constant',constant_values=0j)
            if self.leftMPS: self.Ml[site] = np.pad(self.Ml[site],((0,0),(0,sz1-ny),(0,sz2-nz)),'constant',constant_values=0j)
        self.return_psi()
        if self.rpsi is not None:
            if self.verbose > 4:
                print('\t\t\tDid increased BD change WF? {}'.format(np.sum(np.abs(old_psi-self.rpsi))))

    def calc_initial_fs(self):
        self.Fs = [None]*(self.N+1)
        self.Fs[-1] = np.array([1])
        for i in range(int(self.N)-1,-1,-1):
            self.Fs[i] = np.einsum('ijk,k->j',self.Mr[i],self.Fs[i+1])

    def generate_f(self):
        if self.verbose > 4:
            print('\t'*2+'Generating initial F arrays')
        self.F = []
        for i in range(self.mpo.nops):
            if self.incore:
                self.F.append([])
                #F_tmp = []
            else:
                #F_tmp = _Xlist()
                self.F.append(_Xlist())
            self.F[-1].append(np.array([[[1]]]))
            for j in range(int(self.N/2)):
                self.F[-1].append(np.zeros((min(self.d**(j+1),self.maxBondDimCurr),2,min(self.d**(j+1),self.maxBondDimCurr)),dtype=np.complex_))
            if self.N%2 is 1:
                self.F[-1].append(np.zeros((min(self.d**(j+2),self.maxBondDimCurr),2,min(self.d**(j+2),self.maxBondDimCurr)),dtype=np.complex_))
            for j in range(int(self.N/2)-1,0,-1):
                self.F[-1].append(np.zeros((min(self.d**(j),self.maxBondDimCurr),2,min(self.d**j,self.maxBondDimCurr)),dtype=np.complex_))
            self.F[-1].append(np.array([[[1]]]))
            #self.F.append(F_tmp)

    def operatorContract(self,Op):
        if Op is not None:
            nops = len(Op)
            contraction = [np.array([[[1]]])]*nops
            for i in range(self.N):
                for j in range(nops):
                    if Op[j][i] is None:
                        if self.leftMPS:
                            contraction[j] = self.einsum('jlo,ijk,iop->klp',contraction[j],np.conj(self.Ml[i]),self.Mr[i])
                        else:
                            contraction[j] = self.einsum('jlo,ijk,iop->klp',contraction[j],np.conj(self.Mr[i]),self.Mr[i])
                    else:
                        if self.leftMPS:
                            contraction[j] = self.einsum('jlo,ijk,lmin,nop->kmp',contraction[j],np.conj(self.Ml[i]),Op[j][i],self.Mr[i])
                        else:
                            contraction[j] = self.einsum('jlo,ijk,lmin,nop->kmp',contraction[j],np.conj(self.Mr[i]),Op[j][i],self.Mr[i])
            result = 0
            for j in range(nops):
                result += contraction[j][0,0,0]
        else: result = 0
        return result

    def squaredOperatorContract(self,Op):
        if Op is not None:
            nops = len(Op)
            contraction = [np.array([[[[1]]]])]*nops
            for i in range(self.N):
                for j in range(nops):
                    if Op[j][i] is None:
                        if self.leftMPS:
                            contraction[j] = self.einsum('jlor,ijk,irs->klos',
                                                         contraction[j],np.conj(self.Ml[i]),self.Mr[i])
                        else:
                            contraction[j] = self.einsum('jlor,ijk,irs->klos',
                                                         contraction[j],np.conj(self.Mr[i]),self.Mr[i])
                    else:
                        if self.leftMPS:
                            contraction[j] = self.einsum('jlor,ijk,lmni,opnq,qrs->kmps',
                                                         contraction[j],np.conj(self.Ml[i]),
                                                         Op[j][i],Op[j][i],self.Mr[i])
                        else:
                            contraction[j] = self.einsum('jlor,ijk,lmni,opnq,qrs->kmps',
                                                         contraction[j],np.conj(self.Mr[i]),
                                                         Op[j][i],Op[j][i],self.Mr[i])
            result = 0
            for i in range(nops):
                result += contraction[i][0,0,0,0]
        else: result = 0
        return result

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
                    tmp_sum1 = self.einsum('cdf,eaf->acde',self.F[i][j+1],self.Mr[j])
                    if self.leftMPS:
                        self.F[i][j] = self.einsum('bxc,acyb->xya',np.conj(self.Ml[j]),tmp_sum1)
                    else:
                        self.F[i][j] = self.einsum('bxc,acyb->xya',np.conj(self.Mr[j]),tmp_sum1)
                else:
                    tmp_sum1 = self.einsum('cdf,eaf->acde',self.F[i][j+1],self.Mr[j])
                    tmp_sum2 = self.einsum('ydbe,acde->abcy',self.mpo.ops[i][j],tmp_sum1)
                    if self.leftMPS:
                        self.F[i][j] = self.einsum('bxc,abcy->xya',np.conj(self.Ml[j]),tmp_sum2)
                    else:
                        self.F[i][j] = self.einsum('bxc,abcy->xya',np.conj(self.Mr[j]),tmp_sum2)
        self.calc_initial_fs()

    def update_fs(self,j,direction):
        if direction is 'right':  self.Fs[j] = np.einsum('j,ijk->k',self.Fs[j-1],self.Mr[j])
        elif direction is 'left': self.Fs[j] = np.einsum('ijk,k->j',self.Mr[j],self.Fs[j+1])
        else: 
            raise NameError('Direction must be left or right')

    def update_f(self,j,direction):
        if self.verbose > 4:
            print('\t'*2+'Updating F at site {}'.format(j))
        if direction is 'right':
            for i in range(self.mpo.nops):
                if self.mpo.ops[i][j] is None:
                    if self.leftMPS:
                        tmp_sum1 = self.einsum('jlp,ijk->iklp',self.F[i][j],np.conj(self.Ml[j]))
                    else:
                        tmp_sum1 = self.einsum('jlp,ijk->iklp',self.F[i][j],np.conj(self.Mr[j]))
                    self.F[i][j+1] = self.einsum('npq,nkmp->kmq',self.Mr[j],tmp_sum1)
                else:
                    if self.leftMPS:
                        tmp_sum1 = self.einsum('jlp,ijk->iklp',self.F[i][j],np.conj(self.Ml[j]))
                    else:
                        tmp_sum1 = self.einsum('jlp,ijk->iklp',self.F[i][j],np.conj(self.Mr[j]))
                    tmp_sum2 = self.einsum('lmin,iklp->kmnp',self.mpo.ops[i][j],tmp_sum1)
                    self.F[i][j+1] = self.einsum('npq,kmnp->kmq',self.Mr[j],tmp_sum2)
        elif direction is 'left':
            for i in range(self.mpo.nops):
                if self.mpo.ops[i][j] is None:
                    tmp_sum1 = self.einsum('cdf,eaf->acde',self.F[i][j+1],self.Mr[j])
                    if self.leftMPS:
                        self.F[i][j] = self.einsum('bxc,acyb->xya',np.conj(self.Ml[j]),tmp_sum1)
                    else:
                        self.F[i][j] = self.einsum('bxc,acyb->xya',np.conj(self.Mr[j]),tmp_sum1)
                else:
                    tmp_sum1 = self.einsum('cdf,eaf->acde',self.F[i][j+1],self.Mr[j])
                    tmp_sum2 = self.einsum('ydbe,acde->abcy',self.mpo.ops[i][j],tmp_sum1)
                    if self.leftMPS:
                        self.F[i][j] = self.einsum('bxc,abcy->xya',np.conj(self.Ml[j]),tmp_sum2)
                    else:
                        self.F[i][j] = self.einsum('bxc,abcy->xya',np.conj(self.Mr[j]),tmp_sum2)
        else: 
            raise NameError('Direction must be left or right')
        self.update_fs(j,direction)

    def add_noise_func(self,j):
        if self.add_noise:
            if self.verbose > 6:
                print('\t\tAdding Noise')
            max_noise = np.amax(self.Mr[j])*(10**(-(self.currIterCnt-1)/2))
            (n1,n2,n3) = self.Mr[j].shape
            noise = np.random.rand(n1,n2,n3)*max_noise
            self.Mr[j] += noise
            if self.leftMPS:
                noisel = np.random.rand(n1,n2,n3)*max_noise
                self.Ml[j] += noisel

    def local_optimization(self,j):
        if self.verbose > 5:
            print('\t'*3+'Using Pyscf optimization routine')
        sgn = 1.0
        if (self.hamType is "tasep") or (self.hamType is "sep") or (self.hamType is "sep_2d"): sgn = -1.0
        (n1,n2,n3) = self.Mr[j].shape
        self.num_opt_fun_calls = 0
        def opt_fun(x):
            self.num_opt_fun_calls += 1
            if self.verbose > 6:
                print('\t'*5+'Right Eigen Iteration')
            x_reshape = np.reshape(x,(n1,n2,n3))
            fin_sum = np.zeros(x_reshape.shape,dtype=np.complex_)
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
        self.davidson_rconv = True
        def callback(envs_dict):
            self.davidson_rconv = envs_dict['icyc']+2 < self.max_eig_iter
        init_rguess = np.reshape(self.Mr[j],-1)
        Er,vr = self.eig(opt_fun,init_rguess,precond,
                        max_cycle = self.max_eig_iter,
                        pick = pick_eigs,
                        follow_state = True,
                        callback = callback,
                        tol = self.tol[self.maxBondDimInd]*1.e-2,
                        nroots = min(self.target_state+1,n1*n2*n3-1))
        try:
            sort_rinds = np.argsort(np.real(E))
            Er = Er[sort_rinds[min(self.target_state,len(sort_inds)-1)]]
            vr = vr[sort_rinds[min(self.target_state,len(sort_inds)-1)]]
        except: Er = Er #vr = np.real(vr) 
        if self.leftMPS:
            self.num_opt_fun_calls = 0
            def opt_fun_H(x):
                self.num_opt_fun_calls += 1
                if self.verbose > 6:
                    print('\t'*5+'Right Eigen Iteration')
                x_reshape = np.reshape(x,(n1,n2,n3))
                fin_sum = np.zeros(x_reshape.shape,dtype=np.complex_)
                for i in range(self.mpo.nops):
                    if self.mpo.ops[i][j] is None:
                        in_sum1 = self.einsum('pnm,opi->nmoi',self.F[i][j].conj(),x_reshape)
                        fin_sum += sgn*self.einsum('ijk,mjli->lmk',self.F[i][j+1].conj(),in_sum1)
                    else:
                        in_sum1 = self.einsum('pnm,opi->nmoi',self.F[i][j].conj(),x_reshape)
                        in_sum2 = self.einsum('njol,nmoi->jlmi',self.mpo.ops[i][j].conj(),in_sum1)
                        fin_sum += sgn*self.einsum('ijk,jlmi->lmk',self.F[i][j+1].conj(),in_sum2)
                return np.reshape(fin_sum,-1)
            self.davidson_lconv = True
            def callback(envs_dict):
                self.davidson_lconv = envs_dict['icyc']+2 < self.max_eig_iter
            init_lguess = np.reshape(self.Ml[j],-1)
            El,vl = self.eig(opt_fun_H,init_lguess,precond,
                            max_cycle = self.max_eig_iter,
                            pick = pick_eigs,
                            follow_state = True,
                            callback = callback,
                            tol = self.tol[self.maxBondDimInd]*1.e-2,
                            nroots = min(self.target_state+1,n1*n2*n3-1))
            try:
                sort_linds = np.argsort(np.real(El))
                El = El[sort_linds[min(self.target_state,len(sort_inds)-1)]]
                vl = vl[sort_linds[min(self.target_state,len(sort_inds)-1)]]
            except: El = El #vl = np.real(vl) #PH Will this work?
        # Use new eigenvectors, if converged
        if self.useNotConv:
            self.Mr[j] = np.reshape(vr,(n1,n2,n3))
            if self.leftMPS:
                if self.davidson_lconv:
                    self.Ml[j] = np.reshape(vl,(n1,n2,n3))
        elif self.davidson_rconv:
            self.Mr[j] = np.reshape(vr,(n1,n2,n3))
            if self.leftMPS:
                if self.davidson_lconv:
                    self.Ml[j] = np.reshape(vl,(n1,n2,n3))
        self.add_noise_func(j)
        # Normalize accordingly
        if not ((self.hamType is "heis") or (self.hamType is "heis_2d") or (self.hamType is "ising")):
            norm_factor = np.einsum('j,ijk,k->',self.Fs[j-1],self.Mr[j],self.Fs[j+1])
            self.Mr[j] /= norm_factor
            if self.leftMPS:
                self.Ml[j] /= np.einsum('ijk,ijk->',self.Mr[j],np.conj(self.Ml[j]))
        # Print Results
        if self.verbose > 3:
            if self.davidson_rconv:
                print('\t'+'Converged at {}\tEnergy = {}'.format(j,sgn*Er))
            else:
                if self.useNotConv:
                    print('\t'+'Converged at {}\tEnergy = {}'.format(j,sgn*Er))
                else:
                    print('\t'+'Not Conv  at {}\tEnergy = {}'.format(j,self.E_curr))
            if self.verbose > 4:
                print('\t\t\t'+'Number of optimization function calls = {}'.format(self.num_opt_fun_calls))
        # Return Energy, if converged
        if self.davidson_rconv:
            return sgn*Er
        else:
            if self.useNotConv:
                return sgn*Er
            else:
                return self.E_curr

    def calc_observables(self,site):
        if self.verbose > 5:
            print('\t'*2+'Calculating Observables')
        if (self.hamType is "heis") or (self.hamType is "heis_2d") or (self.hamType is 'ising'):
            if self.leftMPS:
                self.calc_spin_x[site] = self.einsum('ijk,il,ljk->',self.Ml[site].conj(),self.mpo.Sx,self.Mr[site])
                self.calc_spin_y[site] = self.einsum('ijk,il,ljk->',self.Ml[site].conj(),self.mpo.Sy,self.Mr[site])
                self.calc_spin_z[site] = self.einsum('ijk,il,ljk->',self.Ml[site].conj(),self.mpo.Sz,self.Mr[site])
            else:
                self.calc_spin_x[site] = self.einsum('ijk,il,ljk->',np.conj(self.Mr[site]),self.mpo.Sx,self.Mr[site])
                self.calc_spin_y[site] = self.einsum('ijk,il,ljk->',np.conj(self.Mr[site]),self.mpo.Sy,self.Mr[site])
                self.calc_spin_z[site] = self.einsum('ijk,il,ljk->',np.conj(self.Mr[site]),self.mpo.Sz,self.Mr[site])
        elif (self.hamType is "tasep") or (self.hamType is "sep") or (self.hamType is "sep_2d"):
            if self.leftMPS:
                self.calc_empty[site] = np.real(self.einsum('ijk,il,ljk->',self.Ml[site].conj(),self.mpo.v,self.Mr[site]))
                self.calc_occ[site] = np.real(self.einsum('ijk,il,ljk->',self.Ml[site].conj(),self.mpo.n,self.Mr[site]))
            else:
                self.calc_empty[site] = np.real(self.einsum('ijk,il,ljk->',np.conj(self.Mr[site]),self.mpo.v,self.Mr[site]))
                self.calc_occ[site] = np.real(self.einsum('ijk,il,ljk->',np.conj(self.Mr[site]),self.mpo.n,self.Mr[site]))
        if self.verbose > 4:
            print('\t'*2+'Total Number of particles: {}'.format(np.sum(self.calc_occ)))

    def energy_contraction(self,j):
        E = 0
        for i in range(self.mpo.nops):
            if self.mpo.ops[i][j] is None:
                if self.leftMPS:
                    E += self.einsum('ijk,olp,mio,nkp->',self.F[i][j],self.F[i][j+1],np.conj(self.Ml[j]),self.Mr[j])
                else:
                    E += self.einsum('ijk,olp,mio,nkp->',self.F[i][j],self.F[i][j+1],np.conj(self.Mr[j]),self.Mr[j])
            else:
                if self.leftMPS:
                    #E += self.einsum('ijk,jlmn,olp,mio,nkp->',self.F[i][j],self.mpo.ops[i][j],self.F[i][j+1],np.conj(self.Ml[j]),self.Mr[j])
                    E += self.einsum('ijk,jlmn,olp,mio,nkp->',np.real(self.F[i][j]),self.mpo.ops[i][j],np.real(self.F[i][j+1]),np.real(self.Ml[j]),np.real(self.Mr[j]))
                else:
                    E += self.einsum('ijk,jlmn,olp,mio,nkp->',self.F[i][j],self.mpo.ops[i][j],self.F[i][j+1],np.conj(self.Mr[j]),self.Mr[j])
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
            plt.figure(self.exp_val_figure.number).canvas.manager.window.attributes('-topmost', 0)
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
            plt.figure(self.conv_figure.number).canvas.manager.window.attributes('-topmost', 0)
            plt.pause(0.0001)

    def create_data_dir(self):
        if not (self.dataFolder[-1] is '/'):
            self.dataFolder += '/'
        import os
        cwd = os.getcwd()+'/'
        if not os.path.exists(cwd+self.dataFolder):
            os.mkdir(cwd+self.dataFolder)
        if not os.path.exists(cwd+self.dataFolder+'dmrg/'):
            os.mkdir(cwd+self.dataFolder+'dmrg/')
        if not os.path.exists(cwd+self.dataFolder+'ed/'):
            os.mkdir(cwd+self.dataFolder+'ed/')
        if not os.path.exists(cwd+self.dataFolder+'mf/'):
            os.mkdir(cwd+self.dataFolder+'mf/')

    def saveFinalResults(self,calcType):
        self.create_data_dir()
        if self.verbose > 5:
            print('\t'*2+'Writing final results to output file')
        if self.saveResults:
            # Create Filename:
            #    filename += ('_'+str(self.hamParams[i]))
            # PH - Come up with a better way of naming & storing files (perhaps with subdirectories)
            filename = 'results_'+self.hamType+'_N'+str(self.N)+'_M'+str(self.maxBondDim[-1])+'_time_'+str(int(time.time()*10))
            if calcType is 'dmrg':
                # Make a dict to save MPS
                Mdict = {}
                for i in range(len(self.Mr)):
                    Mdict['Mr'+str(i)] = self.Mr[i]
                    if self.leftMPS: Mdict['Ml'+str(i)] = self.Ml[i]
                if self.hamType is "sep_2d":
                    np.savez(self.dataFolder+'dmrg/'+filename,
                             N = self.N,
                             maxBondDim = self.maxBondDim,
                             periodic_x = self.periodic_x,
                             periodic_y = self.periodic_y,
                             bondDimEnergies = self.bondDimEnergies,
                             hamParams = self.hamParams[:len(self.hamParams)-1],
                             s = self.hamParams[-1],
                             dmrg_energy = self.finalEnergy,
                             calc_empty = self.calc_empty,
                             calc_occ = self.calc_occ,
                             current = self.current,
                             **Mdict)
                else:
                    np.savez(self.dataFolder+'dmrg/'+filename,
                             N=self.N,
                             maxBondDim=self.maxBondDim,
                             hamParams = self.hamParams,
                             periodic_x = self.periodic_x,
                             periodic_y = self.periodic_y,
                             bondDimEnergies = self.bondDimEnergies,
                             dmrg_energy = self.finalEnergy,
                             calc_empty = self.calc_empty,
                             calc_occ = self.calc_occ,
                             calc_spin_x = self.calc_spin_x,
                             calc_spin_y = self.calc_spin_y,
                             calc_spin_z = self.calc_spin_z,
                             current = self.current,
                             **Mdict)
            elif calcType is 'mf':
                np.savez(self.dataFolder+'mf/'+filename,
                         E_mf = self.E_mf)
            elif calcType is 'ed':
                np.savez(self.dataFolder+'ed/'+filename,
                         E_ed = self.E_ed)

    def return_psi(self):
        if self.N < self.ed_limit and self.calc_psi:
            rpsi = np.zeros(2**self.N,dtype=np.complex128)
            if self.leftMPS: lpsi = np.zeros(2**self.N,dtype=np.complex128)
            occ = np.zeros((2**self.N,self.N),dtype=int)
            sum_occ = np.zeros(2**self.N)
            for i in range(2**self.N):
                occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(self.N-len(bin(i)[2:]))+bin(i)[2:])))
                sum_occ[i] = np.sum(occ[i,:])
            # PH - Sort Inds by blocks, optional
            #inds = np.argsort(sum_occ)
            #sum_occ = sum_occ[inds]
            #occ = occ[inds,:]
            for i in range(2**self.N):
                for j in range(self.N):
                    if j is 0:
                        tmp_mat = self.Mr[j][occ[i,j],:,:]
                        if self.leftMPS: tmp_mat_l = self.Ml[j][occ[i,j],:,:]
                    else:
                        tmp_mat = np.einsum('ij,jk->ik',tmp_mat,self.Mr[j][occ[i,j],:,:])
                        if self.leftMPS: tmp_mat_l = np.einsum('ij,jk->ik',tmp_mat_l,self.Ml[j][occ[i,j],:,:])
                rpsi[i] = tmp_mat[[0]][0][0]
                if self.leftMPS: lpsi[i] = tmp_mat_l[[0]][0][0]
            self.rpsi = rpsi
            if self.leftMPS: 
                self.lpsi = lpsi
            else:
                self.lpsi = None
        else:
            self.rpsi = None
            self.lpsi = None
        if False:
            print('\nOccupation\t\t\tred\t\t\tled')
            print('-'*100)
            for i in range(len(self.rpsi)):
                print('{}\t\t\t{},\t{}'.format(occ[i,:],np.real(self.rpsi[i]),np.real(self.lpsi[i])))

    def kernel(self):
        if self.verbose > 1:
            print('Beginning DMRG Ground State Calculation')
        self.t0 = time.time()
        self.initialize_containers()
        self.generate_mps()
        self.generate_mpo()
        self.right_canonicalize_mps(initialSweep=True)
        self.calc_initial_f()
        converged = False
        self.currIterCnt = 0
        self.totIterCnt = 0
        self.calc_observables(0)
        E_prev = self.operatorContract(self.mpo.ops)
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
                self.canonicalize(i,'right')
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
                self.canonicalize(i,'left')
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
                # Converged. at final Max Bond Dim --------------------------------------------------------------------------------
                if self.maxBondDimInd is (len(self.maxBondDim)-1):
                    self.finalEnergy = self.E_conv
                    self.bondDimEnergies[self.maxBondDimInd] = self.E_conv
                    self.bondDimEntanglement[self.maxBondDimInd] = self.entanglement_entropy[int(self.N/2)]
                    self.time_total = time.time() - self.time_total
                    converged = True
                    self.current = self.operatorContract(self.mpo.currentOp(self.hamType))
                    self.susc = self.squaredOperatorContract(self.mpo.currentOp(self.hamType)) - self.current**2
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
                # Converged, move to next Max Bond Dim -----------------------------------------------------------------------
                else:
                    self.current = self.operatorContract(self.mpo.currentOp(self.hamType))
                    self.susc = self.squaredOperatorContract(self.mpo.currentOp(self.hamType)) - self.current**2
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
                    self.bondDimEntanglement[self.maxBondDimInd] = self.entanglement_entropy[int(self.N/2)]
                    self.maxBondDimInd += 1
                    self.maxBondDimCurr = self.maxBondDim[self.maxBondDimInd]
                    self.increaseBondDim()
                    self.right_canonicalize_mps()
                    self.generate_f()
                    self.calc_initial_f()
                    self.totIterCnt += 1
                    self.currIterCnt = 0
            elif self.currIterCnt >= self.maxIter[self.maxBondDimInd]-1:
                # MaxIter Reached, Not Converged at final Bond Dim --------------------------------------------------------------------------------
                if self.maxBondDimInd is (len(self.maxBondDim)-1):
                    self.bondDimEnergies[self.maxBondDimInd] = self.E_conv
                    self.bondDimEntanglement[self.maxBondDimInd] = self.entanglement_entropy[int(self.N/2)]
                    self.finalEnergy = self.E_conv
                    converged = True
                    self.current = self.operatorContract(self.mpo.currentOp(self.hamType))
                    self.susc = self.squaredOperatorContract(self.mpo.currentOp(self.hamType)) - self.current**2
                    self.final_convergence = False
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
                # MaxIter Reached, Not Converged, move to next Max Bond Dim -----------------------------------------------------------------------
                else:
                    self.current = self.operatorContract(self.mpo.currentOp(self.hamType))
                    self.susc = self.squaredOperatorContract(self.mpo.currentOp(self.hamType)) - self.current**2
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
                    self.bondDimEntanglement[self.maxBondDimInd] = self.entanglement_entropy[int(self.N/2)]
                    self.maxBondDimInd += 1
                    self.maxBondDimCurr = self.maxBondDim[self.maxBondDimInd]
                    self.increaseBondDim()
                    self.right_canonicalize_mps()
                    self.generate_f()
                    self.calc_initial_f()
                    self.totIterCnt += 1
                    self.currIterCnt = 0
            else:
                # Not Converged, go to next Max Bond Dim --------------------------------------------------------------------------------
                if self.verbose > 3:
                    print('\t'+'-'*20+'\n\tEnergy Change {}\n\tNeeded <{}'.format(np.abs(self.E_conv-E_prev),self.tol[self.maxBondDimInd]))
                E_prev = self.E_conv
                self.currIterCnt += 1
                self.totIterCnt += 1
        self.saveFinalResults('dmrg')
        self.return_psi()
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
            self.ed = exactDiag_meanField.exactDiag(L=self.N,
                                              clumpSize=self.N,
                                              alpha=self.mpo.alpha[0],
                                              gamma=self.mpo.gamma[0],
                                              beta=self.mpo.beta[-1],
                                              delta=self.mpo.delta[-1],
                                              s=self.mpo.s,
                                              p=self.mpo.p[0],
                                              q=self.mpo.q[0],
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
                                              alpha=self.mpo.alpha[0],
                                              gamma=self.mpo.gamma[0],
                                              beta=self.mpo.beta[-1],
                                              delta=self.mpo.delta[-1],
                                              s=self.mpo.s,
                                              p=self.mpo.p[0],
                                              q=self.mpo.q[0],
                                              maxIter=maxIter,
                                              tol=tol)
        else:
            raise ValueError("Only 1D SEP and TASEP are supported for Mean Field")
        self.E_mf = self.mf.kernel()
        self.saveFinalResults('mf')
        return(self.E_mf)

def pick_eigs(w,v,nroots,x0):
    abs_imag = abs(w.imag)
    max_imag_tol = max(1e-8,min(abs_imag)*1.1)
    realidx = np.where((abs_imag < max_imag_tol))[0]
    idx = realidx[w[realidx].real.argsort()]
    return w[idx], v[:,idx], idx
