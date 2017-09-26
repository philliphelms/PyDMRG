import numpy as np
import matplotlib.pyplot as plt
import time
import mpo
import warnings
#from scipy.linalg import eig
from scipy.sparse.linalg import eigs

class MPS_DMRG:
    
    def __init__(self, L=6, d=2, D=8, tol=1e-5, max_sweep_cnt=10,
                 ham_type="heis", ham_params=(1,0), plot = False,
                 verbose = 4, fileName=False, normCheck=False,
                 distribCheck=False):
        # Input Parameters
        self.L = L
        self.d = d
        self.D = D
        self.tol = tol
        self.max_sweep_cnt = max_sweep_cnt
        self.plot_option = plot
        self.plot_cnt = 0
        self.verbose = verbose
        self.normalizationCheck = normCheck
        self.distribCheck = distribCheck
        self.save_name = fileName
        # Generate MPO
        self.mpo = mpo.MPO(ham_type,ham_params,L) 
        # MPS
        self.M = []
        a_prev = 1
        going_up = True
        for i in range(L):
            if going_up:
                a_curr = min(self.d**(i+1),self.d**(L-(i)))
                if a_curr <= a_prev:
                    going_up = False
                    a_curr = self.d**(L-(i+1))
                    a_prev = self.d**(L-(i))
            else:
                a_curr = self.d**(L-(i+1))
                a_prev = self.d**(L-(i))
            max_ind_curr = min([a_curr,self.D])
            max_ind_prev = min([a_prev,self.D])
            if going_up:
                newMat = np.random.rand(self.d,max_ind_curr,max_ind_prev)
            else:
                newMat = np.random.rand(self.d,max_ind_curr,max_ind_prev)
            self.M.insert(0,newMat)
            a_prev = a_curr
        for i in range(1,len(self.M))[::-1]:
            self.normalize(i,'left')
        # Observable Containers
        self.calc_spin_x = [None]*L
        self.calc_spin_y = [None]*L
        self.calc_spin_z = [None]*L
        self.calc_empty = [None]*L
        self.calc_full = [None]*L
    
    def initialize_f(self):
        self.F = []
        self.F.insert(0,np.array([[[1]]]))
        for site in range(1,self.L)[::-1]:
                self.F.insert(0,np.einsum('ijk,lmio,opq,kmq->jlp',\
                                              np.conjugate(self.M[site]),self.mpo.W(site),self.M[site],self.F[0]))
        self.F.insert(0,np.array([[[1]]]))
    
    def h_optimization(self,site,dir):
        if self.mpo.ham_type is "heis":
            pick_ind = 0
            which = 'SR'
        else:
            pick_ind = -1
            which = 'LR'
        h = np.einsum('ijk,jlmn,olp->mionkp',self.F[site],self.mpo.W(site),self.F[site+1]) 
        si,aim,ai,sip,aimp,aip = h.shape
        h = np.reshape(h,(si*aim*ai,sip*aimp*aip))
        #w,vl,vr = eig(h,left=True,right=True) 
        w,v = eigs(h,k=1,which=which,v0=np.reshape(self.M[site],(-1)))
        # print(w[w.argsort()])
        e = w
        e = w[(w).argsort()][pick_ind]
        """
        if self.mpo.ham_type is "heis":
            v = vl[:,(w).argsort()]
        else:
            if dir is "left":
                #v = vl[:,(w).argsort()]
                v = vr[:,w.argsort()]
            else: 
                v = vr[:,(w).argsort()]
                #v = vl[:,w.argsort()]
            #vl = vl[:,(w).argsort()]
            #vr = vr[:,(w).argsort()]
            #v = np.diag(0.5*(np.einsum('i,j->ij',vl[:,pick_ind],vl[:,pick_ind])+np.einsum('i,j->ij',vr[:,pick_ind],vr[:,pick_ind])))
        """
        self.M[site] = np.reshape(v[:,pick_ind],(si,aim,ai))
        #self.M[site] = np.reshape(v,(si,aim,ai))
        return e
    
    def normalize(self,site,dir):
        si,aim,ai = self.M[site].shape
        if dir == 'right':
            M_prev = self.M[site]
            M_prev_p1 = self.M[site+1]
            M_reduced = np.reshape(self.M[site],(si*aim,ai))
            (U,S,V) = np.linalg.svd(M_reduced,full_matrices=0)
            self.M[site] = np.reshape(U,(si,aim,ai))
            self.M[site+1] = np.einsum('i,ij,kjl->kil',S,V,self.M[site+1])
            if self.distribCheck:
                #SOMETHING!!!
                print('Distribution Check Not Yet Implemented')
            if self.normalizationCheck:
                print('\t\t\tNormalization Check: {}'.format(np.sum(np.abs(np.einsum('ijk,ikl->ijl',M_prev,M_prev_p1)-np.einsum('ijk,ikl->ijl',self.M[site],self.M[site+1])))))
        elif dir == 'left':
            M_prev = self.M[site-1]
            M_prev_p1 = self.M[site]
            M_swapped = np.swapaxes(self.M[site],0,1)
            M_reduced = np.reshape(M_swapped,(aim,si*ai))
            (U,S,V) = np.linalg.svd(M_reduced,full_matrices=0)
            self.M[site] = np.swapaxes(np.reshape(V,(aim,si,ai)),0,1)
            self.M[site-1] = np.einsum('ijk,kl,l->ijl',self.M[site-1],U,S)
            if self.distribCheck:
                #Something!!!
                print('Distribution Check Not Yet Implemented')
            if self.normalizationCheck:
                print('\t\t\tNormalization Check: {}'.format(np.sum(np.abs(np.einsum('ijk,ikl->ijl',M_prev,M_prev_p1)-np.einsum('ijk,ikl->ijl',self.M[site-1],self.M[site])))))
        
    def update_f(self,site,dir):
        if dir == 'right':
            self.F[site+1] = np.einsum('ijkl,knm,nip,lpq->mjq',\
                                     self.mpo.W(site),np.conjugate(self.M[site]),self.F[site],self.M[site])
        elif dir == 'left':
            self.F[site] = np.einsum('ijkl,knm,mjq,lpq->nip',\
                                     self.mpo.W(site),np.conjugate(self.M[site]),self.F[site+1],self.M[site])
       
    def calc_observables(self,site):
        self.energy_calc = np.einsum('ijk,jlmn,olp,mio,nkp->',\
                              self.F[site],self.mpo.W(site),self.F[site+1],np.conjugate(self.M[site]),self.M[site])
        # print(self.energy_calc)
        self.calc_spin_x[site] = np.einsum('ijk,il,ljk->',self.M[site].conj(),self.mpo.S_x,self.M[site]).real
        self.calc_spin_y[site] = np.einsum('ijk,il,ljk->',self.M[site].conj(),self.mpo.S_y,self.M[site]).real
        self.calc_spin_z[site] = np.einsum('ijk,il,ljk->',self.M[site].conj(),self.mpo.S_z,self.M[site]).real
        self.calc_empty[site] = np.einsum('ijk,il,ljk->',self.M[site].conj(),self.mpo.v,self.M[site]).real
        self.calc_full[site] = np.einsum('ijk,il,ljk->',self.M[site].conj(),self.mpo.n,self.M[site]).real
        return(self.energy_calc)
    
    def show_plot(self):
        if self.plot_option:
            plt.style.use('ggplot')
            plt.figure()
            plt.rc('text', usetex=True)
            plt.rcParams['text.latex.preamble'] = [r'\boldmath']
            plt.rc('font', family='serif')
            if self.mpo.ham_type is "heis":
                plt.subplot(3,1,1)
                plt.quiver(range(1,self.L),np.zeros(self.L-1),np.zeros(self.L-1),self.calc_spin_x[1:])
                frame1 = plt.gca()
                frame1.axes.yaxis.set_ticklabels([])
                frame1.axes.xaxis.set_ticklabels([])
                plt.ylabel('$S_x$',fontsize=20)
                plt.subplot(3,1,2)
                plt.quiver(range(1,self.L),np.zeros(self.L-1),np.zeros(self.L-1),self.calc_spin_y[1:])
                frame2 = plt.gca()
                frame2.axes.yaxis.set_ticklabels([])
                frame2.axes.xaxis.set_ticklabels([])
                plt.ylabel('$S_y$',fontsize=20)
                plt.subplot(3,1,3)
                plt.quiver(range(1,self.L),np.zeros(self.L-1),np.zeros(self.L-1),self.calc_spin_z[1:])
                frame3 = plt.gca()
                frame3.axes.yaxis.set_ticklabels([])
                plt.xlabel('Site',fontsize=20)
                plt.ylabel('$S_z$',fontsize=20)
                plt.show()
            elif self.mpo.ham_type is "tasep":
                plt.plot(np.linspace(0,self.L-1,num=self.L),self.calc_empty)
                plt.ylabel('Probability',fontsize=20)
                plt.xlabel('Site',fontsize=20)
                plt.hold(True)
                plt.plot(np.linspace(0,self.L-1,num=self.L),self.calc_full)
                plt.show()

    def run_optimization(self):
        converged = False
        sweep_cnt = 0
        self.energy_vec = [0]
        self.energy_vec_all = [0]
        self.energy_vec.insert(0,0)
        while not converged:
            if self.verbose > 0:
                print('Beginning Sweep Set {}'.format(sweep_cnt))
            if self.verbose > 1:
                print('\tBeginning Right Sweep')
            self.sweep_energy_vec = []
            for site in range(self.L-1):
                self.energy_vec_all.insert(len(self.energy_vec_all),self.h_optimization(site,'right'))
                if self.verbose > 4:
                    self.calc_observables(site)
                else:
                    self.calc_observables(site)
                self.normalize(site,'right')
                self.update_f(site,'right')
                if self.verbose > 2:
                    print('\t\tOptimized site {}: {}'.format(site,self.energy_vec_all[-1]))
                self.sweep_energy_vec.insert(len(self.sweep_energy_vec),self.energy_vec_all[-1])
            if self.verbose > 1:
                print('\tBeginning Left Sweep')
            for site in range(1,self.L)[::-1]:
                self.energy_vec_all.insert(len(self.energy_vec_all),self.h_optimization(site,'right'))
                if self.verbose > 4:
                    self.calc_observables(site)
                else:
                    self.calc_observables(site)
                self.normalize(site,'left')
                self.update_f(site,'left')
                if self.verbose > 2:
                    print('\t\tOptimized site {}: {}'.format(site,self.energy_vec_all[-1]))
                self.sweep_energy_vec.insert(len(self.sweep_energy_vec),self.energy_vec_all[-1])
            self.energy_vec.insert(len(self.energy_vec),self.sweep_energy_vec[-1].real)
            if np.abs(self.energy_vec[-1]-self.energy_vec[-2]) < self.tol:
                converged = True
                self.show_plot()
                if self.verbose > 0:
                    print(('#'*75+'\nGround state energy: {}\n'+'#'*75).format(self.energy_vec[-1]))
                if self.save_name:
                    np.savez(self.save_name,L=self.L,E=self.energy_vec,allE=self.energy_vec_all,Empty=self.calc_empty,Full=self.calc_full)
                return self.energy_vec[-1]
            elif sweep_cnt > self.max_sweep_cnt:
                converged = True
                warnings.warn('Total number of iterations exceeded limit')
                if self.verbose > 0:
                    print('Resulting calculated energy: {}'.format(self.energy_vec[-1]))
                if self.save_name:
                    np.savez(self.save_name,L=self.L,E=self.energy_vec,allE=self.energy_vec_all,Empty=self.calc_empty,Full=self.calc_full)
                return self.energy_vec[-1]
            else:
                sweep_cnt += 1
    
    def calc_ground_state(self):
        self.initialize_f()
        return self.run_optimization()

if __name__ == "__main__":
    print('No Main Function Currently Implemented)
