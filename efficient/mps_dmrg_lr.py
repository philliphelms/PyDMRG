import numpy as np
import matplotlib.pyplot as plt
import time
import mpo
import warnings
from scipy.linalg import eig as fullEig
from scipy.sparse.linalg import eigs as arnoldiEig

class MPS_DMRG:
    
    def __init__(self, L=6, d=2, D=8, tol=1e-5, max_sweep_cnt=10,
                 ham_type="heis", ham_params=(1,0), plotExpVal = True,
                 plotConv = True, verbose = 4, fileName=False, normCheck=False,
                 distribCheck=False):
        # Input Parameters
        plt.close('all')
        self.L = L
        self.d = d
        self.D = D
        self.tol = tol
        self.max_sweep_cnt = max_sweep_cnt
        self.plotExpVal = plotExpVal
        self.exp_val_figure = False
        self.plotConv = plotConv
        self.conv_figure = False
        self.verbose = verbose
        self.normalizationCheck = normCheck
        self.distribCheck = distribCheck
        self.save_name = fileName
        self.eigMethod = 'full' # 'full'
        self.compareEigs = False
        if L is 6:
            self.checkEnergyNormalization = True
        else:
            self.checkEnergyNormalization = False
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
        self.Mr = self.M
        self.Ml = self.M
        for i in range(1,len(self.M))[::-1]:
            self.normalize(i,'left')
        # Observable Containers
        self.calc_spin_x = [0]*L#[None]*L
        self.calc_spin_y = [0]*L#[None]*L
        self.calc_spin_z = [0]*L#[None]*L
        self.calc_empty = [0]*L#[None]*L
        self.calc_full = [0]*L#[None]*L
    
    def initialize_f(self):
        self.F = []
        self.Fl = []
        self.Fr = []
        self.F.insert(0,np.array([[[1]]]))
        self.Fl.insert(0,np.array([[[1]]]))
        self.Fr.insert(0,np.array([[[1]]]))
        for site in range(1,self.L)[::-1]:
                self.F.insert(0,np.einsum('ijk,lmio,opq,kmq->jlp',\
                                              np.conjugate(self.Ml[site]),self.mpo.W(site),self.Mr[site],self.F[0]))
                self.Fl.insert(0,np.einsum('ijk,lmio,opq,kmq->jlp',\
                                              np.conjugate(self.Ml[site]),self.mpo.W(site),self.Ml[site],self.Fl[0]))
                self.Fr.insert(0,np.einsum('ijk,lmio,opq,kmq->jlp',\
                                              np.conjugate(self.Mr[site]),self.mpo.W(site),self.Mr[site],self.Fr[0]))
        self.F.insert(0,np.array([[[1]]]))
    
    def h_optimization(self,site,dir):
        pick_ind = -1
        h = np.einsum('ijk,jlmn,olp->mionkp',self.F[site],self.mpo.W(site),self.F[site+1]) 
        si,aim,ai,sip,aimp,aip = h.shape
        h = np.reshape(h,(si*aim*ai,sip*aimp*aip))
        w,vr,vl = fullEig(h,left=True,right=True)
        e = w[w.argsort()][pick_ind]
        vr = vr[:,w.argsort()]
        vl = vl[:,w.argsort()]
        self.Ml[site] = np.reshape(vl[:,pick_ind],(si,aim,ai))
        self.Mr[site] = np.reshape(vr[:,pick_ind],(si,aim,ai))
        return e
    
    def normalize(self,site,dir):
        si,aim,ai = self.Ml[site].shape
        if dir == 'right':
            M_prev = self.Ml[site]
            M_prev_p1 = self.Ml[site+1]
            M_reduced = np.reshape(self.Ml[site],(si*aim,ai))
            (U,S,V) = np.linalg.svd(np.real(M_reduced),full_matrices=0)
            #(U,S,V) = np.linalg.svd(M_reduced,full_matrices=0)
            self.Ml[site] = np.reshape(U,(si,aim,ai))
            self.Ml[site+1] = np.einsum('i,ij,kjl->kil',S,V,self.Ml[site+1])
            if self.distribCheck:
                print('\t\t\tDistribution Check: {}'.format(np.sum(np.abs(np.einsum('ijk,ikl->ijl',M_prev,M_prev_p1)-np.einsum('ijk,ikl->ijl',self.Ml[site],self.Ml[site+1])))))
            if self.normalizationCheck:
                print('\t\t\tNormalization Check:\n{}'.format(np.einsum('ijk,ikl->jl',np.transpose(self.Ml[site],(0,2,1)),self.Ml[site])))
        elif dir == 'left':
            M_prev = self.Ml[site-1]
            M_prev_p1 = self.Ml[site]
            M_swapped = np.swapaxes(self.Ml[site],0,1)
            M_reduced = np.reshape(M_swapped,(aim,si*ai))
            (U,S,V) = np.linalg.svd(np.real(M_reduced),full_matrices=0)
            #(U,S,V) = np.linalg.svd(M_reduced,full_matrices=0)
            self.Ml[site] = np.swapaxes(np.reshape(V,(aim,si,ai)),0,1)
            self.Ml[site-1] = np.einsum('ijk,kl,l->ijl',self.Ml[site-1],U,S)
            if self.distribCheck:
                print('\t\t\tDistribution Check: {}'.format(np.sum(np.abs(np.einsum('ijk,ikl->ijl',M_prev,M_prev_p1)-np.einsum('ijk,ikl->ijl',self.Ml[site-1],self.Ml[site])))))
            if self.normalizationCheck:
                print('Check for normalization:\n{}'.format(np.einsum('ijk,ikl->jl',self.Ml[site],np.transpose(self.Ml[site],(0,2,1)))))
        si,aim,ai = self.Mr[site].shape
        if dir == 'right':
            M_prev = self.Mr[site]
            M_prev_p1 = self.Mr[site+1]
            M_reduced = np.reshape(self.Mr[site],(si*aim,ai))
            (U,S,V) = np.linalg.svd(np.real(M_reduced),full_matrices=0)
            #(U,S,V) = np.linalg.svd(M_reduced,full_matrices=0)
            self.Mr[site] = np.reshape(U,(si,aim,ai))
            self.Mr[site+1] = np.einsum('i,ij,kjl->kil',S,V,self.Mr[site+1])
            if self.distribCheck:
                print('\t\t\tDistribution Check: {}'.format(np.sum(np.abs(np.einsum('ijk,ikl->ijl',M_prev,M_prev_p1)-np.einsum('ijk,ikl->ijl',self.Mr[site],self.Mr[site+1])))))
            if self.normalizationCheck:
                print('\t\t\tNormalization Check:\n{}'.format(np.einsum('ijk,ikl->jl',np.transpose(self.Mr[site],(0,2,1)),self.Mr[site])))
        elif dir == 'left':
            M_prev = self.Mr[site-1]
            M_prev_p1 = self.Mr[site]
            M_swapped = np.swapaxes(self.Mr[site],0,1)
            M_reduced = np.reshape(M_swapped,(aim,si*ai))
            (U,S,V) = np.linalg.svd(np.real(M_reduced),full_matrices=0)
            #(U,S,V) = np.linalg.svd(M_reduced,full_matrices=0)
            self.Mr[site] = np.swapaxes(np.reshape(V,(aim,si,ai)),0,1)
            self.Mr[site-1] = np.einsum('ijk,kl,l->ijl',self.Mr[site-1],U,S)
            if self.distribCheck:
                print('\t\t\tDistribution Check: {}'.format(np.sum(np.abs(np.einsum('ijk,ikl->ijl',M_prev,M_prev_p1)-np.einsum('ijk,ikl->ijl',self.Mr[site-1],self.Mr[site])))))
            if self.normalizationCheck:
                print('Check for normalization:\n{}'.format(np.einsum('ijk,ikl->jl',self.Mr[site],np.transpose(self.Mr[site],(0,2,1)))))
        
    def update_f(self,site,dir):
        if dir == 'right':
            self.F[site+1] = np.einsum('ijkl,knm,nip,lpq->mjq',\
                                     self.mpo.W(site),np.conjugate(self.Ml[site]),self.F[site],self.Mr[site])
        elif dir == 'left':
            self.F[site] = np.einsum('ijkl,knm,mjq,lpq->nip',\
                                     self.mpo.W(site),np.conjugate(self.Ml[site]),self.F[site+1],self.Mr[site])

    def energy_contraction(self,psi_H_psi,psi_psi,i):
        psi_H_psi = np.einsum('ijk,jnlm,lio,mkp->onp',psi_H_psi,self.mpo.W(i),np.conj(self.Ml[i]),self.Mr[i])
        psi_psi = np.einsum('ij,kjl,kim->lm',psi_psi,np.conj(self.Ml[i]),self.Mr[i])
        if i < self.L-1:
            self.energy_contraction(psi_H_psi,psi_psi,i+1)
        else:
            psi_H_psi = np.einsum('iii->i',psi_H_psi)
            psi_psi = np.einsum('ii->i',psi_psi)
            self.energy_via_contraction = psi_H_psi/psi_psi
                                                                  
    def full_energy_calculation(self):
        psi_H_psi = np.array([[[1]]])
        psi_psi = np.array([[1]])
        self.energy_contraction(psi_H_psi,psi_psi,0)
        return(self.energy_via_contraction)
       
    def calc_observables(self,site):
        self.energy_calc = np.einsum('ijk,jlmn,olp,mio,nkp->',\
                              self.F[site],self.mpo.W(site),self.F[site+1],np.conjugate(self.Ml[site]),self.Mr[site])
        if self.checkEnergyNormalization:
            top_cntr = np.einsum('bac,dce,feg,hgi,jik,lkm->bdfhjl',\
                       self.Mr[0],self.Mr[1],self.Mr[2],self.Mr[3],self.Mr[4],\
                       self.Mr[5])#,self.M[6],self.M[7],self.M[8],self.M[9])
            bottom_cntr = np.einsum('bac,dce,feg,hgi,jik,lkm->bdfhjl',\
                       np.conj(self.Ml[0]),np.conj(self.Ml[1]),np.conj(self.Ml[2]),np.conj(self.Ml[3]),np.conj(self.Ml[4]),\
                       np.conj(self.Ml[5]))#,np.conj(self.M[6]),np.conj(self.M[7]),np.conj(self.M[8]),np.conj(self.M[9]))
            norm = np.einsum('bdfhjl,bdfhjl->',top_cntr,bottom_cntr)
            print('\t\tNormalization Factor= {}'.format(norm))
        self.calc_spin_x[site] = np.einsum('ijk,il,ljk->',self.Ml[site].conj(),self.mpo.S_x,self.Mr[site]).real
        self.calc_spin_y[site] = np.einsum('ijk,il,ljk->',self.Ml[site].conj(),self.mpo.S_y,self.Mr[site]).real
        self.calc_spin_z[site] = np.einsum('ijk,il,ljk->',self.Ml[site].conj(),self.mpo.S_z,self.Mr[site]).real
        self.calc_empty[site] = np.einsum('ijk,il,ljk->',self.Ml[site].conj(),self.mpo.v,self.Mr[site]).real
        self.calc_full[site] = np.einsum('ijk,il,ljk->',self.Ml[site].conj(),self.mpo.n,self.Mr[site]).real
        return(self.energy_calc)
    
    def plot_observables(self):
        if self.plotExpVal:
            plt.style.use('ggplot')
            plt.ion()
            if not self.exp_val_figure:
                self.exp_val_figure = plt.figure()
            else:
                plt.figure(self.exp_val_figure.number)
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
            elif self.mpo.ham_type is "tasep":
                plt.plot(np.linspace(0,self.L-1,num=self.L),self.calc_empty)
                plt.ylabel('Average Occupation',fontsize=20)
                plt.xlabel('Site',fontsize=20)
            plt.pause(0.01)

    def plot_convergence(self,site):
        if self.plotConv:
            plt.style.use('ggplot')
            plt.ion()
            if not self.conv_figure:
                self.conv_figure = plt.figure()
                self.site_vec = [site]
            else:
                plt.figure(self.conv_figure.number)
                self.site_vec.insert(-1,site)
            plt.cla()
            if len(self.energy_vec_all) > 4 and np.abs(self.energy_vec_all[1]-np.min(self.energy_vec_all)) > 1e-15:
                if self.mpo.ham_type is "heis":
                    plt.semilogy(self.site_vec[:-2],np.real(self.energy_vec_all[1:-2]-np.min(self.energy_vec_all)),'r-',linewidth=2)
                    plt.ylabel('E-E_{min}',fontsize=20)
                elif self.mpo.ham_type is "tasep":
                    plt.semilogy(self.site_vec[:-2],np.abs(np.real(self.energy_vec_all[1:-2]-np.max(self.energy_vec_all))),'r-',linewidth=2)
                    plt.ylabel('E_{max}-E',fontsize=20)
            else:
                plt.plot(self.site_vec,self.energy_vec_all[1:]-np.min(self.energy_vec_all),'r-',linewidth=2)
            plt.xlabel('Site',fontsize=20)
            plt.hold(False)
            plt.pause(0.01)

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
                    print('\t\tOptimized site {}: {} (contracted E)'.format(site,self.full_energy_calculation()))
                self.sweep_energy_vec.insert(len(self.sweep_energy_vec),self.energy_vec_all[-1])
                self.plot_convergence(site)
                self.plot_observables()
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
                    print('\t\tOptimized site {}: {} (contracted E)'.format(site,self.full_energy_calculation()))
                self.sweep_energy_vec.insert(len(self.sweep_energy_vec),self.energy_vec_all[-1])
                self.plot_convergence(site)
                self.plot_observables()
            self.energy_vec.insert(len(self.energy_vec),self.sweep_energy_vec[-1].real)
            print(len(self.energy_vec))
            if (np.abs(self.energy_vec[-1]-self.energy_vec[-2]) < self.tol) and (len(self.energy_vec) > 3):
                converged = True
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
    print('No Main Function Currently Implemented')
