import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

class HeisDMRG:

    def __init__(self,mpo,mps,D,tol,max_sweep_cnt,reshape_order):
        self.mpo = mpo
        self.mps = mps
        self.D = D
        self.tol = tol
        self.max_sweep_cnt = max_sweep_cnt
        self.reshape_order = reshape_order
        
    def H_opt(self,site):
        H = np.einsum('ijk,jlmn,olp->mionkp',self.mps.L_array[site],self.mpo.W(site),self.mps.R_array[site+1])
        sl,alm,al,slp,almp,alp = H.shape
        H = np.reshape(H,(sl*alm*al,sl*alm*al),order=self.reshape_order)
        w,v = np.linalg.eig(H)
        w = np.sort(w)
        v = v[:,w.argsort()]
        self.mps.M[site] = np.reshape(v[:,0],(sl,alm,al),order=self.reshape_order)
        energy = w[0]
        return energy
    
    def normalize(self,site,direction):
        si,aim,ai = self.mps.M[site].shape
        if direction == 'right':
            M_2d = np.reshape(self.mps.M[site],(si*aim,ai),order=self.reshape_order)
            (U,S,V) = np.linalg.svd(M_2d,full_matrices=0)
            self.mps.M[site] = np.reshape(U,(si,aim,ai),order=self.reshape_order)  
            self.mps.M[site+1] = np.einsum('i,ij,kjl->kil',S,V,self.mps.M[site+1])
        elif direction == 'left':
            M_3d_swapped = np.swapaxes(self.mps.M[site],0,1)
            M_2d = np.reshape(M_3d_swapped,(aim,si*ai),order=self.reshape_order)
            (U,S,V) = np.linalg.svd(M_2d,full_matrices=0)
            self.mps.M[site] = np.swapaxes(np.reshape(V,(aim,si,ai),order=self.reshape_order),0,1)
            # self.mps.M[site-1] = np.einsum('ijk,kl,l->ijl',self.mps.M[site-1],U,S)
            self.mps.M[site-1] = np.einsum('ijk,lk,k->ijl',self.mps.M[site-1],U,S)
            
    
    def run_optimization(self):
        converged = False
        sweep_cnt = 0
        energy_prev = 0
        while not converged:
            print('Beginning Sweep Set {}'.format(sweep_cnt))
            print('\tRight Sweep')
            for site in range(self.mps.L-1):
                energy_curr = self.H_opt(site)
                self.normalize(site,'right')
                self.mps.update_lr(site,'right',self.mpo.W)
                print('\t\tCompleted Site {}: {}'.format(site,energy_curr))
            print('\tLeft Sweep')
            for site in range(self.mps.L-1,0,-1):
                energy_curr = self.H_opt(site)
                self.normalize(site,'left')
                self.mps.update_lr(site,'left',self.mpo.W)
                print('\t\tCompleted Site {}: {}'.format(site,energy_curr))
            
            
            
            # Check for convergence
            if np.abs(energy_prev-energy_curr) < self.tol:
                converged = True
                print('#'*68)
                print('System has converged at:')
                print('E = {}'.format(energy_curr))
                print('#'*68)
            elif sweep_cnt >= self.max_sweep_cnt-1:
                converged = True
                print('Maximum number of sweeps exceeded - system not converged')
            else:
                print('\tResulting Energy: {}'.format(energy_curr))
                energy_prev = energy_curr
                sweep_cnt += 1