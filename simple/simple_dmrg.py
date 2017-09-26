import numpy as np
import matplotlib.pyplot as plt
import time

class simpleHeisDMRG:
    
    def __init__(self, L=6, d=2, D=8, tol=1e-3, max_sweep_cnt=3, h=1, J=1):
        # Input Parameters
        self.L = L
        self.d = d
        self.D = D
        self.h = h
        self.J = J
        self.tol = tol
        self.max_sweep_cnt = max_sweep_cnt
        # MPO
        S_p = np.array([[0,1],
                        [0,0]])
        S_m = np.array([[0,0],
                        [1,0]])
        S_z = np.array([[0.5,0],
                        [0,-0.5]])
        zero_mat = np.zeros([2,2])
        I = np.eye(2)
        self.w_arr = np.array([[I,           zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [S_p,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [S_m,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [S_z,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [-self.h*S_z, self.J/2.*S_m, self.J/2.*S_p, self.J*S_z, I       ]])
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
    
    def W(self,site):
        if site == 0:
            return np.expand_dims(self.w_arr[-1,:],0)
        elif site == self.L-1:
            return np.expand_dims(self.w_arr[:,0],1)
        else:
            return self.w_arr
        
    def initialize_f(self):
        self.F = []
        self.F.insert(0,np.array([[[1]]]))
        for site in range(1,self.L)[::-1]:
                self.F.insert(0,np.einsum('ijk,lmio,opq,kmq->jlp',\
                                              np.conjugate(self.M[site]),self.W(site),self.M[site],self.F[0]))
        self.F.insert(0,np.array([[[1]]]))
    
    def h_optimization(self,site,dir):
        h = np.einsum('ijk,jlmn,olp->mionkp',self.F[site],self.W(site),self.F[site+1]) 
        si,aim,ai,sip,aimp,aip = h.shape
        h = np.reshape(h,(si*aim*ai,sip*aimp*aip))
        w,v = np.linalg.eig(h) 
        e = w[(w).argsort()][0]
        v = v[:,(w).argsort()]
        self.M[site] = np.reshape(v[:,0],(si,aim,ai))
        return e
    
    def normalize(self,site,dir):
        si,aim,ai = self.M[site].shape
        if dir == 'right':
            M_reduced = np.reshape(self.M[site],(si*aim,ai))
            (U,S,V) = np.linalg.svd(M_reduced,full_matrices=0)
            self.M[site] = np.reshape(U,(si,aim,ai))
            self.M[site+1] = np.einsum('i,ij,kjl->kil',S,V,self.M[site+1])
        elif dir == 'left':
            M_swapped = np.swapaxes(self.M[site],0,1)
            M_reduced = np.reshape(M_swapped,(aim,si*ai))
            (U,S,V) = np.linalg.svd(M_reduced,full_matrices=0)
            self.M[site] = np.swapaxes(np.reshape(V,(aim,si,ai)),0,1)
            self.M[site-1] = np.einsum('ijk,kl,l->ijl',self.M[site-1],U,S)
        
    def update_f(self,site,dir):
        if dir == 'right':
            self.F[site+1] = np.einsum('ijkl,knm,nip,lpq->mjq',\
                                     self.W(site),np.conjugate(self.M[site]),self.F[site],self.M[site])
        elif dir == 'left':
            self.F[site] = np.einsum('ijkl,knm,mjq,lpq->nip',\
                                     self.W(site),np.conjugate(self.M[site]),self.F[site+1],self.M[site])
       
    def calc_observables(self,site):
        self.energy_calc = np.einsum('ijk,jlmn,olp,mio,nkp->',\
                              self.F[site],self.W(site),self.F[site+1],np.conjugate(self.M[site]),self.M[site])
        return(self.energy_calc)
    
    def run_optimization(self):
        converged = False
        sweep_cnt = 0
        energy_vec = [0]
        energy_vec_all = [0]
        while not converged:
            print('Beginning Sweep Set {}'.format(sweep_cnt))
            print('\tBeginning Right Sweep')
            for site in range(self.L-1):
                energy_vec_all.insert(len(energy_vec_all),self.h_optimization(site,'right'))
                self.normalize(site,'right')
                self.update_f(site,'right')
                print('\t\tOptimized site {}: {}'.format(site,energy_vec_all[-1]))
            print('\tBeginning Left Sweep')
            for site in range(1,self.L)[::-1]:
                energy_vec_all.insert(len(energy_vec_all),self.h_optimization(site,'right'))
                self.normalize(site,'left')
                self.update_f(site,'left')
                print('\t\tOptimized site {}: {}'.format(site,energy_vec_all[-1]))
            energy_vec.insert(len(energy_vec),energy_vec_all[-1])
            if np.abs(energy_vec[-1]-energy_vec[-2]) < self.tol:
                converged = True
                print(('#'*75+'\nGround state energy: {}\n'+'#'*75).format(energy_vec[-1]))
            elif sweep_cnt > self.max_sweep_cnt:
                converged = True
                print('Total number of iterations exceeded limit')
                print('Resulting calculated energy: {}'.format(energy_vec[-1]))
            else:
                sweep_cnt += 1
    
    def calc_ground_state(self):
        self.initialize_f()
        self.run_optimization()

if __name__ == "__main__":
    t0 = time.time()
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    x = simpleHeisDMRG(L=50)
    x.calc_ground_state()
    t1 = time.time()
    print(('#'*75+'\nTotal Time: {}\n'+'#'*75).format(t1-t0))
