import numpy as np
from scipy.linalg import eig

class DMRG_CALC:
    def __init__(self, L=10, D=8, d=2, num_sweeps=3, conv_tol=1e-3, ham_type='sep', ham_params=(0.35,0,2/3)):
        self.L = L
        self.D = D
        self.d = d
        self.num_sweeps = num_sweeps
        self.conv_tol = conv_tol
        self.ham_type = ham_type
        self.ham_params = ham_params

    def generate_mpo(self):
        self.S_p = np.array([[0,1],
                             [0,0]])
        self.S_m = np.array([[0,0],
                             [1,0]])
        self.S_z = np.array([[0.5,0],
                             [0,-0.5]])
        self.S_x = np.array([[0,0.5],
                             [0.5,0]])
        self.S_y = 1/(2j)*np.array([[0,1],
                                    [-1,0]])
        self.n = np.array([[0,0],
                           [0,1]])
        self.v = np.array([[1,0],
                           [0,0]])
        self.I = np.eye(2)
        self.zero_mat = np.zeros([2,2])
        if self.ham_type is 'sep':
            self.alpha = self.ham_params[0]
            self.s = self.ham_params[1]
            self.beta = self.ham_params[2]
            self.w_arr = np.array([[self.I,         self.zero_mat,              self.zero_mat,  self.zero_mat],
                                   [self.S_m,       self.zero_mat,              self.zero_mat,  self.zero_mat],
                                   [self.v,         self.zero_mat,              self.zero_mat,  self.zero_mat],
                                   [self.zero_mat,  np.exp(-self.s)*self.S_p,   -self.n,        self.I       ]])
        elif self.ham_type is 'heis':
            self.J = self.ham_params[0]
            self.h = self.ham_params[1]
            self.w_arr = np.array([[self.I,           self.zero_mat,      self.zero_mat,      self.zero_mat,   self.zero_mat],
                                   [self.S_p,         self.zero_mat,      self.zero_mat,      self.zero_mat,   self.zero_mat],
                                   [self.S_m,         self.zero_mat,      self.zero_mat,      self.zero_mat,   self.zero_mat],
                                   [self.S_z,         self.zero_mat,      self.zero_mat,      self.zero_mat,   self.zero_mat],
                                   [-self.h*self.S_z, self.J/2.*self.S_m, self.J/2.*self.S_p, self.J*self.S_z, self.I       ]])

    def W(self,site):
        if self.ham_type is 'sep':
            if site == 0:
                tmp_arr = np.expand_dims(self.w_arr[-1,:],0)
                tmp_arr[0,0,:,:] = self.alpha*(np.exp(-self.s)*self.S_m-self.v)
                return tmp_arr
            if site == self.L-1:
                tmp_arr = np.expand_dims(self.w_arr[:,0],1)
                tmp_arr[-1,0,:,:] = self.beta*(np.exp(-self.s)*self.S_p-self.n)
                return tmp_arr
            else:
                return self.w_arr
        elif self.ham_type is 'heis':
            if site == 0:
                return np.expand_dims(self.w_arr[-1,:],0)
            elif site == self.L-1:
                return np.expand_dims(self.w_arr[:,0],1)
            else:
                return self.w_arr

    def generate_rand_mps(self):
        # Create list to hold top and bottom MPSs
        self.M_top = []
        self.M_bottom = []
        # Create correctly sized random MPSs
        a_prev = 1
        going_up = True
        for i in range(self.L):
            if going_up:
                a_curr = min(self.d**(i+1),self.d**(self.L-(i)))
                if a_curr <= a_prev:
                    going_up = False
                    a_curr = self.d**(self.L-(i+1))
                    a_prev = self.d**(self.L-(i))
            else:
                a_curr = self.d**(self.L-(i+1))
                a_prev = self.d**(self.L-(i))
            max_ind_curr = min([a_curr,self.D])
            max_ind_prev = min([a_prev,self.D])
            if going_up:
                #newMat = np.random.rand(self.d,max_ind_curr,max_ind_prev)
                newMat = np.ones((self.d,max_ind_curr,max_ind_prev))
            else:
                #newMat = np.random.rand(self.d,max_ind_curr,max_ind_prev)
                newMat = np.ones((self.d,max_ind_curr,max_ind_prev))
            self.M_top.insert(0,np.copy(newMat))
            self.M_bottom.insert(0,np.copy(newMat))
            a_prev = a_curr
        # Left Normalize the MPS
        for i in range(1,len(self.M_top))[::-1]:
            self.normalize(i,'left')

    def normalize(self,site,direction):
        si,aim,ai = self.M_top[site].shape
        if direction == "left":
            M_top_copy = self.M_top###
            M_bottom_copy = self.M_bottom###
            print(M_top_copy[site])
            M_swapped = np.swapaxes(self.M_top[site],0,1)
            M_reduced = np.reshape(M_swapped,(aim,si*ai))
            (U,S,V) = np.linalg.svd(M_reduced,full_matrices=0)
            V_reshaped = np.swapaxes(np.reshape(V,(aim,si,ai)),0,1)
            V_inv = np.zeros((si,aim,ai))
            for i in range(si):
                V_inv[i,:,:] = np.linalg.pinv(V_reshaped[i,:,:]).transpose()
            self.M_top[site-1] = np.einsum('ijk,kl,l->ijl',self.M_top[site-1],U,S) ### ORDER COULD BE CHANGED
            self.M_bottom[site-1] = np.einsum('ijk,ikl,iml->ijm',self.M_bottom[site-1],self.M_bottom[site],V_inv) ### ORDER COULD BE CHANGED
            self.M_top[site] = np.copy(V_reshaped)
            self.M_bottom[site] = np.copy(V_reshaped)
            print(M_top_copy[site])
            print('\t\t\tTop Distribution Check: {}'.format(np.sum(np.abs(np.einsum('ijk,ikl->ijl',M_top_copy[site-1],M_top_copy[site])-\
                                                                          np.einsum('ijk,ikl->ijl',self.M_top[site-1],self.M_top[site])))))
            print('\t\t\tBottom Distribution Check: {}'.format(np.sum(np.abs(np.einsum('ijk,ikl->ijl',M_bottom_copy[site-1],M_bottom_copy[site])-\
                                                                             np.einsum('ijk,ikl->ijl',self.M_bottom[site-1],self.M_bottom[site])))))
            print('\t\t\tTop normalization check:\n{}'.format(np.einsum('ijk,ikl->jl',self.M_top[site],np.transpose(self.M_top[site],(0,2,1)))))
            print('\t\t\tBottom normalization check:\n{}'.format(np.einsum('ijk,ikl->jl',self.M_bottom[site],np.transpose(self.M_bottom[site],(0,2,1)))))
        elif direction == "right":
            M_top_copy = self.M_top
            M_bottom_copy = self.M_bottom
            M_reduced = np.reshape(self.M_top[site],(si*aim,ai))
            (U,S,V) = np.linalg.svd(M_reduced,full_matrices=0)
            U_reshaped = np.reshape(U,(si,aim,ai))
            U_inv = np.zeros((si,aim,ai))
            for i in range(si):
                U_inv[i,:,:] = np.linalg.pinv(U_reshaped[i,:,:]).transpose()
            self.M_top[site+1] = np.einsum('i,ij,kjl->kil',S,V,self.M_top[site+1]) ### ORDER COULD BE CHANGED
            self.M_bottom[site+1] = np.einsum('ikj,ikl,ilm->ijm',U_inv,self.M_bottom[site],self.M_bottom[site+1]) # ORDER COULD BE CHANGED
            self.M_top[site] = np.copy(U_reshaped)
            self.M_bottom[site] = np.copy(U_reshaped)
            print('\t\t\tTop Distribution Check: {}'.format(np.sum(np.abs(np.einsum('ijk,ikl->ijl',M_top_copy[site],M_top_copy[site+1])-\
                                                                          np.einsum('ijk,ikl->ijl',self.M_top[site],self.M_top[site+1])))))
            print('\t\t\tBottom Distribution Check: {}'.format(np.sum(np.abs(np.einsum('ijk,ikl->ijl',M_bottom_copy[site],M_bottom_copy[site+1])-\
                                                                          np.einsum('ijk,ikl->ijl',self.M_bottom[site],self.M_bottom[site+1])))))
            print('\t\t\tTop Normalization Check:\n{}'.format(np.einsum('ijk,ikl->jl',np.transpose(self.M_top[site],(0,2,1)),self.M_top[site])))
            print('\t\t\tBottom Normalization Check:\n{}'.format(np.einsum('ijk,ikl->jl',np.transpose(self.M_bottom[site],(0,2,1)),self.M_bottom[site])))

    def initialize_f(self):
        self.F = []
        self.F.insert(0,np.array([[[1]]]))
        for site in range(1,self.L)[::-1]:
            self.F.insert(0,np.einsum('ijk,lmio,opq,kmq->jlp',\
                          np.conjugate(self.M_bottom[site]),self.W(site),self.M_top[site],self.F[0]))
        self.F.insert(0,np.array([[[1]]])) # DOES THIS NEED TO BE A SUMMATION AS WELL???

    def update_f(self,direction):
        if direction is 'right':
            self.F[site+1] = np.einsum('ijkl,knm,nip,lpq->mjq',\
                                       self.W(site),np.conjugate(self.M_bottom[site]),self.F[site],self.M_top[site])
        elif direction is 'left':
            self.F[site] = np.einsum('ijkl,knm,mjq,lpq->nip',\
                                     self.W(site),np.conjugate(self.M_bottom[site]),self.F[site+1],self.M_top[site])

    def optimize_site(self,site):
        H = np.einsum('ijk,jlmn,olp->mionkp',self.F[site],self.W(site),self.F[site+1])
        si,aim,ai,sip,aimp,aip = H.shape
        H = np.reshape(H,(si*aim*ai,sip*aimp*aip))
        w,v_top,v_bottom = eig(H,left=True,right=True)
        e = w[w.argsort()[0]]
        v_top = v_top[:,w.argsort()[0]]
        v_bottom = v_bottom[:,w.argsort()[0]]
        self.M_top[site] = np.reshape(v_top,(si,aim,ai))
        self.M_bottom[site] = np.reshape(v_bottom,(si,aim,ai))
        return e

    def energy_contraction(self,psi_H_psi=np.array([[[1]]]),
                           psi_psi=np.array([[1]]),i=0):
        psi_H_psi = np.einsum('ijk,jnlm,lio,mkp->onp',psi_H_psi,self.W(i),np.conj(self.M_bottom[i]),self.M_top[i])
        psi_psi = np.einsum('ij,kjl,kim->lm',psi_psi,np.conj(self.M_bottom[i]),self.M_top[i])
        if i < self.L-1:
            return self.energy_contraction(psi_H_psi,psi_psi,i+1)
        else:
            psi_H_psi = np.einsum('iii->i',psi_H_psi)
            psi_psi = np.einsum('ii->i',psi_psi)
            return psi_H_psi/psi_psi

    def dmrg_optimization(self):
        converged = False
        self.energy_vec = [self.energy_contraction()]
        sweep_cnt = 0
        print('Beginning Optimization')
        while not converged:
            print('\tBeginning Sweep {}'.format(sweep_cnt))
            for site in range(self.L-1):
                energy_optimized = self.optimize_site(site)
                energy_calc = self.energy_contraction()
                self.normalize(site,'right')
                self.update_f(site)
                print('\t\tEnergy at Site {} = {}'.format(site,energy_calc))
                print('\t\t\tContracted Energy = {}'.format(energy_optimized))
            for site in range(1,self.L)[::-1]:
                energy_optimized = self.optimize_site(site)
                energy_calc = self.energy_contraction()
                self.normalize(site,'left')
                self.update_f(site)
                print('\t\tEnergy at Site {} = {}'.format(site,energy_calc))
                print('\t\t\tContracted Energy = {}'.format(energy_optimized))
            self.energy_vec.insert(0,energy_optimized)
            if np.abs(self.energy_vec[0]-self.energy_vec[1]) < self.conv_tol:
                converged = True
                print('#'*50+'\nConverged at E = {}'.format(self.energy_vec[0])+'\n'+'#'*50)
            elif sweep_cnt >= self.num_sweeps:
                converged = True
                print('!'*50+'\nConvergence Not Acheived\nFinal E = {}'.format(self.energy_vec[0])+'\n'+'!'*50)
            else:
                sweep_cnt += 1

    def run_calc(self):
        self.generate_mpo()
        self.generate_rand_mps()
        self.initialize_f()
        self.dmrg_optimization()

if __name__ == "__main__":
    obj = DMRG_CALC(L=8,ham_type='heis',ham_params=(1,0))
    obj.run_calc()
