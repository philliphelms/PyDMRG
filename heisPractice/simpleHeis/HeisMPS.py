import numpy as np

class HeisMPS:

    def __init__(self,L,init_guess_type,d,reshape_order):
        self.L = L
        self.init_guess_type = init_guess_type
        self.d = d
        self.reshape_order = reshape_order
        self.M = None
    
    def create_initial_guess(self):
        print('Generating Initial MPS Guess')
        L = self.L
        for i in range(L):
            if i == 0:
                if self.init_guess_type is 'rand':
                    psi = np.random.rand(self.d**(L-1),self.d)
                elif self.init_guess_type is 'hf':
                    psi = np.zeros([self.d**(L-1),self.d])
                    psi[0,0] = 1
                elif self.init_guess_type is 'gs':
                    if self.L == 4:
                        psi = np.array([[ -2.33807217e-16,  -3.13227746e-15,  -2.95241364e-15,   1.49429245e-01],
                                        [ -2.64596902e-15,   4.08248290e-01,  -5.57677536e-01,   7.68051068e-16],
                                        [  4.35097968e-17,  -5.57677536e-01,   4.08248290e-01,   1.28519114e-15],
                                        [  1.49429245e-01,   6.39650363e-16,   9.36163055e-17,  -2.17952587e-16]])
                        psi = psi.reshape(self.d**(L-1),self.d)
                    elif self.L == 2:
                        psi = np.array([[ -5.90750001e-17,  -7.07106781e-01],
                                        [  7.07106781e-01,  -6.55904148e-17]])
                        psi = np.reshape(psi,(self.d**(L-1),self.d),order=self.reshape_order)
                    else:
                        raise ValueError('Ground State initial guess is not available for more than four sites')
                elif self.init_guess_type is 'eye':
                    psi = np.eye(self.L)
                    psi = np.reshape(psi,(self.d**(L-1),self.d),order=self.reshape_order)
                else:
                    raise ValueError('Indicated initial guess type is not available')
                B = []
                a_prev = 1
            else:
                psi = np.einsum('ij,j->ij',u,s)
                nx,ny = u.shape
                psi = np.reshape(psi,(int(nx/self.d),int(ny*self.d)),order=self.reshape_order) # PROBLEM???
                a_prev = a_curr
            (u,s,v) = np.linalg.svd(psi,full_matrices=0)
            a_curr = min(self.d**(i+1),self.d**(L-(i)))
            if a_curr > a_prev:
                v = np.swapaxes(np.reshape(v,(a_curr,self.d,-1),order=self.reshape_order),0,1)
                B.insert(0,v)
            else:
                v = np.swapaxes(np.reshape(v,(-1,self.d,a_curr),order=self.reshape_order),0,1)
                B.insert(0,v)
        self.M = B
        
    def initialize_r(self,W):
        self.R_array = []
        self.L_array = []
        self.R_array.insert(0,np.array([[[1]]])) 
        self.L_array.insert(0,np.array([[[1]]])) 
        for out_cnt in range(self.L)[::-1]:
            if out_cnt == 0: 
                tmp_array = np.array([[[1]]])
            else:
                tmp_array = np.einsum('ijk,lmin,nop,kmp->jlo',\
                                      np.conjugate(self.M[out_cnt]),W(out_cnt),self.M[out_cnt],self.R_array[0])
            self.R_array.insert(0,tmp_array)
            
    def update_lr(self,site,swp_dir,W):
        if swp_dir == 'right': 
            # We update the L expressions
            tmp_array = np.einsum('ijk,lmin,nop,jlo->kmp',\
                                  np.conjugate(self.M[site]),W(site),self.M[site],self.L_array[site])
            if len(self.L_array) <= site+1:
                self.L_array.insert(len(self.L_array),tmp_array)
            else:
                self.L_array[site+1] = tmp_array
        elif swp_dir == 'left':
            # We update the R expressions
            self.R_array[site] = np.einsum('ijk,lmin,nop,kmp->jlo',\
                                           np.conjugate(self.M[site]),W(site),self.M[site],self.R_array[site+1])