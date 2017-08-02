import numpy as np

class HeisMPS:
    """
    Description:
        An object containing all information pertaining to the matrix product state
        associated with the heisenberg spin model of length L. 
        
    Class Members:
        > self.L               - The number of sites for the system
        > self.d               - The local state-space dimension
        > self.D               - The cut-off for the state-space dimension of each 
                                 matrix product state. This limits the size of the 
                                 calculations using the Schmidt Decomposition.
        > self.reshape_order   - The ordering for reshaping of matrices, should always
                                 be set at "F", indicating Fortran ordering.
        > self.M               - A list containing the total MPS. At each site is an
                                 accompanying numpy array.
    
    Key Functions:
        1) create_initial_guess() - Generates the initial MPS such that it is in 
                                    'right-canonical' form. Allows for generation 
                                    of random matrices ('rand'), matrices such that the 
                                    initial guess for psi is a zero matrix with the 
                                    first element set to one ('hf'), or a matrix such 
                                    that the initial guess is the ground state as 
                                    estimated by an alternative calculation ('gs').
        2) initialize_r()         - A function that calculates all of the R-expressions
                                    associated with the initial MPS. Performed according to 
                                    Equations 40-43 of the accompanying text.
        2) update_lr()            - A function that calculates the L- and R-expressions
                                    for a given site such. Performed according to Equations
                                    40-43 of the accompanying text.
    """
    def __init__(self,L,init_guess_type,d,reshape_order,D):
        self.L = L
        self.init_guess_type = init_guess_type
        self.d = d
        self.D = D
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
            max_ind = min([a_curr,self.D])
            if a_curr > a_prev:
                v = v[:max_ind,:]
                v = np.reshape(v,(max_ind,self.d,-1),order=self.reshape_order)
                v = np.swapaxes(v,0,1)
                B.insert(0,v)
            else:
                v = v[:max_ind,:]
                v = np.reshape(v,(-1,self.d,max_ind),order=self.reshape_order)
                v = np.swapaxes(v,0,1)
                B.insert(0,v)
            s = s[:max_ind]
            u = u[:,:max_ind]
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