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
                                    right-canonical form. If init_guess_type is set
                                    as 'default', then this will generate a random MPS
                                    then put it into right-canonical form.
        2) create_initial_guess_special() - If a special initial guess such as the previously
                                    calculated ground state or the hartree fock state is 
                                    desired, create_initial_guess() will pass to this function.
        2) normalize(site,dir)    - A function that uses singular value decomposition to put 
                                    the resulting MPS in the correct mixed-canonical form. This
                                    is performed at the given site and done according to whether
                                    the sweep direction is 'left' or 'right'. Done according to 
                                    Equations 37-39 of the accompanying description.
        3) initialize_r()         - A function that calculates all of the R-expressions
                                    associated with the initial MPS. Performed according to 
                                    Equations 40-43 of the accompanying text.
        4) update_lr()            - A function that calculates the L- and R-expressions
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
        if self.init_guess_type is 'default':
            # Create random MPS
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
                    # newMat = np.zeros([self.d,max_ind_curr,max_ind_prev])
                    newMat = np.random.rand(self.d,max_ind_curr,max_ind_prev)
                else:
                    # newMat = np.zeros([self.d,max_ind_curr,max_ind_prev])
                    newMat = np.random.rand(self.d,max_ind_curr,max_ind_prev)
                # newMat[0,0,0] = 1
                self.M.insert(0,newMat)
                a_prev = a_curr
            # Normalize the MPS
            for i in range(1,len(self.M))[::-1]:
                self.normalize(i,'left')
        else: 
            create_initial_guess_special(self)
        
    def create_initial_guess_special(self):
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
        
    def normalize(self,site,direction):
        si,aim,ai = self.M[site].shape
        if direction == 'right':
            M_3d_swapped = np.swapaxes(self.M[site],0,1)
            M_2d = np.reshape(M_3d_swapped,(aim*si,ai),order=self.reshape_order)
            (U,S,V) = np.linalg.svd(M_2d,full_matrices=0)
            # prevProd = np.dot(self.M[site],self.M[site+1])
            self.M[site] = np.swapaxes(np.reshape(U,(aim,si,ai),order=self.reshape_order),0,1)
            self.M[site+1] = np.einsum('i,ij,kjl->kil',S,V,self.M[site+1])
            # newProd = np.dot(self.M[site],self.M[site+1])
            # print('Right - Difference between initial and final products: {:f}'.format(np.sum(prevProd-newProd,axis=(0,1,2,3))))
            # print('Check for normalization:\n{}'.format(np.einsum('ijk,ikl->jl',np.transpose(self.M[site],(0,2,1)),self.M[site])))
        elif direction == 'left':
            M_3d_swapped = np.swapaxes(self.M[site],0,1)
            M_2d = np.reshape(M_3d_swapped,(aim,si*ai),order=self.reshape_order)
            (U,S,V) = np.linalg.svd(M_2d,full_matrices=0)
            # prevProd = np.dot(self.M[site-1],self.M[site])
            self.M[site] = np.swapaxes(np.reshape(V,(aim,si,ai),order=self.reshape_order),0,1)
            self.M[site-1] = np.einsum('ijk,kl,l->ijl',self.M[site-1],U,S)
            # newProd = np.dot(self.M[site-1],self.M[site])
            # print('Left - Difference between initial and final products: {:f}'.format(np.sum(prevProd-newProd,axis=(0,1,2,3))))
            # print('Check for normalization:\n{}'.format(np.einsum('ijk,ikl->jl',self.M[site],np.transpose(self.M[site],(0,2,1)))))
    
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

    def calc_c(self,sigma_vec):
        c = np.array([[1]])
        for i in range(len(self.M)):
            c = np.dot(c,self.M[i][sigma_vec[i],:,:])
        return c

    def write_all_c(self,name):
        import xlsxwriter
        workbook = xlsxwriter.Workbook(name)
        worksheet = workbook.add_worksheet()
        worksheet.write(0,0,"C")
        worksheet.write(0,1,"C^2")
        for i in range(self.L):
            worksheet.write(0,i+2,"Site {}".format(i))
        for i in range(2 ** self.L):
            bin_str = '{0:b}'.format(i).rjust(self.L, '0')
            sigma_vec = np.fromstring(bin_str,'u1') - ord('0')
            c_val = self.calc_c(sigma_vec)
            col = 0
            worksheet.write(i+1,col,c_val[0,0])
            col += 1
            worksheet.write(i+1,col,(c_val[0,0]**2))
            col += 1
            for j in range(len(sigma_vec)):
                worksheet.write(i+1,col,sigma_vec[j])
                col += 1
        workbook.close()
