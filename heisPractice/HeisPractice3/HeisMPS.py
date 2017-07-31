import numpy as np

class HeisMPS:
    # Class that constructs and returns a matrix product states object for the 
    # Heisenberg model.
    #
    # Key Functions:
    #
    #   1! - create_initial_guess - creates the initial guess matrices (correctly
    #       normalized) for the system.
    #   2! - initialize_r - generates the r-expressions iteratively, to be used 
    #       as part of the initialization of the DMRG algorithm
    #   3 - update_lr - 
    #   4 - calc_energy - 
    #
    # Other Functions:
    #
    #   1 - print_all_m - prints all matrices associated with the MPS
    #   2 - print_all_multiplied_B - 
    #   3 - print_R - 
    #   4 - print_L - 
    def __init__(self,L,init_guess_type,d,verbose):
        self.L = L
        self.init_guess_type = init_guess_type
        self.d = d
        self.verbose = verbose
        self.M = None
                
    def print_all_m(self,ntabs):
        # Method to print all of the matrices for the MPS 
        # 
        # Arguments:
        #   ntabs - indicates the number of additional tabs to be placed before each line
        for i in range(self.L):
            print(('\t'*ntabs+'\tSite {}').format(i))
            for j in range(self.d):
                print(('\t'*ntabs+'\t\tOccupation {}:').format(j))
                ns,_ = self.M[i][j,:,:].shape
                for k in range(ns):
                    print(('\t'*ntabs+'\t\t\t{}').format(self.M[i][j,k,:]))
        
    def print_all_multiplied_B(self,ntabs):
        if self.d == 2:
            for i in range(len(self.M)):
                print(('\t'*ntabs+'\tSite {}').format(i))
                result = np.dot(self.M[i][0,:,:],np.transpose(self.M[i][0,:,:]))+np.dot(self.M[i][1,:,:],np.transpose(self.M[i][1,:,:]))
                nx,_ = result.shape
                for j in range(nx):
                    print(('\t'*ntabs+'\t\t{}').format(result[j,:]))
        else:
            print('\t'*ntabs+'\tChecking for identity matrices not possible for d != 2')
    
    def create_initial_guess(self):
        # Function to create a Right-Canonical MPS as the initial guess
        # Options are controlled by setting parameters in the Heis_MPS_MPO object
        # Follows the procedure and notation carried out in section 4.1.3.ii of Schollwock (2011)
        if self.verbose:
            print('#'*68)
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
                        psi = psi.reshape(self.d**(L-1),self.d)
                    else:
                        raise ValueError('Ground State initial guess is not available for more than four sites')
                else:
                    raise ValueError('Indicated initial guess type is not available')
                B = []
                a_prev = 1
            else:
                psi = np.einsum('ij,j->ij',u,s)
                nx,ny = u.shape
                psi = psi.reshape(nx/self.d,ny*self.d)
                # psi = np.dot(u,np.diag(s)).reshape(self.d**(L-(i+1)),-1)
                a_prev = a_curr
            (u,s,v) = np.linalg.svd(psi,full_matrices=0)
            a_curr = min(self.d**(i+1),self.d**(L-(i)))
            # v = np.transpose(v)
            if a_curr > a_prev:
                v = np.swapaxes(v.reshape(a_curr,self.d,-1),0,1)
                B.insert(0,v)
            else:
                v = np.swapaxes(v.reshape(-1,self.d,a_curr),0,1)
                B.insert(0,v)
        self.M = B
        if self.verbose:
            print('\t'*1+'-'*60)
            print('\t'*1+'Resulting MPS:')
            self.print_all_m(1)
            print('\t'*1+'Check for Right-Normalization:')
            self.print_all_multiplied_B(1)
        
    def print_R(self,ntabs):
        # Method to print all of the matrices for the MPS 
        # 
        # Arguments:
        #   ntabs - indicates the number of additional tabs to be placed before each line
        for i in range(len(self.R_array)):
            print(('\t'*ntabs+'\tSite {}').format(i-1))
            n1,n2,n3 = self.R_array[i].shape
            for j in range(n1):
                print(('\t'*ntabs+'\t\t[{},:,:]').format(j))
                for k in range(n2):
                    print(('\t'*ntabs+'\t\t\t{}').format(self.R_array[i][j,k,:]))
                    
    def print_L(self,ntabs):
        # Method to print all of the matrices for the MPS 
        # 
        # Arguments:
        #   ntabs - indicates the number of additional tabs to be placed before each line
        for i in range(len(self.L_array)):
            print(('\t'*ntabs+'\tSite {}').format(i-1))
            n1,n2,n3 = self.L_array[i].shape
            for j in range(n1):
                print(('\t'*ntabs+'\t\t[{},:,:]').format(j))
                for k in range(n2):
                    print(('\t'*ntabs+'\t\t\t{}').format(self.L_array[i][j,k,:]))
        
    def initialize_r(self,W):
        # Calculate all R-expressions iteratively for sites L-1 through 1
        # Follows the procedure and notation outlined in Equation 197 of Section 6.2 of Schollwock (2011)
        self.R_array = []
        self.L_array = []
        # Insert R[L] dummy array
        self.R_array.insert(0,np.array([[[1]]])) 
        self.L_array.insert(0,np.array([[[1]]])) 
        for out_cnt in range(self.L)[::-1]:
            if out_cnt == 0: 
                tmp_array = np.array([[[1]]])
            else:
                # From Eqn 43 of my report
                tmp_array = np.einsum('ijk,lmin,nop,kmp->jlo',\
                                      np.conjugate(self.M[out_cnt]),W(out_cnt),self.M[out_cnt],self.R_array[0])
            self.R_array.insert(0,tmp_array)
        if self.verbose > 1:
            print('\t'*1+'-'*60)
            print('\t'*1+'Resulting R:')
            self.print_R(1)
            
    def update_lr(self,site,swp_dir,W):
        if swp_dir == 'right': 
            # We update the L expressions
            if True:
                tmp_array = np.einsum('ijk,lmin,nop,jlo->kmp',\
                                  np.conjugate(self.M[site]),W(site),self.M[site],self.L_array[site])
            else:
                LA = np.einsum('ijk,lkm->ijlm',self.L_array[site],self.M[site])
                WLA = np.einsum('jnol,ijlm->imno',W(site),LA)
                tmp_array = np.einsum('oip,imno->pnm',self.M[site],WLA)
            print(np.abs(tmp_array-tmp_array2))
            if len(self.L_array) <= site+1:
                self.L_array.insert(len(self.L_array),tmp_array)
            else:
                self.L_array[site+1] = tmp_array
            if self.verbose > 2:
                print('\t'*3+'Updated L:')
                self.print_L(3)
        elif swp_dir == 'left':
            # We update the R expressions
            self.R_array[site] = np.einsum('ijk,lmin,nop,kmp->jlo',\
                                           np.conjugate(self.M[site]),W(site),self.M[site],self.R_array[site+1])
            if self.verbose > 2:
                print('\t'*3+'Updated R:')
                self.print_R(3)
        else:
            raise ValueError('Sweep Direction must be left or right')
            
    def calc_energy(self,site,W):
        # Calculates the energy of a given state using the hamilonian operators
        # Done according section 6 of Schollwock (2011)
        numerator = np.einsum('ijk,jlmn,olp,mio,nkp->',\
                              self.L_array[site],W(site),self.R_array[site+1],np.conjugate(self.M[site]),self.M[site])
        si,aim,ai = self.M[site].shape
        psi_a = np.eye(aim)
        psi_b = np.eye(ai)
        denominator = np.einsum('ij,kil,kjm,lm->',\
                                psi_a,np.conjugate(self.M[site]),self.M[site],psi_b)
        energy = numerator / denominator
        return energy