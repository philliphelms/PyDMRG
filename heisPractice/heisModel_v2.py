import numpy as np
import matplotlib.pyplot as plt

class HeisMPS:
    def __init__(self,nsite):
        # Basic Model Information ######################
        self.nsite = nsite
        self.d = 2 # Dimension of local state space
        self.h = 1 # First interaction parameter
        self.J = 1 # Second interaction parameter
        # Optimization Parameters ######################
        self.init_guess_type = 'rand' # 'rand' or 'hf'
        self.tol = 1e-5 # Optimization Tolerance
        self.plot_option = True # Not Operational
        self.plot_cnt = 0 # Count plot updates
        self.max_iter = 100 # Maximum number of iterations
        # Store MPO ####################################
        # Key Matrices
        s_hat = np.array([[[0,1],  [1,0]],
                                   [[0,-1j],[1j,0]],
                                   [[1,0],  [0,-1]]])
        zero_mat = np.zeros([2,2])
        I = np.eye(2)
        # Construct MPO
        self.w_arr = np.array([[I,                   zero_mat,               zero_mat,               zero_mat,               zero_mat],
                              [s_hat[0,:,:],         zero_mat,               zero_mat,               zero_mat,               zero_mat],
                              [s_hat[1,:,:],         zero_mat,               zero_mat,               zero_mat,               zero_mat],
                              [s_hat[2,:,:],         zero_mat,               zero_mat,               zero_mat,               zero_mat],
                              [-self.h*s_hat[0,:,:], -self.J/2*s_hat[0,:,:], -self.J/2*s_hat[1,:,:], -self.J/2*s_hat[2,:,:], I       ]])
        # Construct container for resulting energies
        self.E = np.zeros(nsite)
    
    def create_initial_guess(self):
        # Function to create a Right-Canonical MPS as the initial guess
        # Follows the procedure and notation carried out in section 4.1.3.ii of Schollwock (2011)
        L = self.nsite
        for i in range(L):
            if i == 0:
                if self.init_guess_type is 'rand':
                    psi = np.random.rand(self.d**(L-1),self.d)
                elif self.init_guess_type is 'hf':
                    psi = np.zeros([self.d**(L-1),self.d])
                    psi[0,0] = 1
                else:
                    raise ValueError('Indicated initial guess type is not available')
                B = [[] for x in range(self.d)]
                a_prev = 1
            else:
                psi = np.dot(u,np.diag(s)).reshape(self.d**(L-(i+1)),-1)
                a_prev = a_curr
            (u,s,v) = np.linalg.svd(psi,full_matrices=0)
            a_curr = min(self.d**(i+1),self.d**(L-(i)))
            v = np.transpose(v)
            for j in range(self.d):
                if a_curr > a_prev:
                    v = v.reshape(a_curr*self.d,-1)
                    B[j].insert(0,v[j*a_curr:(j+1)*a_curr,:])
                else:
                    v = v.reshape(-1,a_curr*self.d)
                    B[j].insert(0,v[:,j*a_curr:(j+1)*a_curr])
        self.M = B

    def W(self,ind):
        # Simple function that acts as an index for the MPO matrices W. 
        # Returns correct vectors for first and last site and the full matrix for all intermediate sites
        if ind == 0:
            return np.array([self.w_arr[-1,:]])
        elif ind == self.nsite-1:
            return np.array([self.w_arr[:,0]])
        else:
            return self.w_arr

    def calc_all_lr(self):
        # Calculate all L- andR-expressions iteratively for sites L-1 through 1
        # Follows the procedure and notation outlined in Equation 197 of Section 6.2 of Schollwock (2011)
        self.R = []
        self.L = []
        # Insert R[L] dummy array
        self.R.insert(0,np.array([[[1]]])) 
        self.L.insert(0,np.array([[[1]]])) 
        for out_cnt in range(self.nsite)[::-1]:
            for i in range(self.d):
                for j in range(self.d):
                    if out_cnt == 0: 
                        tmp_array = np.array([[[1]]])
                    else:
                        if i+j == 0:
                            tmp_array = np.einsum('ji,kl,mn,iln->jkm',self.M[i][out_cnt],self.W(out_cnt)[:,:,i,j],self.M[j][out_cnt],self.R[0])
                        else:
                            tmp_array += np.einsum('ji,kl,mn,iln->jkm',self.M[i][out_cnt],self.W(out_cnt)[:,:,i,j],self.M[j][out_cnt],self.R[0])
            self.R.insert(0,tmp_array)
    
    def reshape_hamiltonian(self,H):
        # Function to reshape H array from six dimensional to correct 2D matrix
        sl,alm,al,slp,almp,alp = H.shape
        return H.reshape(sl*alm*al,-1)

    def shape_m(self,site,v):
        # The input eigenvector, v, is reshaped into two MPS matrices
        (ai,aim) = self.M[0][site].shape
        v = v.reshape(self.d,ai,aim)
        for i in range(self.d):
            self.M[i][site] = v[i,:,:]

    def make_m_2d(self,site,sweep_dir):
        # 
        (aim,ai) = self.M[0][site].shape
        m_3d = np.dstack((self.M[0][site],self.M[1][site])).transpose()
        if sweep_dir is 'R':
            return m_3d.reshape(self.d*aim,ai)
        elif sweep_dir is 'L':
            return m_3d.reshape(aim,self.d*ai)
        else:
            raise ValueError('Sweep Direction must be L or R')

    def convert_u2a(self,site,U):
        (aim,ai) = self.M[0][site].shape
        u_3d = U.reshape(self.d,aim,ai)
        for i in range(self.d):
            self.M[i][site] = u_3d[i,:,:]

    def update_L(self,site):
        # Update L array associated with the partition occuring at site
        for i in range(self.d):
             for j in range(self.d):
                if i+j == 0:
                    tmp_array = np.einsum('ji,kl,mn,jlm->ikn',self.M[i][site],self.W(site)[:,:,i,j],self.M[j][site],self.L[site])
                else:
                    tmp_array += np.einsum('ji,kl,mn,jlm->ikn',self.M[i][site],self.W(site)[:,:,i,j],self.M[j][site],self.L[site])
        if len(self.L) <= site+1:
            self.L.insert(len(self.L),tmp_array)
        else:
            self.L[site+1] = tmp_array

    def update_R(self,site):
        # Update R-expression associated with the partition occuring at site
        for i in range(self.d):
            for j in range(self.d):
                if i+j == 0:
                    tmp_array = np.einsum('ji,kl,mn,iln->jkm',self.M[i][site],self.W(site)[:,:,i,j],self.M[j][site],self.R[site+1])
                else:
                    tmp_array += np.einsum('ji,kl,mn,iln->jkm',self.M[i][site],self.W(site)[:,:,i,j],self.M[j][site],self.R[site+1])
        self.R[site] = tmp_array

    def eval_energy(self):
        totalE = 0
        for i in range(len(self.E)):
            totalE += self.E[i]
        return -totalE

    def update_plot(self,new_entry):
        if self.plot_option:
            if self.plot_cnt == 0:
                plt.style.use('ggplot')
                plt.ion()
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111)
                self.conv_xdat = [self.plot_cnt]
                self.conv_ydat = [new_entry]
                self.ax.plot(self.conv_xdat,self.conv_ydat,'b-')
                plt.xlabel('Iteration')
                plt.ylabel('Energy')
                self.fig.canvas.draw()
            else:
                self.conv_xdat.insert(len(self.conv_xdat),self.plot_cnt)
                self.conv_ydat.insert(len(self.conv_ydat),new_entry)
                self.ax.plot(self.conv_xdat,self.conv_ydat,'b-')
                self.fig.canvas.draw()
            self.plot_cnt += 1
    
    
    def create_all_occ_strs_a(self,final_site,curr_str):
        if len(curr_str) == final_site:
            if self.occ_strs_a is None:
                self.occ_strs_a = np.array([curr_str])
            else:
                self.occ_strs_a = np.append(self.occ_strs_a,np.array([curr_str]),axis=0)
        else:
            for i in range(self.d):
                self.create_all_occ_strs_a(final_site,np.append(curr_str,[i]))

    def create_all_occ_strs_b(self,final_site,curr_str):
        if len(curr_str) == final_site:
            if self.occ_strs_b is None:
                self.occ_strs_b = np.array([curr_str])
            else:
                self.occ_strs_b = np.append(self.occ_strs_b,np.array([curr_str]),axis=0)
        else:
            for i in range(self.d):
                self.create_all_occ_strs_b(final_site,np.append(curr_str,[i]))

    def psi_a_multiply_m(self,curr_site,final_site,current_result,first_iter,occ_str_ind,direction):
        if curr_site == final_site and not first_iter:
            return current_result
        else:
            d_val = self.occ_strs_a[occ_str_ind,curr_site]
            if direction is 'd':
                current_result = np.einsum('ij,jk->ik',current_result,np.transpose(self.M[int(d_val)][curr_site]))
            else:
                current_result = np.einsum('ij,jk->ik',current_result,self.M[int(d_val)][curr_site])
            if direction is 'd':
                if curr_site == 0:
                    next_site = 0
                    direction = 'u'
                else:
                    curr_site -= 1
            else:
                curr_site += 1
            return self.psi_a_multiply_m(curr_site,final_site,current_result,False,occ_str_ind,direction)

    def psi_b_multiply_m(self,curr_site,final_site,current_result,first_iter,occ_str_ind,direction,site):
        print('{},{}'.format(curr_site,final_site))
        if curr_site == final_site and not first_iter:
            return current_result
        else:
            d_val = self.occ_strs_b[occ_str_ind,curr_site+site]
            if direction is 'u':
                current_result = np.einsum('ij,jk->ik',current_result,self.M[int(d_val)][curr_site+site])
            else:
                current_result = np.einsum('ij,jk->ik',current_result,np.transpose(self.M[int(d_val)][curr_site+site]))
            if direction is 'u':
                if curr_site == self.nsite-1:
                    next_site = curr_site
                    direction = 'd'
                else:
                    curr_site += 1
            else:
                curr_site -= 1
            return self.psi_b_multiply_m(curr_site,final_site,current_result,False,occ_str_ind,direction,site)

    def calc_psi_a(self,site):
        if site is 0:
            return np.array([[1]])
        # Create occupation strings
        self.occ_strs_a = None
        self.create_all_occ_strs_a(site-1,np.array([]))
        ns,_ = self.occ_strs_a.shape
        for i in range(ns):
            if i == 0:
                psi_a = self.psi_a_multiply_m(site-1,site-1,np.array([[1]]),True,i,'d')
            else:
                psi_a += self.psi_a_multiply_m(site-1,site-1,np.array([[1]]),True,i,'d')
        return psi_a

    def calc_psi_b(self,site):
        if site is self.nsite-1:
            return np.array([[1]])
        self.occ_strs_b = None
        self.create_all_occ_strs_b(self.nsite-site,np.array([]))
        ns,_ = self.occ_strs_b.shape
        for i in range(ns):
            if i == 0:
                psi_b = self.psi_b_multiply_m(site+1,site+1,np.array([[1]]),True,i,'u',site)
            else:
                psi_b += self.psi_b_multiply_m(site+1,site+1,np.array([[1]]),True,i,'u',site)
        return psi_b

    def calc_energy(self,site):
        # Calculates the energy of a given state using the hamilonian operators
        # Done according section 6 of Schollwock (2011)
        psi_a = self.calc_psi_a(site)
        psi_b = self.calc_psi_b(site)
        # Calc E
        for i in range(self.d):
            for j in range(self.d):
                numerator = np.einsum('ijk,lmjn,opn,oi,->',self.L[site],self.W[site],self.R[site+1],self.M[i][site],self.M[i][site])
                denominator = np.einsum('->',)
        return numerator/denominator

    def calc_ground_state(self):
        # Run the DMRG optimization to calculate the system's ground state
        # Follows the procedure and notation outlined in section 6.3 of Schollwock (2011)
        self.create_initial_guess()
        self.calc_all_lr()
        converged = False
        sweep_cnt = 0
        currE = 0
        while not converged:
            print('Beginning Sweep Set {}'.format(sweep_cnt))
            # Sweep Right ###########################################
            print('\tRight Sweep')
            for i in range(self.nsite-1):
                print('\t\tSite {}'.format(i))
                # Solve eigenvalue problem
                H = np.einsum('ijk,lmno,pmq->nipokq',self.L[i],self.W(i),self.R[i+1])
                H = self.reshape_hamiltonian(H)
                w,v = np.linalg.eig(H)
                w = np.sort(w)
                v = v[:,w.argsort()]
                self.shape_m(i,v[:,0])
                # Left-normalize and distribute matrices
                (U,S,V) = np.linalg.svd(self.make_m_2d(i,'R'),full_matrices=0)
                self.convert_u2a(i,U)
                for j in range(self.d):
                    self.M[j][i+1] = np.einsum('i,ij,jk->ik',S,V,self.M[j][i+1])
                # Update L-expression
                self.update_L(i)
                # Save Resulting energies
                self.E[i] = self.calc_energy(i)
            # Sweep Left 
            print('\tLeft Sweep')
            for i in range(self.nsite-1,0,-1):
                print('\t\tSite {}'.format(i))
                # Solve eigenvalue problem
                H = np.einsum('ijk,lmno,pmq->nipokq',self.L[i],self.W(i),self.R[i+1]) # Indices should be correct
                H = self.reshape_hamiltonian(H)
                w,v = np.linalg.eig(H)
                w = np.sort(w)
                v= v[:,w.argsort()]
                self.shape_m(i,v[:,0])
                # Right-normalize and distribute matrices
                (U,S,V) = np.linalg.svd(self.make_m_2d(i,'L'),full_matrices=0)
                self.convert_u2a(i,V)
                for j in range(self.d):
                    self.M[j][i] = np.einsum('i,ij,jk->ik',S,U,self.M[j][i])
                # Update R-expression
                self.update_R(i)
                # Save Resulting Energies
                self.E[i] = self.calc_energy(i)
            # Check for Convergence #################################
            prevE = currE
            currE = self.eval_energy()
            print('\tResulting Energy:\t{}'.format(currE))
            self.update_plot(currE)
            if np.abs(prevE-currE) < self.tol:
                converged = True
                print('Energy Converged at:\t{}'.format(currE))
            elif sweep_cnt > self.max_iter:
                converged = True
                print('Maximum number of iterations exceeded before convergence')
            sweep_cnt += 1

if __name__ == "__main__":
    x = HeisMPS(6)
    x.calc_ground_state()
