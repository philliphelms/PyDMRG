import numpy as np

class HeisMPS:
    def __init__(self,nsite):
        # Basic Model Information ######################
        self.nsite = nsite
        self.d = 2 # Dimension of local state space
        self.h = 1 # First interaction parameter
        self.J = 1 # Second interaction parameter
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
        
    def create_initial_guess(self):
        # Function to create a Right-Canonical MPS as the initial guess
        # Follows the procedure and notation carried out in section 4.1.3.ii of Schollwock (2011)
        L = self.nsite
        for i in range(L):
            if i == 0:
                psi = np.zeros([self.d**(L-1),self.d])
                psi[0,0] = 1
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

    def make_m_2d(self,site):
        # 
        (aim,ai) = self.M[0][site].shape
        m_3d = np.dstack((self.M[0][site],self.M[1][site])).transpose()
        return m_3d.reshape(self.d*aim,ai)

    def convert_u2a(self,site,U):
        (aim,ai) = self.M[0][site].shape
        u_3d = U.reshape(self.d,aim,ai)
        for i in range(self.d):
            self.M[i][site] = u_3d[i,:,:]

    def update_L(self,site):
        # Update L array associated with the partition occuring at site+1
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

    def calc_ground_state(self):
        # Run the DMRG optimization to calculate the system's ground state
        # Follows the procedure and notation outlined in section 6.3 of Schollwock (2011)
        self.create_initial_guess()
        self.calc_all_lr()
        converged = False
        while not converged:
            # Sweep Right ###########################################
            for i in range(self.nsite-1):
                # Solve eigenvalue problem
                H = np.einsum('ijk,lmno,pmq->nipokq',self.L[i],self.W(i),self.R[i+1])
                H = self.reshape_hamiltonian(H)
                w,v = np.linalg.eig(H)
                w = np.sort(w)
                v = v[:,w.argsort()]
                self.shape_m(i,v[:,0])
                # Left-normalize and distribute matrices
                (U,S,V) = np.linalg.svd(self.make_m_2d(i),full_matrices=0)
                self.convert_u2a(i,U)
                for j in range(self.d):
                    self.M[j][i+1] = np.einsum('i,ij,jk->ik',S,V,self.M[j][i+1])
                # Update L-expression
                self.update_L(i)
                print(i)
            # Sweep Left ############################################
            for i in range(self.nsite-1,1,-1):
                # Solve eigenvalue problem
                # Left-normalize and distribute matrices
                print(i)
            # Check for Convergence #################################

            converged = True

if __name__ == "__main__":
    x = HeisMPS(6)
    x.calc_ground_state()
