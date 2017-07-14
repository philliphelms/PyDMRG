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
        self.W = np.array([[I,                   zero_mat,               zero_mat,               zero_mat,               zero_mat],
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
                B = [[],[]]
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
            return self.W[-1,:]
        elif ind == self.nsite-1:
            return self.W[:,0]
        else:
            return self.W

    def calc_all_r(self):
        # Calculate all R-expressions iteratively for sites L-1 through 1
        # Follows the procedure and notation outlined in Equation 197 of Section 6.2 of Schollwock (2011)
        self.R = np.empty(self.nsite-1)
        # Insert R[L] dummy array
        np.insert(self.R,0,np.array([[[1]]])) 
        for i in range(L-1):



    def calc_ground_state(self):
        # Run the DMRG optimization to calculate the system's ground state
        # Follows the procedure and notation outlined in section 6.3 of Schollwock (2011)
        self.create_initial_guess()
        self.calc_all_r()
        converged = False
        while not converged:
             converged = True

if __name__ == "__main__":
    x = HeisMPS(4)
    x.calc_ground_state()
