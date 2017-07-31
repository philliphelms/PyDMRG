import numpy as np

class HeisMPO:
    # Class that constructs and returns the heisenberg hamiltonian matrix 
    # product operator.
    #
    # Functions:
    #   W(ind) - Returns the MPO associated with the given index.
    def __init__(self,h,J,L):
        self.h = h
        self.J = J
        self.L = L
        S_p = np.array([[0,1],
                        [0,0]])/2
        S_m = np.array([[0,0],
                        [1,0]])/2
        S_z = np.array([[1,0],
                        [0,-1]])/2
        zero_mat = np.zeros([2,2])
        I = np.eye(2)
        # Construct MPO
        self.w_arr = np.array([[I,           zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [S_p,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [S_m,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [S_z,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [-self.h*S_z, self.J/2.*S_m, self.J/2.*S_p, self.J*S_z, I       ]])
    
    def W(self,ind):
        # Simple function that acts as an index for the MPO matrices W. 
        # Returns correct vectors for first and last site and the full matrix for all intermediate sites
        if ind == 0:
            return np.expand_dims(self.w_arr[-1,:],0)
        elif ind == self.L-1:
            return np.expand_dims(self.w_arr[:,0],1)
        else:
            return self.w_arr
