import numpy as np

class HeisMPO:
    """
    Description:
        An object containing the MPO for the heisenberg hamiltonian with a single
        function that returns the MPO associated with each site in the spin chain.
        
    Class Members:
        > self.L               - The number of sites for the system
        > self.h               - The strength of the orienting force
        > self.J               - The interaction strenght between neighboring spins
        > self.w_arr           - The full four-dimensional numpy array containing the 
                                 MPO for the heisenberg model hamiltonian.
    
    Key Functions:
        1) W(ind)              - Simply returns the correct MPO for the given site (ind).
                                 Main job is to return a column vector on the left boundary,
                                 a row vector on the right boundary and the full matrix on
                                 all others.
    """
    def __init__(self,h,J,L):
        self.h = h
        self.J = J
        self.L = L
        S_p = np.array([[0,1],
                        [0,0]])
        S_m = np.array([[0,0],
                        [1,0]])
        S_z = np.array([[0.5,0],
                        [0,-0.5]])
        zero_mat = np.zeros([2,2])
        I = np.eye(2)
        # Construct MPO
        self.w_arr = np.array([[I,           zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [S_p,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [S_m,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [S_z,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [-self.h*S_z, self.J/2.*S_m, self.J/2.*S_p, self.J*S_z, I       ]])
        # self.w_arr = np.swapaxes(self.w_arr,2,3)
    
    def W(self,ind):
        if ind == 0:
            return np.expand_dims(self.w_arr[-1,:],0)
        elif ind == self.L-1:
            return np.expand_dims(self.w_arr[:,0],1)
        else:
            return self.w_arr
