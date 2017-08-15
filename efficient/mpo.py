import numpy as np

class MPO:

    def __init__(self, ham_type, param, L):
        """
        ham_type choices:
            heis - Heisenberg Model
                param = (J,h)
            ising - Ising Model
                param = 
            sep - Simply Exclusion Process Model
        """
        self.ham_type = ham_type
        self.L = L
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
        self.v = np.array([[0,0],
                           [0,1]])
        self.n = np.array([[1,0],
                           [0,0]])
        self.I = np.eye(2)
        self.zero_mat = np.zeros([2,2])
        if ham_type is "heis":
            self.J = param[0]
            self.h = param[1]
            self.w_arr = np.array([[self.I,           self.zero_mat,      self.zero_mat,      self.zero_mat,   self.zero_mat],
                                   [self.S_p,         self.zero_mat,      self.zero_mat,      self.zero_mat,   self.zero_mat],
                                   [self.S_m,         self.zero_mat,      self.zero_mat,      self.zero_mat,   self.zero_mat],
                                   [self.S_z,         self.zero_mat,      self.zero_mat,      self.zero_mat,   self.zero_mat],
                                   [-self.h*self.S_z, self.J/2.*self.S_m, self.J/2.*self.S_p, self.J*self.S_z, self.I       ]])
        elif ham_type is "sep":
            print("I haven't done the work for the sep model yet")
        elif ham_typ is "ising":
            print("I haven't implemented the heisenberg model yet")
        else:
            raise ValueError("Input Hamiltonian type is not supported")

    def W(self,site):
        if self.ham_type is "heis":
            if site == 0:
                return np.expand_dims(self.w_arr[-1,:],0)
            elif site == self.L-1:
                return np.expand_dims(self.w_arr[:,0],1)
            else:
                return self.w_arr
        elif self.ham_type is "sep":
            print("I haven't implemented sep model yet")
        elif self.ham_type is "ising":
            print("I haven't implemented the Ising model yet")
        else:
            raise ValueError("Input Hamiltonian Type is not supported")
