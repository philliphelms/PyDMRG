import numpy as np

class MPO:

    def __init__(self, hamType, param, N):
        """
        hamType choices:
            heis - Heisenberg Model
                param = (J,h)
            ising - Ising Model
                param = 
            tasep - Totally Assymetric Simple Exclusion Process Model
                param = (alpha,s,beta)
        """
        self.hamType = hamType
        self.N = N
        self.Sp = np.array([[0,1],
                             [0,0]])
        self.Sm = np.array([[0,0],
                             [1,0]])
        self.Sz = np.array([[0.5,0],
                             [0,-0.5]])
        self.Sx = np.array([[0,0.5],
                             [0.5,0]])
        self.Sy = 1/(2j)*np.array([[0,1],
                                     [-1,0]])
        self.n = np.array([[0,0],
                           [0,1]])
        self.v = np.array([[1,0],
                           [0,0]])
        self.I = np.eye(2)
        self.z = np.zeros([2,2])
        if hamType is "heis":
            raise ValueError('Heisenberg model not yet implemented')
            #self.J = param[0]
            #self.h = param[1]
            #self.w_arr = np.array([[self.I,           self.zero_mat,      self.zero_mat,      self.zero_mat,   self.zero_mat],
            #                       [self.S_p,         self.zero_mat,      self.zero_mat,      self.zero_mat,   self.zero_mat],
            #                       [self.S_m,         self.zero_mat,      self.zero_mat,      self.zero_mat,   self.zero_mat],
            #                       [self.S_z,         self.zero_mat,      self.zero_mat,      self.zero_mat,   self.zero_mat],
            #                       [-self.h*self.S_z, self.J/2.*self.S_m, self.J/2.*self.S_p, self.J*self.S_z, self.I       ]])
        elif hamType is "tasep":
            self.alpha = param[0]
            self.s = param[1]
            self.beta = param[2]
            self.W = []
            self.W.insert(len(self.W),np.array([[self.alpha*(np.exp(-self.s)*self.Sm-self.v),np.exp(-self.s)*self.Sp,-self.n,self.I]]))
            for i in range(self.N-2):
                self.W.insert(len(self.W),np.array([[self.I,  self.z,                 self.z,  self.z],\
                                                    [self.Sm, self.z,                 self.z,  self.z],\
                                                    [self.v,  self.z,                 self.z,  self.z],\
                                                    [self.z,  np.exp(-self.s)*self.Sp,-self.n, self.I]]))
            self.W.insert(len(self.W),np.array([[self.I],[self.Sm],[self.v],[self.beta*(np.exp(-self.s)*self.Sp-self.n)]]))
        elif ham_typ is "ising":
            print("I haven't implemented the heisenberg model yet")
        else:
            raise ValueError("Input Hamiltonian type is not supported")
"""
    def W(self,site):
        if self.hamType is "heis":
            if site == 0:
                return np.expand_dims(self.w_arr[-1,:],0)
            elif site == self.L-1:
                return np.expand_dims(self.w_arr[:,0],1)
            else:
                return self.w_arr
        elif self.hamType is "tasep":
            if site == 0:
                tmp_arr = np.expand_dims(self.w_arr[-1,:],0)
                tmp_arr[0,0,:,:] = self.alpha*(np.exp(-self.s)*self.S_m-self.v)
                return tmp_arr
            if site == self.L-1:

                tmp_arr = np.expand_dims(self.w_arr[:,0],1)
                tmp_arr[-1,0,:,:] = self.beta*(np.exp(-self.s)*self.S_p-self.n)
                return tmp_arr
            else:
                return self.w_arr
        elif self.hamType is "ising":
            print("I haven't implemented the Ising model yet")
        else:
            raise ValueError("Input Hamiltonian Type is not supported")"""
