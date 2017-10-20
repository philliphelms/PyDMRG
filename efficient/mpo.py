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
            sep - Simple Exclusion Process Model
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
            self.J = param[0]
            self.h = param[1]
            self.w_arr = np.array([[self.I,           self.z,      self.z,      self.z,   self.z],
                                   [self.Sp,         self.z,      self.z,      self.z,   self.z],
                                   [self.Sm,         self.z,      self.z,      self.z,   self.z],
                                   [self.Sz,         self.z,      self.z,      self.z,   self.z],
                                   [-self.h*self.Sz, self.J/2.*self.Sm, self.J/2.*self.Sp, self.J*self.Sz, self.I       ]])
            self.W = []
            self.W.insert(len(self.W),np.expand_dims(self.w_arr[-1,:],0))
            for i in range(int(self.N-2)):
                self.W.insert(len(self.W),self.w_arr)
            self.W.insert(len(self.W),np.expand_dims(self.w_arr[:,0],1))
        elif hamType is "tasep":
            self.alpha = param[0]
            self.s = param[1]
            self.beta = param[2]
            self.W = []
            self.W.insert(len(self.W),np.array([[self.alpha*(np.exp(-self.s)*self.Sm-self.v),np.exp(-self.s)*self.Sp,-self.n,self.I]]))
            for i in range(int(self.N-2)):
                self.W.insert(len(self.W),np.array([[self.I,  self.z,                 self.z,  self.z],\
                                                    [self.Sm, self.z,                 self.z,  self.z],\
                                                    [self.v,  self.z,                 self.z,  self.z],\
                                                    [self.z,  np.exp(-self.s)*self.Sp,-self.n, self.I]]))
            self.W.insert(len(self.W),np.array([[self.I],[self.Sm],[self.v],[self.beta*(np.exp(-self.s)*self.Sp-self.n)]]))
        elif hamType is "ising":
            print("I haven't implemented the heisenberg model yet")
        elif hamType is "sep":
            print("I haven't implemented the heisenberg model yet")
        else:
            raise ValueError("Input Hamiltonian type is not supported")
