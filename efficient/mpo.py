import numpy as np

class MPO:

    def __init__(self, hamType, param, N):
        """
        hamType choices:
            heis - Heisenberg Model
                param = (J,h)
            heis_2d - Two Dimensional Heisenberg Model
                param = (J,h)
            ising - Ising Model
                param = (J) 
            tasep - Totally Assymetric Simple Exclusion Process Model
                param = (alpha,s,beta)
            sep - Simple Exclusion Process Model
                param = (alpha,gamma,p,q,beta,delta,s)
        """
        self.hamType = hamType
        self.N = N
        # Define various operators
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
        # Creat hamiltonian based on operator type
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
        lif hamType is "heis_2d":
            self.N2d = int(np.sqrt(self.N))
            self.J = param[0]
            self.h = param[1]
            ham_dim = 8+(self.N2d-2)*3
            self.w_arr = np.zeros((ham_dim,ham_dim,2,2))
            # Build first column
            self.w_arr[0,0,:,:] = self.I
            self.w_arr[1,0,:,:] = self.Sp
            self.w_arr[self.N2d,0,:,:] = self.Sp
            self.w_arr[self.N2d+1,0,:,:] = self.Sm
            self.w_arr[2*self.N2d,0,:,:] = self.Sm
            self.w_arr[2*self.N2d+1,0,:,:] = self.Sz
            self.w_arr[3*self.N2d,0,:,:] = self.Sz
            self.w_arr[-1,0,:,:] = -self.h*self.Sz
            # Build Interior
            col_ind = 1
            row_ind = 2
            for i in range(self.N2d-1):
                self.w_arr[row_ind,col_ind,:,:] = self.I
                col_ind += 1
                row_ind += 1
            col_ind += 1
            row_ind += 1
            for i in range(self.N2d-1):
                self.w_arr[row_ind,col_ind,:,:] = self.I
                col_ind += 1
                row_ind += 1
            col_ind += 1
            row_ind += 1
            for i in range(self.N2d-1):
                self.w_arr[row_ind,col_ind,:,:] = self.I
                col_ind += 1
                row_ind += 1
            col_ind += 1
            row_ind += 1
            # Build bottom row
            self.w_arr[-1,self.N2d,:,:] = self.J/2*self.Sm
            self.w_arr[-1,2*self.N2d,:,:] = self.J/2*self.Sp
            self.w_arr[-1,3*self.N2d,:,:] = self.J*self.Sz
            self.w_arr[-1,3*self.N2d+1,:,:] = self.I
            # Create alternate array for points at top boundary
            self.border_w_arr = self.w_arr
            self.border_w_arr[self.N2d,0,:,:] = self.z
            self.border_w_arr[2*self.N2d,0,:,:] = self.z
            self.border_w_arr[3*self.N2d,0,:,:] = self.z
            # Create W
            self.W = []
            self.W.insert(len(self.W),np.expand_dims(self.w_arr[-1,:],0))
            for i in range(self.N2d**2):
                if (i+1)%self.N2d:
                    self.W.insert(len(self.W),self.border_w_arr)
                else:
                    self.W.insert(len(self.W),self.w_arr)
            self.W.insert(len(self.W),np.expand_dims(self.w_arr[:,0],1))
        elif hamType is "tasep":
            self.alpha = param[0]
            self.s = param[1]
            self.beta = param[2]
            self.W = []
            self.W.insert(len(self.W),np.array([[self.alpha*(np.exp(-self.s)*self.Sm-self.v),
                                                 np.exp(-self.s)*self.Sp,
                                                 -self.n,
                                                 self.I]]))
            for i in range(int(self.N-2)):
                self.W.insert(len(self.W),np.array([[self.I,  self.z,                 self.z,  self.z],\
                                                    [self.Sm, self.z,                 self.z,  self.z],\
                                                    [self.v,  self.z,                 self.z,  self.z],\
                                                    [self.z,  np.exp(-self.s)*self.Sp,-self.n, self.I]]))
            self.W.insert(len(self.W),np.array([[self.I],
                                                [self.Sm],
                                                [self.v],
                                                [self.beta*(np.exp(-self.s)*self.Sp-self.n)]]))
        elif hamType is "sep":
            self.alpha = param[0]
            self.gamma = param[1]
            self.p = param[2]
            self.q = param[3]
            self.beta = param[4]
            self.delta = param[5]
            self.s = param[6]
            self.W = []
            self.W.insert(len(self.W),np.array([[self.alpha*(np.exp(-self.s)*self.Sm-self.v)+
                                                 self.gamma*(np.exp(-self.s)*self.Sp-self.n),
                                                 np.exp(-self.s)*self.Sp,
                                                 -self.n,
                                                 np.exp(-self.s)*self.Sm,
                                                 -self.v,
                                                 self.I]]))
            for i in range(int(self.N-2)):
                self.W.insert(len(self.W),np.array([[self.I,         self.z,                  self.z,  self.z,                  self.z,  self.z],
                                                    [self.p*self.Sm, self.z,                  self.z,  self.z,                  self.z,  self.z],
                                                    [self.p*self.v,  self.z,                  self.z,  self.z,                  self.z,  self.z],
                                                    [self.q*self.Sp, self.z,                  self.z,  self.z,                  self.z,  self.z],
                                                    [self.q*self.n,  self.z,                  self.z,  self.z,                  self.z,  self.z],
                                                    [self.z,         np.exp(-self.s)*self.Sp, -self.n, np.exp(-self.s)*self.Sm, -self.v, self.I]]))
            self.W.insert(len(self.W),np.array([[self.I],
                                                [self.p*self.Sm],
                                                [self.p*self.v],
                                                [self.q*self.Sp],
                                                [self.q*self.n],
                                                [self.beta*(np.exp(-self.s)*self.Sm-self.v)+
                                                 self.delta*(np.exp(-self.s)*self.Sp-self.n)]]))
                                                 
        elif hamType is "ising":
            print("I haven't implemented the heisenberg model yet")
        else:
            raise ValueError("Input Hamiltonian type is not supported")
    
    def return_full_ham(self):
        # This function calculates the full hamiltonian matrix
        # As a warning, it is computationally expensive 
        H = np.zeros((self.N,self.N))
        for i in range(self.N):
            i_occ = list(map(lambda x: int(x),'0'*(self.N-len(bin(i)[2:]))+bin(i)[2:]))
            for j in range(self.N):
                j_occ = list(map(lambda x: int(x),'0'*(self.N-len(bin(j)[2:]))+bin(j)[2:]))
                tmp_mat = np.array([[1]])
                for k in range(self.N):
                    tmp_mat = np.einsum('ij,jk->ik',tmp_mat,self.W[k][:,:,i_occ[k],j_occ[k]])
                H[i,j] = tmp_mat[[0]]
        return H
