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
            sep_2d - Two Dimensional Simple Exclusion Process Model
                param = (jump_left,jump_right,enter_left,enter_right,
                         exit_left,exit_right,jump_up,jump_down,
                         enter_top,enter_bottom,exit_top,exit_bottom,s))
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
        if hamType is "heis_2d":
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
            # Temporary fix to match ushnish 
            #self.alpha = param[0]
            #self.beta = param[1]
            #self.p = param[2]
            #self.q = param[3]
            #self.gamma = param[4]
            #self.delta = param[5]
            # Same thing
            self.exp_alpha = self.alpha*np.exp(self.s)
            self.exp_gamma = self.gamma*np.exp(-self.s)
            self.exp_p = self.p*np.exp(-self.s)
            self.exp_q = self.q*np.exp(self.s)
            self.exp_beta = self.beta*np.exp(-self.s)
            self.exp_delta = self.delta*np.exp(-self.s)
            # Create MPO
            self.W = []
            self.W.insert(len(self.W),np.array([[(self.exp_alpha*self.Sm-self.alpha*self.v)+
                                                 (self.exp_gamma*self.Sp-self.gamma*self.n),
                                                 self.Sp,
                                                 -self.n,
                                                 self.Sm,
                                                 -self.v,
                                                 self.I]]))
            for i in range(int(self.N-2)):
                self.W.insert(len(self.W),np.array([[self.I,             self.z,  self.z,  self.z, self.z,  self.z],
                                                    [self.exp_p*self.Sm, self.z,  self.z,  self.z, self.z,  self.z],
                                                    [self.p*self.v,      self.z,  self.z,  self.z, self.z,  self.z],
                                                    [self.exp_q*self.Sp, self.z,  self.z,  self.z, self.z,  self.z],
                                                    [self.q*self.n,      self.z,  self.z,  self.z, self.z,  self.z],
                                                    [self.z,             self.Sp, -self.n, self.Sm,-self.v, self.I]]))
            self.W.insert(len(self.W),np.array([[self.I],
                                                [self.exp_p*self.Sm],
                                                [self.p*self.v],
                                                [self.exp_q*self.Sp],
                                                [self.q*self.n],
                                                [(self.exp_beta*self.Sm-self.beta*self.v)+
                                                 (self.exp_delta*self.Sp-self.delta*self.n)]]))
        elif hamType is "sep_2d":
            self.N2d = int(np.sqrt(self.N))
            self.jl = param[0]
            self.jr = param[1]
            self.il = param[2]
            self.ir = param[3]
            self.ol = param[4]
            self.outr = param[5]
            self.ju = param[6]
            self.jd = param[7]
            self.it = param[8]
            self.ib = param[9]
            self.ot = param[10]
            self.ob = param[11]
            self.s = param[12]
            #print('jump left = {}'.format(self.jl))
            #print('jump right = {}'.format(self.jr))
            #print('in left = {}'.format(self.il))
            #print('in right = {}'.format(self.ir))
            #print('out left = {}'.format(self.ol))
            #print('out right = {}'.format(self.outr))
            #print('jump up = {}'.format(self.ju))
            #print('jump down = {}'.format(self.jd))
            #print('in top = {}'.format(self.it))
            #print('in bottom = {}'.format(self.ib))
            #print('out top = {}'.format(self.ot))
            #print('out bottom = {}'.format(self.ob))
            #print('s = {}'.format(self.s))
            ham_dim = 10+(self.N2d-2)*4
            self.w_arr = np.zeros((ham_dim,ham_dim,2,2))
            # Build generic first column
            self.w_arr[0,0,:,:] = self.I
            self.w_arr[1,0,:,:] = self.ju*self.Sm
            self.w_arr[self.N2d,0,:,:] = self.jr*self.Sm
            self.w_arr[self.N2d+1,0,:,:] = self.ju*self.v
            self.w_arr[2*self.N2d,0,:,:] = self.jr*self.v
            self.w_arr[2*self.N2d+1,0,:,:] = self.jd*self.Sp
            self.w_arr[3*self.N2d,0,:,:] = self.jl*self.Sp
            self.w_arr[3*self.N2d+1,0,:,:] = self.jd*self.n
            self.w_arr[4*self.N2d,0,:,:] = self.jl*self.n
            # Build generic interior
            col_ind = 1
            row_ind = 2
            for j in range(4):
                for i in range(self.N2d-1):
                    self.w_arr[row_ind,col_ind,:,:] = self.I
                    col_ind += 1
                    row_ind += 1
                col_ind += 1
                row_ind += 1
            # Build bottom row
            self.w_arr[-1,self.N2d,:,:] = np.exp(-self.s)*self.Sp
            self.w_arr[-1,2*self.N2d,:,:] = -self.n
            self.w_arr[-1,3*self.N2d,:,:] = np.exp(-self.s)*self.Sm
            self.w_arr[-1,4*self.N2d,:,:] = -self.v
            self.w_arr[-1,4*self.N2d+1,:,:] = self.I
            self.W = []
            for i in range(self.N2d):
                for j in range(self.N2d):
                    # copy generic mpo
                    curr_w_arr = self.w_arr.copy()
                    # Add interaction with external reservoirs
                    if j is 0:
                        curr_w_arr[-1,0,:,:] += self.il*(np.exp(-self.s)*self.Sm-self.v) +\
                                                self.ol*(np.exp(-self.s)*self.Sp-self.n)
                    if (j is 0) and (i is not 0): # Prevents interaction between ends
                        curr_w_arr[self.N2d,0,:,:] = self.z
                        curr_w_arr[2*self.N2d,0,:,:] = self.z
                        curr_w_arr[3*self.N2d,0,:,:] = self.z
                        curr_w_arr[4*self.N2d,0,:,:] = self.z
                    if i is 0:
                        curr_w_arr[-1,0,:,:] += self.ib*(np.exp(-self.s)*self.Sm-self.v) +\
                                                self.ob*(np.exp(-self.s)*self.Sp-self.n)
                    if j is self.N2d-1:
                        curr_w_arr[-1,0,:,:] += self.ir*(np.exp(-self.s)*self.Sm-self.v) +\
                                                self.outr*(np.exp(-self.s)*self.Sp-self.n)
                    if i is self.N2d-1:
                        curr_w_arr[-1,0,:,:] += self.it*(np.exp(-self.s)*self.Sm-self.v) +\
                                                self.ot*(np.exp(-self.s)*self.Sp-self.n)
                    if (i is 0) and (j is 0):
                        self.W.insert(len(self.W),np.expand_dims(curr_w_arr[-1,:],0))
                    elif (i is self.N2d-1) and (j is self.N2d-1):
                        self.W.insert(len(self.W),np.expand_dims(curr_w_arr[:,0],1))
                    else:
                        self.W.insert(len(self.W),curr_w_arr)

        elif hamType is "ising":
            self.J = param[0]
            self.h = param[1]
            self.w_arr = np.array([[self.I,         self.z,         self.z],
                                   [self.Sz,        self.z,         self.z],
                                   [self.h*self.Sz, self.J*self.Sz, self.I]])
            self.W = []
            self.W.insert(len(self.W),np.expand_dims(self.w_arr[-1,:],0))
            for i in range(int(self.N-2)):
                self.W.insert(len(self.W),self.w_arr)
            self.W.insert(len(self.W),np.expand_dims(self.w_arr[:,0],1))
        else:
            raise ValueError("Specified Hamiltonian type is not supported")
    
    def return_full_ham(self):
        # This function calculates the full hamiltonian matrix
        # As a warning, it is computationally expensive 
        H = np.zeros((self.N**2,self.N**2))
        for i in range(self.N**2):
            i_occ = list(map(lambda x: int(x),'0'*(self.N-len(bin(i)[2:]))+bin(i)[2:]))
            for j in range(self.N**2):
                j_occ = list(map(lambda x: int(x),'0'*(self.N-len(bin(j)[2:]))+bin(j)[2:]))
                tmp_mat = np.array([[1]])
                for k in range(self.N):
                    tmp_mat = np.einsum('ij,jk->ik',tmp_mat,self.W[k][:,:,i_occ[k],j_occ[k]])
                H[i,j] = tmp_mat[[0]]
        return H
