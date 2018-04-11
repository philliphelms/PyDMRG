import numpy as np
import collections

class MPO:

    def __init__(self, hamType, param, N,periodic_x=False,periodic_y=False,verbose=4):
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
                param = (jdmp_left,jump_right,enter_left,enter_right,
                         exit_left,exit_right,jdmp_up,jump_down,
                         enter_top,enter_bottom,exit_top,exit_bottom,s))
        """
        self.hamType = hamType
        self.N = N
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.verbose = verbose
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
        # Create hamiltonian based on operator type
        if hamType is "heis":
            self.J = param[0]
            self.h = param[1]
            w_arr = np.array([[self.I, self.z, self.z, self.z, self.z],
                              [self.Sp, self.z, self.z, self.z, self.z],
                              [self.Sm, self.z, self.z, self.z, self.z],
                              [self.Sz, self.z, self.z, self.z, self.z],
                              [-self.h*self.Sz, self.J/2.*self.Sm, self.J/2.*self.Sp, self.J*self.Sz, self.I ]])
            self.ops = []
            tmp_op = []
            tmp_op = [None]*self.N
            tmp_op[0] = np.expand_dims(w_arr[-1,:],0)
            for i in range(1,self.N-1):
                tmp_op[i] = w_arr
            tmp_op[-1] = np.expand_dims(w_arr[:,0],1)
            self.ops.insert(len(self.ops),tmp_op)
            # Include periodic terms
            if self.periodic_x:
                tmp_op1 = [None]*self.N
                tmp_op2 = [None]*self.N
                tmp_op3 = [None]*self.N
                tmp_op1[-1] = np.array([[self.J/2*self.Sm]])
                tmp_op2[-1] = np.array([[self.J/2*self.Sp]])
                tmp_op3[-1] = np.array([[self.J*self.Sz]])
                tmp_op1[0] = np.array([[self.Sp]])
                tmp_op2[0] = np.array([[self.Sm]])
                tmp_op3[0] = np.array([[self.Sz]])
                self.ops.insert(len(self.ops),tmp_op1)
                self.ops.insert(len(self.ops),tmp_op2)
                self.ops.insert(len(self.ops),tmp_op3)
        elif hamType is "heis_2d":
            self.Nx = self.N[0]
            self.Ny = self.N[1]
            self.N = self.Nx*self.Ny
            self.J = param[0]
            self.h = param[1]
            # Build Two site terms
            self.ops = []
            if self.J != 0:
                coupled_sites = []
                # Determine all coupled sites along x-axis
                for i in range(self.Ny):
                    for j in range(self.Nx-1):
                        coupled_sites.insert(0,[j+self.Nx*(i),j+1+self.Nx*(i)])
                # Determine all coupled sites along y-axis
                for i in range(self.Nx):
                    for j in range(self.Ny-1):
                        coupled_sites.insert(0,[i+self.Nx*(j),i+self.Nx*(j+1)])
                # Determine periodic coupling along x-axis
                if self.periodic_x:
                    for i in range(self.Ny):
                        coupled_sites.insert(0,[self.Nx*(i+1)-1,self.Nx*i])
                # Determine periodic coupling along y-axis
                if self.periodic_y:
                    for i in range(self.Nx):
                        coupled_sites.insert(0,[self.Nx*(self.Ny-1)+i,i])
                # Build All two-site Operators
                for i in range(len(coupled_sites)):
                    inds = coupled_sites[i]
                    tmp_op1 = [None]*self.N
                    tmp_op2 = [None]*self.N
                    tmp_op3 = [None]*self.N
                    tmp_op1[inds[0]] = np.array([[self.J/2*self.Sm]])
                    tmp_op2[inds[0]] = np.array([[self.J/2*self.Sp]])
                    tmp_op3[inds[0]] = np.array([[self.J*self.Sz]])
                    tmp_op1[inds[1]] = np.array([[self.Sp]])
                    tmp_op2[inds[1]] = np.array([[self.Sm]])
                    tmp_op3[inds[1]] = np.array([[self.Sz]])
                    self.ops.insert(len(self.ops),tmp_op1)
                    self.ops.insert(len(self.ops),tmp_op2)
                    self.ops.insert(len(self.ops),tmp_op3)
            # Add one site terms
            if self.h != 0:
                for i in range(self.N):
                    tmp_op1 = []
                    for j in range(self.N):
                        if i == j:
                            tmp_op1.insert(len(tmp_op1),np.array([[-self.h*self.Sz]]))
                        else:
                            tmp_op1.insert(len(tmp_op1),None)
                    self.ops.insert(len(self.ops),tmp_op1)
        elif hamType is "tasep":
            self.alpha = param[0]
            self.s = param[1]
            self.beta = param[2]
            w_arr = np.array([[self.I,  self.z,                  self.z, self.z],
                              [self.Sm, self.z,                  self.z, self.z],
                              [self.v,  self.z,                  self.z, self.z],
                              [self.z,  np.exp(-self.s)*self.Sp,-self.n, self.I]])
            self.ops = []
            tmp_op = []
            tmp_op = [None]*self.N
            if not self.periodic_x:
                tmp_op[0] = np.array([[self.alpha*(np.exp(-self.s)*self.Sm-self.v),np.exp(-self.s)*self.Sp,-self.n,self.I]])
                tmp_op[-1] = np.array([[self.I],[self.Sm],[self.v],[self.beta*(np.exp(-self.s)*self.Sp-self.n)]])
            else:
                tmp_op[0] = np.expand_dims(w_arr[-1,:],0)
                tmp_op[-1] = np.expand_dims(w_arr[:,0],1)
            for i in range(1,self.N-1):
                tmp_op[i] = w_arr
            self.ops.insert(len(self.ops),tmp_op)
            # Include periodic terms
            if self.periodic_x:
                tmp_op1 = [None]*self.N
                tmp_op2 = [None]*self.N
                tmp_op1[-1] = np.array([[np.exp(-self.s)*self.Sp]])
                tmp_op2[-1] = np.array([[-self.n]])
                tmp_op1[0] = np.array([[self.Sm]])
                tmp_op2[0] = np.array([[self.v]])
                #tmp_op1[0] = np.array([[np.exp(-self.s)*self.Sp]])
                #tmp_op2[0] = np.array([[-self.n]])
                #tmp_op1[-1] = np.array([[self.Sm]])
                #tmp_op2[-1] = np.array([[self.v]])
                self.ops.insert(len(self.ops),tmp_op1)
                self.ops.insert(len(self.ops),tmp_op2)
        elif hamType is "sep":
            self.alpha = param[0]
            self.gamma = param[1]
            self.p = param[2]
            self.q = param[3]
            self.beta = param[4]
            self.delta = param[5]
            self.s = param[6]
            # multiply these by exponential weighting
            self.exp_alpha = self.alpha*np.exp(-self.s)
            self.exp_gamma = self.gamma*np.exp(self.s)
            self.exp_p = self.p*np.exp(-self.s)
            self.exp_q = self.q*np.exp(self.s)
            self.exp_beta = self.beta*np.exp(self.s)
            self.exp_delta = self.delta*np.exp(-self.s)
            # Construct generic operator array to use at each site
            w_arr = np.array([[self.I, self.z, self.z, self.z, self.z, self.z],
                              [self.exp_p*self.Sm, self.z, self.z, self.z, self.z, self.z],
                              [self.p*self.v, self.z, self.z, self.z, self.z, self.z],
                              [self.exp_q*self.Sp, self.z, self.z, self.z, self.z, self.z],
                              [self.q*self.n, self.z, self.z, self.z, self.z, self.z],
                              [self.z, self.Sp, -self.n, self.Sm,-self.v, self.I]])
            self.ops = []
            tmp_op = [None]*self.N
            if not self.periodic_x:
                tmp_op[0] = np.array([[(self.exp_alpha*self.Sm-self.alpha*self.v)+(self.exp_gamma*self.Sp-self.gamma*self.n),\
                                       self.Sp,\
                                       -self.n,\
                                       self.Sm,\
                                       -self.v,\
                                       self.I]])
                tmp_op[-1] = np.array([[self.I],\
                                       [self.exp_p*self.Sm],\
                                       [self.p*self.v],\
                                       [self.exp_q*self.Sp],\
                                       [self.q*self.n],\
                                       [(self.exp_beta*self.Sm-self.beta*self.v)+(self.exp_delta*self.Sp-self.delta*self.n)]])
            else:
                tmp_op[0] = np.expand_dims(w_arr[-1,:],0)
                tmp_op[-1] = np.expand_dims(w_arr[:,0],1)
            for i in range(1,self.N-1):
                tmp_op[i] = w_arr
            self.ops.insert(len(self.ops),tmp_op)
            # Include periodic terms
            if self.periodic_x:
                if self.p != 0:
                    tmp_op1 = [None]*self.N
                    tmp_op2 = [None]*self.N
                    tmp_op1[-1] = np.array([[self.exp_p*self.Sp]])
                    tmp_op2[-1] = np.array([[-self.p*self.n]])
                    tmp_op1[0] = np.array([[self.Sm]])
                    tmp_op2[0] = np.array([[self.v]])
                    self.ops.insert(len(self.ops),tmp_op1)
                    self.ops.insert(len(self.ops),tmp_op2)
                if self.q != 0:
                    tmp_op1 = [None]*self.N
                    tmp_op2 = [None]*self.N
                    tmp_op1[-1] = np.array([[self.exp_q*self.Sm]])
                    tmp_op2[-1] = np.array([[-self.q*self.v]])
                    tmp_op1[0] = np.array([[self.Sp]])
                    tmp_op2[0] = np.array([[self.n]])
                    self.ops.insert(len(self.ops),tmp_op1)
                    self.ops.insert(len(self.ops),tmp_op2)
        elif hamType is "sep_2d":
            # Collect Parameters
            self.Nx = self.N[0]
            self.Ny = self.N[1]
            self.N = self.Nx*self.Ny
            self.jl = param[0]
            if not isinstance(self.jl,(collections.Sequence,np.ndarray)):
                self.jr = param[1]
                self.il = param[2]
                self.ir = param[3]
                self.ol = param[4]
                self.outr = param[5]
                self.jd = param[6]
                self.ju = param[7]
                self.it = param[8]
                self.ib = param[9]
                self.ot = param[10]
                self.ob = param[11]
                try:
                    self.sx = param[12][0]
                    self.sy = param[12][1]
                except:
                    self.sx = param[12]
                    self.sy = param[12]
                # Convert these to matrices
                self.jl = self.jl*np.ones((self.Nx,self.Ny))
                self.jr = self.jr*np.ones((self.Nx,self.Ny))
                self.jd = self.jd*np.ones((self.Nx,self.Ny))
                self.ju = self.ju*np.ones((self.Nx,self.Ny))
                self.cr_r = np.zeros((self.Nx,self.Ny))
                self.cr_l = np.zeros((self.Nx,self.Ny))
                self.cr_d = np.zeros((self.Nx,self.Ny))
                self.cr_u = np.zeros((self.Nx,self.Ny))
                self.de_r = np.zeros((self.Nx,self.Ny))
                self.de_l = np.zeros((self.Nx,self.Ny))
                self.de_d = np.zeros((self.Nx,self.Ny))
                self.de_u = np.zeros((self.Nx,self.Ny))
                self.cr_r[:,0] += self.il
                self.cr_l[:,-1] += self.ir
                self.cr_u[-1,:] += self.it
                self.cr_d[0,:] += self.ib
                self.de_l[:,0] += self.ol
                self.de_r[:,-1] += self.outr
                self.de_d[-1,:] += self.ot
                self.de_u[0,:] += self.ob
            else:
                self.jr = param[1]
                self.jd = param[2]
                self.ju = param[3]
                self.cr_r = param[4]
                self.cr_l = param[5]
                self.cr_d = param[6]
                self.cr_u = param[7]
                self.de_r = param[8]
                self.de_l = param[9]
                self.de_d = param[10]
                self.de_u = param[11]
                try: 
                    self.sx = param[12][0]
                    self.sy = param[12][1]
                except:
                    self.sx = param[12]
                    self.sy = param[12]
            # Multiply params by an exponential
            self.exp_jl = self.jl*np.exp(self.sx)  # Moving Left
            self.exp_jr = self.jr*np.exp(-self.sx) # Moving Right
            self.exp_jd = self.jd*np.exp(self.sy)   # Moving up
            self.exp_ju = self.ju*np.exp(-self.sy)  # Moving Down
            self.exp_cr_r = self.cr_r*np.exp(-self.sx) 
            self.exp_cr_l = self.cr_l*np.exp(self.sx)
            self.exp_cr_d = self.cr_d*np.exp(self.sy)
            self.exp_cr_u = self.cr_u*np.exp(-self.sy)
            self.exp_de_r = self.de_r*np.exp(-self.sx)
            self.exp_de_l = self.de_l*np.exp(self.sx)
            self.exp_de_d = self.de_d*np.exp(self.sy)
            self.exp_de_u = self.de_u*np.exp(-self.sy)
            # Allocate general operator container
            self.ops = []
            # Build generic operator (not including periodicity)
            ham_dim = 10+(self.Ny-2)*4
            # Now build actual operator
            tmp_op = []
            for i in range(self.Nx):
                for j in range(self.Ny):
                    # Build generic mpo
                    w_arr = np.zeros((ham_dim,ham_dim,2,2))
                    w_arr[0,0,:,:] = self.I
                    w_arr[1,0,:,:] = self.exp_jr[j,i-1]*self.Sm
                    w_arr[self.Ny,0,:,:] = self.exp_jd[j-1,i]*self.Sm
                    w_arr[self.Ny+1,0,:,:] = self.jr[j,i-1]*self.v
                    w_arr[2*self.Ny,0,:,:] = self.jd[j-1,i]*self.v
                    w_arr[2*self.Ny+1,0,:,:] = self.exp_jl[j,i]*self.Sp
                    w_arr[3*self.Ny,0,:,:] = self.exp_ju[j,i]*self.Sp
                    w_arr[3*self.Ny+1,0,:,:] = self.jl[j,i]*self.n
                    w_arr[4*self.Ny,0,:,:] = self.ju[j,i]*self.n
                    # Build generic interior
                    col_ind = 1
                    row_ind = 2
                    for k in range(4): # Because we have four operators?
                        for l in range(self.Ny-1):
                            w_arr[row_ind,col_ind,:,:] = self.I
                            col_ind += 1
                            row_ind += 1
                        col_ind += 1
                        row_ind += 1
                    # Build bottom row
                    w_arr[-1,self.Ny,:,:] = self.Sp
                    w_arr[-1,2*self.Ny,:,:] = -self.n
                    w_arr[-1,3*self.Ny,:,:] = self.Sm
                    w_arr[-1,4*self.Ny,:,:] = -self.v
                    w_arr[-1,4*self.Ny+1,:,:] = self.I
                    # Creation & Annihilation of Particles
                    w_arr[-1,0,:,:] += (self.exp_cr_r[j,i]+self.exp_cr_l[j,i]+self.exp_cr_d[j,i]+self.exp_cr_u[j,i])*self.Sm -\
                                       (self.cr_r[j,i]    +self.cr_l[j,i]    +self.cr_d[j,i]    +self.cr_u[j,i]    )*self.v  +\
                                       (self.exp_de_r[j,i]+self.exp_de_l[j,i]+self.exp_de_d[j,i]+self.exp_de_u[j,i])*self.Sp -\
                                       (self.de_r[j,i]    +self.de_l[j,i]    +self.de_d[j,i]    +self.de_u[j,i]    )*self.n
                    # Prevents interaction between ends
                    if (j is 0) and (i is not 0): 
                        w_arr[self.Ny,0,:,:] = self.z
                        w_arr[2*self.Ny,0,:,:] = self.z
                        w_arr[3*self.Ny,0,:,:] = self.z
                        w_arr[4*self.Ny,0,:,:] = self.z
                    # Add operator to list of ops (compress if on left or right edge)
                    if (i is 0) and (j is 0):
                        tmp_op.insert(len(tmp_op),np.expand_dims(w_arr[-1,:],0))
                    elif (i is self.Nx-1) and (j is self.Ny-1):
                        tmp_op.insert(len(tmp_op),np.expand_dims(w_arr[:,0],1))
                    else:
                        tmp_op.insert(len(tmp_op),w_arr)
            self.ops.insert(len(self.ops),tmp_op)
            # Build Two site terms
            coupled_sites = []
            # Determine periodic coupling along x-axis
            if self.periodic_x:
                if self.verbose > 2:
                    print('including periodicity in x-direction')
                for i in range(self.Ny):
                    coupled_sites.insert(0,[self.Nx*(i+1)-1,self.Nx*i,'horz'])
            # Determine periodic coupling along y-axis
            if self.periodic_y:
                if self.verbose > 2:
                    print('including periodicity in y-direction')
                for i in range(self.Nx):
                    coupled_sites.insert(0,[i,self.Nx*(self.Ny-1)+i,'vert'])
            # Build All Operators
            for i in range(len(coupled_sites)):
                inds = coupled_sites[i][:2]
                if coupled_sites[i][2] is 'horz':
                    # Convert to x,y coords
                    y_ind1 = int((inds[0]-(inds[0]%self.Nx))/self.Nx)
                    x_ind1 = self.Nx-1
                    y_ind2 = int(inds[1]/self.Nx)
                    x_ind2 = 0
                    if self.jr[y_ind1,x_ind1] != 0:
                        if self.verbose > 3:
                            print('Jump Right Terms:')
                            print('\t{}*Sm({})*Sp({})-{}v({})*n({})'.\
                                    format(self.exp_jr[y_ind1,x_ind1],inds[0],inds[1],self.jr[y_ind1,x_ind1],inds[0],inds[1]))
                        tmp_op1 = [None]*self.N
                        tmp_op1[inds[0]] = np.array([[self.exp_jr[y_ind1,x_ind1]*self.Sp]])
                        tmp_op1[inds[1]] = np.array([[self.Sm]])
                        tmp_op2 = [None]*self.N
                        tmp_op2[inds[0]] = np.array([[self.jr[y_ind1,x_ind1]*self.n]])
                        tmp_op2[inds[1]] = np.array([[-self.v]])
                        self.ops.insert(len(self.ops),tmp_op1)
                        self.ops.insert(len(self.ops),tmp_op2)
                    if self.jl[y_ind2,x_ind2] != 0:
                        if self.verbose > 3:
                            print('Jump Left Terms:')
                            print('\t{}*Sp({})*Sm({})-{}n({})*v({})'.\
                                    format(self.exp_jl[y_ind2,x_ind2],inds[0],inds[1],self.jl[y_ind2,x_ind2],inds[0],inds[1]))
                        tmp_op3 = [None]*self.N
                        tmp_op3[inds[0]] = np.array([[self.exp_jl[y_ind2,x_ind2]*self.Sm]])
                        tmp_op3[inds[1]] = np.array([[self.Sp]])
                        tmp_op4 = [None]*self.N
                        tmp_op4[inds[0]] = np.array([[self.jl[y_ind2,x_ind2]*self.v]])
                        tmp_op4[inds[1]] = np.array([[-self.n]])
                        self.ops.insert(len(self.ops),tmp_op3)
                        self.ops.insert(len(self.ops),tmp_op4)
                else:
                    # Convert to x,y coords
                    x_ind1 = inds[0]
                    y_ind1 = -1
                    x_ind2 = inds[0]
                    y_ind2 = 0
                    if self.jd[y_ind1,x_ind1] != 0:
                        if self.verbose > 3:
                            print('Jump Up Terms:')
                            print('\t{}*Sm({})*Sp({})-{}v({})*n({})'.\
                                format(self.exp_jd[y_ind1,x_ind1],inds[0],inds[1],self.jd[y_ind1,x_ind1],inds[0],inds[1]))
                        tmp_op1 = [None]*self.N
                        tmp_op1[inds[0]] = np.array([[self.exp_jd[y_ind1,x_ind1]*self.Sm]])
                        tmp_op1[inds[1]] = np.array([[self.Sp]])
                        tmp_op2 = [None]*self.N
                        tmp_op2[inds[0]] = np.array([[self.jd[y_ind1,x_ind1]*self.v]])
                        tmp_op2[inds[1]] = np.array([[-self.n]])
                        self.ops.insert(len(self.ops),tmp_op1)
                        self.ops.insert(len(self.ops),tmp_op2)
                    if self.ju[y_ind2,x_ind2] != 0:
                        print(self.ju)
                        print('{}{}'.format(y_ind2,x_ind2))
                        if self.verbose > 3:
                            print('Jump Down Terms:')
                            print('\t{}*Sp({})*Sm({})-{}n({})*v({})'.\
                                format(self.exp_ju[y_ind2,x_ind2],inds[0],inds[1],self.ju[y_ind2,x_ind2],inds[0],inds[1]))
                        tmp_op3 = [None]*self.N
                        tmp_op3[inds[0]] = np.array([[self.exp_ju[y_ind2,x_ind2]*self.Sp]])
                        tmp_op3[inds[1]] = np.array([[self.Sm]])
                        tmp_op4 = [None]*self.N
                        tmp_op4[inds[0]] = np.array([[self.ju[y_ind2,x_ind2]*self.n]])
                        tmp_op4[inds[1]] = np.array([[-self.v]])
                        self.ops.insert(len(self.ops),tmp_op3)
                        self.ops.insert(len(self.ops),tmp_op4)
            #for i in range(len(self.ops)):
            #    for j in range(len(self.ops[i])):
            #        if self.ops[i][j] is not None:
            #            self.ops[i][j] = -self.ops[i][j]
        elif hamType is "ising":
            self.J = param[0]
            self.h = param[1]
            w_arr = np.array([[self.I,          self.z,         self.z],
                              [self.Sz,         self.z,         self.z],
                              [self.h*self.Sz, self.J*self.Sz, self.I]])
            self.ops = []
            tmp_op = []
            tmp_op = [None]*self.N
            tmp_op[0] = np.expand_dims(w_arr[-1,:],0)
            for i in range(1,self.N-1):
                tmp_op[i] = w_arr
            tmp_op[-1] = np.expand_dims(w_arr[:,0],1)
            self.ops.insert(len(self.ops),tmp_op)
            # Include periodic terms
            if self.periodic_x:
                tmp_op1 = [None]*self.N
                tmp_op1[-1] = np.array([[self.Sz]])
                tmp_op1[0] = np.array([[self.J*self.Sz]])
                self.ops.insert(len(self.ops),tmp_op1)
        else:
            raise ValueError("Specified Hamiltonian type is not supported")
        self.nops = len(self.ops)
    
    def return_full_ham(self,verbose=2):
        # This function calculates the full hamiltonian matrix
        # As a warning, it is computationally expensive 
        H = np.zeros((self.N**2,self.N**2))
        if verbose > 0:
            print('Hamiltonian Size: {}'.format(H.shape))
        for i in range(self.N**2):
            if verbose > 1:
                print('\ti-Loop Progress: {}%'.format(i/self.N**2*100))
            i_occ = list(map(lambda x: int(x),'0'*(self.N-len(bin(i)[2:]))+bin(i)[2:]))
            for j in range(self.N**2):
                if verbose > 2:
                    print('\t\tj-Loop Progress: {}%'.format(j/self.N**2*100))
                j_occ = list(map(lambda x: int(x),'0'*(self.N-len(bin(j)[2:]))+bin(j)[2:]))
                tmp_mat = np.array([[1]])
                for k in range(self.N):
                    if verbose > 3:
                        print('\t\t\tk-Loop progress: {}%'.format(k/self.N*100))
                    tmp_mat = np.einsum('ij,jk->ik',tmp_mat,self.W[k][:,:,i_occ[k],j_occ[k]])
                H[i,j] = tmp_mat[[0]]
        return H
