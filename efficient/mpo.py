import numpy as np

class MPO:

    def __init__(self, hamType, param, N,periodic_x=False,periodic_y=False,verbose=1):
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
            self.ops = []
            # Add two site terms
            if self.J != 0:
                for i in range(self.N-1):
                    tmp_op1 = [None]*self.N
                    tmp_op2 = [None]*self.N
                    tmp_op3 = [None]*self.N
                    tmp_op1[i] = self.J/2*self.Sm
                    tmp_op2[i] = self.J/2*self.Sp
                    tmp_op3[i] = self.J*self.Sz
                    tmp_op1[i+1] = self.Sp
                    tmp_op2[i+1] = self.Sm
                    tmp_op3[i+1] = self.Sz
                    self.ops.insert(len(self.ops),tmp_op1)
                    self.ops.insert(len(self.ops),tmp_op2)
                    self.ops.insert(len(self.ops),tmp_op3)
                # Include periodic terms
                if self.periodic_x:
                    tmp_op1 = [None]*self.N
                    tmp_op2 = [None]*self.N
                    tmp_op3 = [None]*self.N
                    tmp_op1[-1] = self.J/2*self.Sm
                    tmp_op2[-1] = self.J/2*self.Sp
                    tmp_op3[-1] = self.J*self.Sz
                    tmp_op1[0] = self.Sp
                    tmp_op2[0] = self.Sm
                    tmp_op3[0] = self.Sz
                    self.ops.insert(len(self.ops),tmp_op1)
                    self.ops.insert(len(self.ops),tmp_op2)
                    self.ops.insert(len(self.ops),tmp_op3)
            # Add one site terms
            if self.h != 0:
                for i in range(self.N):
                    tmp_op1 = []
                    for j in range(self.N):
                        if i == j:
                            tmp_op1.insert(len(tmp_op1),-self.h*self.Sz)
                        else:
                            tmp_op1.insert(len(tmp_op1),None)
                    self.ops.insert(len(self.ops),tmp_op1)
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
                    tmp_op1[inds[0]] = self.J/2*self.Sm
                    tmp_op2[inds[0]] = self.J/2*self.Sp
                    tmp_op3[inds[0]] = self.J*self.Sz
                    tmp_op1[inds[1]] = self.Sp
                    tmp_op2[inds[1]] = self.Sm
                    tmp_op3[inds[1]] = self.Sz
                    self.ops.insert(len(self.ops),tmp_op1)
                    self.ops.insert(len(self.ops),tmp_op2)
                    self.ops.insert(len(self.ops),tmp_op3)
            # Add one site terms
            if self.h != 0:
                for i in range(self.N):
                    tmp_op1 = []
                    for j in range(self.N):
                        if i == j:
                            tmp_op1.insert(len(tmp_op1),-self.h*self.Sz)
                        else:
                            tmp_op1.insert(len(tmp_op1),None)
                    self.ops.insert(len(self.ops),tmp_op1)
        elif hamType is "tasep":
            self.alpha = param[0]
            self.s = param[1]
            self.beta = param[2]
            self.ops = []
            for i in range(self.N-1):
                tmp_op1 = [None]*self.N
                tmp_op2 = [None]*self.N
                tmp_op1[i] = np.exp(-self.s)*self.Sp
                tmp_op2[i] = -self.n
                tmp_op1[i+1] = self.Sm
                tmp_op2[i+1] = self.v
                self.ops.insert(len(self.ops),tmp_op1)
                self.ops.insert(len(self.ops),tmp_op2)
            # Include periodic terms
            if self.periodic_x:
                tmp_op1 = [None]*self.N
                tmp_op2 = [None]*self.N
                tmp_op1[-1] = np.exp(-self.s)*self.Sp
                tmp_op2[-1] = -self.n
                tmp_op1[0] = self.Sm
                tmp_op2[0] = self.v
                self.ops.insert(len(self.ops),tmp_op1)
                self.ops.insert(len(self.ops),tmp_op2)
            else:
                tmp_op1 = [None]*self.N
                tmp_op2 = [None]*self.N
                tmp_op1[0] = self.alpha*(np.exp(-self.s)*self.Sm-self.v)
                tmp_op2[-1] = self.beta*(np.exp(-self.s)*self.Sp-self.n)
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
            # Create List of All Operators
            self.ops = []
            for i in range(self.N-1):
                if self.p != 0:
                    tmp_op1 = [None]*self.N
                    tmp_op2 = [None]*self.N
                    tmp_op1[i] = self.Sp
                    tmp_op2[i] = -self.n
                    tmp_op1[i+1] = self.exp_p*self.Sm
                    tmp_op2[i+1] = self.p*self.v
                    self.ops.insert(len(self.ops),tmp_op1)
                    self.ops.insert(len(self.ops),tmp_op2)
                if self.q != 0:
                    tmp_op3 = [None]*self.N
                    tmp_op4 = [None]*self.N
                    tmp_op3[i] = self.Sm
                    tmp_op4[i] = -self.v
                    tmp_op3[i+1] = self.exp_q*self.Sp
                    tmp_op4[i+1] = self.q*self.n
                    self.ops.insert(len(self.ops),tmp_op3)
                    self.ops.insert(len(self.ops),tmp_op4)
            # Include periodic terms
            if self.periodic_x:
                if self.p != 0:
                    tmp_op1 = [None]*self.N
                    tmp_op2 = [None]*self.N
                    tmp_op1[-1] = self.Sp
                    tmp_op2[-1] = -self.n
                    tmp_op1[0] = self.exp_p*self.Sm
                    tmp_op2[0] = self.p*self.v
                    self.ops.insert(len(self.ops),tmp_op1)
                    self.ops.insert(len(self.ops),tmp_op2)
                if self.q != 0:
                    tmp_op1 = [None]*self.N
                    tmp_op2 = [None]*self.N
                    tmp_op1[-1] = self.Sm
                    tmp_op2[-1] = -self.v
                    tmp_op1[0] = np.exp_q*self.Sp
                    tmp_op2[0] = self.q*self.n
                    self.ops.insert(len(self.ops),tmp_op1)
                    self.ops.insert(len(self.ops),tmp_op2)
            else:
                tmp_op1 = [None]*self.N
                tmp_op2 = [None]*self.N
                tmp_op1[0] = (self.exp_alpha*self.Sm-self.alpha*self.v)+(self.exp_gamma*self.Sp-self.gamma*self.n)
                tmp_op2[-1] = (self.exp_beta*self.Sm-self.beta*self.v)+(self.exp_delta*self.Sp-self.delta*self.n)
                self.ops.insert(len(self.ops),tmp_op1)
                self.ops.insert(len(self.ops),tmp_op2)
        elif hamType is "sep_2d":
            # Collect Parameters
            self.Nx = self.N[0]
            self.Ny = self.N[1]
            self.N = self.Nx*self.Ny
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
            # Multiply params by an exponential
            self.exp_jl = self.jl*np.exp(self.s)  # Moving Left
            self.exp_jr = self.jr*np.exp(-self.s) # Moving Right
            self.exp_il = self.il*np.exp(-self.s) # Moving Right
            self.exp_ir = self.ir*np.exp(self.s)  # Moving Left
            self.exp_ol = self.ol*np.exp(self.s)  # Moving Left
            self.exp_or = self.outr*np.exp(-self.s)# Moving Right
            self.exp_ju = self.ju*np.exp(self.s)   # Moving up
            self.exp_jd = self.jd*np.exp(-self.s)  # Moving Down
            self.exp_it = self.it*np.exp(-self.s)  # Moving Down
            self.exp_ib = self.ib*np.exp(self.s)   # Moving Up
            self.exp_ot = self.ot*np.exp(self.s)   # Moving Up
            self.exp_ob = self.ob*np.exp(-self.s)  # Moving Down
            # Build Two site terms
            self.ops = []
            coupled_sites = []
            # Determine all coupled sites along x-axis
            for i in range(self.Ny):
                for j in range(self.Nx-1):
                    coupled_sites.insert(0,[j+self.Nx*(i),j+1+self.Nx*(i),'horz'])
            # Determine all coupled sites along y-axis
            for i in range(self.Nx):
                for j in range(self.Ny-1):
                    coupled_sites.insert(0,[i+self.Nx*(j),i+self.Nx*(j+1),'vert'])
            # Determine periodic coupling along x-axis
            if self.periodic_x:
                if self.verbose > 2:
                    print('including periodicity in x-direction')
                for i in range(Ny):
                    coupled_sites.insert(0,[Nx*(i+1)-1,Nx*i,'horz'])
            # Determine periodic coupling along y-axis
            if self.periodic_y:
                if self.verbose > 2:
                    print('including periodicity in y-direction')
                for i in range(Nx):
                    coupled_sites.insert(0,[Nx*(Ny-1)+i,i],'vert')
            # Build All two-site Operators
            for i in range(len(coupled_sites)):
                inds = coupled_sites[i][:2]
                if coupled_sites[i][2] is 'horz':
                    if self.jr != 0:
                        if self.verbose > 3:
                            print('Jump Right Terms:')
                            print('\t{}*Sm({})*Sp({})-{}v({})*n({})'.\
                                    format(self.exp_jr,inds[0],inds[1],self.jr,inds[0],inds[1]))
                        tmp_op1 = [None]*self.N
                        tmp_op1[inds[0]] = self.exp_jr*self.Sp
                        tmp_op1[inds[1]] = self.Sm
                        tmp_op2 = [None]*self.N
                        tmp_op2[inds[0]] = self.jr*self.n
                        tmp_op2[inds[1]] = -self.v
                        self.ops.insert(len(self.ops),tmp_op1)
                        self.ops.insert(len(self.ops),tmp_op2)
                    if self.jl != 0:
                        if self.verbose > 3:
                            print('Jump Left Terms:')
                            print('\t{}*Sp({})*Sm({})-{}n({})*v({})'.\
                                    format(self.exp_jl,inds[0],inds[1],self.jl,inds[0],inds[1]))
                        tmp_op3 = [None]*self.N
                        tmp_op3[inds[0]] = self.exp_jl*self.Sm
                        tmp_op3[inds[1]] = self.Sp
                        tmp_op4 = [None]*self.N
                        tmp_op4[inds[0]] = self.jl*self.v
                        tmp_op4[inds[1]] = -self.n
                        self.ops.insert(len(self.ops),tmp_op3)
                        self.ops.insert(len(self.ops),tmp_op4)
                else:
                    if self.ju != 0:
                        if self.verbose > 3:
                            print('Jump Up Terms:')
                            print('\t{}*Sm({})*Sp({})-{}v({})*n({})'.\
                                format(self.exp_ju,inds[0],inds[1],self.ju,inds[0],inds[1]))
                        tmp_op1 = [None]*self.N
                        tmp_op1[inds[0]] = self.exp_ju*self.Sm
                        tmp_op1[inds[1]] = self.Sp
                        tmp_op2 = [None]*self.N
                        tmp_op2[inds[0]] = self.ju*self.v
                        tmp_op2[inds[1]] = -self.n
                        self.ops.insert(len(self.ops),tmp_op1)
                        self.ops.insert(len(self.ops),tmp_op2)
                    if self.jd != 0:
                        if self.verbose > 3:
                            print('Jump Down Terms:')
                            print('\t{}*Sp({})*Sm({})-{}n({})*v({})'.\
                                format(self.exp_jd,inds[0],inds[1],self.jd,inds[0],inds[1]))
                        tmp_op3 = [None]*self.N
                        tmp_op3[inds[0]] = self.exp_jd*self.Sp
                        tmp_op3[inds[1]] = self.Sm
                        tmp_op4 = [None]*self.N
                        tmp_op4[inds[0]] = self.jd*self.n
                        tmp_op4[inds[1]] = -self.v
                        self.ops.insert(len(self.ops),tmp_op3)
                        self.ops.insert(len(self.ops),tmp_op4)
            if not self.periodic_x:
                if self.verbose > 2:
                    print('Adding x boundary conditions')
                for i in range(self.Ny):
                    if self.il != 0 or self.ol != 0:
                        if self.verbose > 3:
                            print('\t{}*Sm({})-{}*v({})+{}*Sp({})-{}*n({})'.\
                                format(self.exp_il,self.Nx*i,self.il,self.Nx*i,self.exp_ol,self.Nx*i,self.ol,self.Nx*i))
                        tmp_op1 = [None]*self.N
                        tmp_op1[self.Nx*i] = self.exp_il*self.Sm-self.il*self.v + self.exp_ol*self.Sp-self.ol*self.n
                        self.ops.insert(len(self.ops),tmp_op1)
                    if self.ir != 0 or self.outr != 0:
                        if self.verbose > 3:
                            print('\t{}*Sm({})-{}*v({})+{}*Sp({})-{}*n({})'.\
                              format(self.exp_ir,self.Nx*(i+1)-1,self.ir,self.Nx*(i+1)-1,self.exp_or,self.Nx*(i+1)-1,self.outr,self.Nx*(i+1)-1))
                        tmp_op2 = [None]*self.N
                        tmp_op2[self.Nx*(i+1)-1] = self.exp_ir*self.Sm-self.ir*self.v + self.exp_or*self.Sp-self.outr*self.n
                        self.ops.insert(len(self.ops),tmp_op2)
            if not self.periodic_y:
                for i in range(self.Nx):
                    if self.it != 0 or self.ot != 0:
                        if self.verbose > 3:
                            print('\t{}*Sm({})-{}*v({})+{}*Sp({})-{}*n({})'.\
                              format(self.exp_it,i,self.it,i,self.exp_ot,i,self.ot,i))
                        tmp_op1 = [None]*self.N
                        tmp_op1[i] = self.exp_it*self.Sm-self.it*self.v + self.exp_ot*self.Sp-self.ot*self.n
                        self.ops.insert(len(self.ops),tmp_op1)
                    if self.ib != 0 or self.ob != 0:
                        if self.verbose > 3:
                            print('\t{}*Sm({})-{}*v({})+{}*Sp({})-{}*n({})'.\
                                format(self.exp_it,(self.Ny-1)*self.Nx+i,self.it,(self.Ny-1)*self.Nx+i,\
                                       self.exp_ot,(self.Ny-1)*self.Nx+i,self.ot,(self.Ny-1)*self.Nx+i))
                        tmp_op2 = [None]*self.N
                        tmp_op2[(self.Ny-1)*self.Nx+i] = self.exp_ib*self.Sm-self.ib*self.v + self.exp_ob*self.Sp-self.ob*self.n
                        self.ops.insert(len(self.ops),tmp_op2)
        elif hamType is "ising":
            self.J = param[0]
            self.h = param[1]
            self.ops = []
            # Add two site terms
            if self.J != 0:
                for i in range(self.N-1):
                    tmp_op1 = [None]*self.N
                    tmp_op1[i] = self.Sz
                    tmp_op1[i+1] = self.J*self.Sz
                    self.ops.insert(len(self.ops),tmp_op1)
                # Include periodic terms
                if self.periodic_x:
                    tmp_op1 = [None]*self.N
                    tmp_op1[-1] = self.Sz
                    tmp_op1[0] = self.J*self.Sz
                    self.ops.insert(len(self.ops),tmp_op1)
            # Add one site terms
            if self.h != 0:
                for i in range(self.N):
                    tmp_op1 = [None]*self.N
                    tmp_op1[i] = -self.h*self.Sz
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
