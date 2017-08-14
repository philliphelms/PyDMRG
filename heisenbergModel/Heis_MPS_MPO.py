import numpy as np
from HeisMPO import *
from HeisMPS import *
from HeisDMRG import *
import time

class Heis_MPS_MPO:
    """
    Description:
        An object containing all information essential to run the DMRG calculation
        finding the ground state of the Heisenberg model for a 1D chain of length L.
        
    Class Members:
        > self.L               - The number of sites for the system
        > self.h               - The strength of the orienting force
        > self.J               - The interaction strenght between neighboring spins
        > self.d               - The local state-space dimension
        > self.D               - The cut-off for the state-space dimension of each 
                                 matrix product state. This limits the size of the 
                                 calculations using the Schmidt Decomposition.
        > self.init_guess_type - Specify the method of generating the initial guess
                                 options include: ground state ('gs'), hartree-fock
                                 ('hf') or random ('rand')
        > self.tol             - Tolerance for energy convergence criteria
        > self.max_sweep_cnt   - The maximum number of sweeps to be performed before
                                 cancelling the calculation
        > self.reshape_order   - The ordering for reshaping of matrices, should always
                                 be set at "F", indicating Fortran ordering.
        > self.mpo             - The heisenberg model matrix product operator object
        > self.mps             - The heisenberg model matrix product state object
        > self.plot_option     - If true, then generate matplotlib plots showing convergence.
                                 Note that the plotting slows calculations significantly.
        > self.plot_cnt        - Keeps track of the number of times the plot's been updated
    
    Key Functions:
        1) Heis_MPS_MPO(L)     - Create the Heis_MPS_MPO object with the length of
                                 the 1D chain being L.
        2) check_params()      - Ensures all parameters are correctly input. Main
                                 purpose is to ensure D is a power of d, fixes this
                                 problem if it is not.
        3) calc_ground_state() - A function that uses the MPS formulation of the 
                                 DMRG algorithm to calculate the ground state of 
                                 the given system. Requires no inputs besides the 
                                 Heis_MPS_MPO object.
    """
    def __init__(self,L):
        self.L = L
        self.h = 0
        self.J = 1
        self.d = 2
        self.D = 8
        self.init_guess_type = 'default' # (default, gs, hf, rand)
        self.tol = 1e-3
        self.max_sweep_cnt = 3
        self.reshape_order = "C" 
        self.plot_option = True
    
    def check_params(self):
        waiting = True
        pwr = 0
        while waiting:
            raised = self.d**pwr
            if raised > self.D:
                self.D = self.d**(pwr-1)
                waiting = False
            else:
                pwr += 1
        
    def calc_ground_state(self):
        self.check_params()
        self.mpo = HeisMPO(self.h,self.J,self.L)
        self.mps = HeisMPS(self.L,self.init_guess_type,self.d,self.reshape_order,self.D)
        self.mps.create_initial_guess()
        self.mps.initialize_r(self.mpo.W)
        self.dmrg = HeisDMRG(self.mpo,self.mps,self.tol,
                             self.max_sweep_cnt,
                             self.reshape_order,self.plot_option)
        self.dmrg.run_optimization()
    
if __name__ == "__main__":
    t0 = time.time()
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    x = Heis_MPS_MPO(20)
    x.calc_ground_state()
    # x.mps.write_all_c('heisenberg_occupation_calc.xlsx')
    t1 = time.time()
    print(('#'*75+'\nTotal Time: {}\n'+'#'*75).format(t1-t0))
