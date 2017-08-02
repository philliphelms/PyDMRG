import numpy as np
from HeisMPO import *
from HeisMPS import *
from HeisDMRG import *

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
    
    Key Functions:
        1) Heis_MPS_MPO(L)     - Create the Heis_MPS_MPO object with the length of
                                 the 1D chain being L.
        2) calc_ground_state() - A function that uses the MPS formulation of the 
                                 DMRG algorithm to calculate the ground state of 
                                 the given system. Requires no inputs besides the 
                                 Heis_MPS_MPO object.
    """
    def __init__(self,L):
        self.L = L
        self.h = 0
        self.J = 1
        self.d = 2
        self.D = 100
        self.init_guess_type = 'rand' # (gs, hf, eye, rand)
        self.tol = 1e-5
        self.max_sweep_cnt = 10
        self.reshape_order = "F" 
        self.mpo = HeisMPO(self.h,self.J,self.L)
        self.mps = HeisMPS(self.L,self.init_guess_type,self.d,self.reshape_order,self.D)
        
    def calc_ground_state(self):
        self.mps.create_initial_guess()
        self.mps.initialize_r(self.mpo.W)
        self.dmrg = HeisDMRG(self.mpo,self.mps,self.tol,self.max_sweep_cnt,self.reshape_order)
        self.dmrg.run_optimization()
    
if __name__ == "__main__":
    x = Heis_MPS_MPO(10)
    x.calc_ground_state()