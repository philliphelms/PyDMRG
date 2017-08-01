import numpy as np
from HeisMPO import *
from HeisMPS import *
from HeisDMRG import *

class Heis_MPS_MPO:
    # A class that contains the MPS and MPO objects that can be used in the 
    # DMRG Calculations
    def __init__(self,L):
        # Basic Model Information
        self.L = L
        self.h = 1
        self.J = 1
        self.d = 2
        self.D = 100
        # Optimization Parameters
        self.init_guess_type = 'eye' # (gs, hf, eye, rand)
        self.tol = 1e-5
        self.max_sweep_cnt = 10
        self.verbose = 4 # 0,1,2,3,4
        self.plot_option = True
        self.plot_cnt = 0
        np.set_printoptions(precision=2) # Precision of verbose printing
        # Create MPO Object
        self.mpo = HeisMPO(self.h,self.J,self.L)
        # Create MPS Object
        self.mps = HeisMPS(self.L,self.init_guess_type,self.d,self.verbose)
        
    def calc_ground_state(self):
        self.mps.create_initial_guess()
        self.mps.initialize_r(self.mpo.W)
        self.dmrg = HeisDMRG(self.mpo,self.mps,self.D,self.tol,
                             self.max_sweep_cnt,self.verbose,
                             self.plot_option,self.plot_cnt)
        self.dmrg.run_optimization()
    
if __name__ == "__main__":
    x = Heis_MPS_MPO(4)
    x.calc_ground_state()