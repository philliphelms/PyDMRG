import numpy as np
from HeisMPO import *
from HeisMPS import *
from HeisDMRG import *

class Heis_MPS_MPO:

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
        self.mps = HeisMPS(self.L,self.init_guess_type,self.d,self.reshape_order)
        
    def calc_ground_state(self):
        self.mps.create_initial_guess()
        self.mps.initialize_r(self.mpo.W)
        self.dmrg = HeisDMRG(self.mpo,self.mps,self.D,self.tol,self.max_sweep_cnt,self.reshape_order)
        self.dmrg.run_optimization()
    
if __name__ == "__main__":
    x = Heis_MPS_MPO(4)
    x.calc_ground_state()