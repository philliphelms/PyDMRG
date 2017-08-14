import numpy as np
from scipy.linalg import eig

class dmrg:
    
    def __init__(self):
        # Info from DMRG Class
        self.N = 6
        self.CurrentSite = 0
        self.Direction = false
        self.CurrentEnergy = 0
        self.MinEnergy = 0
        self.CurrentDm = new double[0]
        self.CurrentErw = 0
        self.CurrentEe = 0
        self.CurrentSweep = 0
        # Info from MPS Class
        self.states = 2
    
    def DoNextStep(self):
        if (self.CurrentSite == self.N-1):
            self.Direction = True;
        if (self.CurrentSite == 0):
            self.Direction = False;
            self.CurrentSweep += 1
        if self.Direction:
            self.DoLeftStep(self.CurrentSite)
            self.CurrentSite -= 1
            return self.CurrentSite+1
        else:
            self.DoRightStep(self.CurrentSite)
            self.CurrentSite += 1
            return self.CurrentSite-1
        
    def DoRightSweep(self):
        for j in range(self.N):
            this.DoRightStep(j)
    
    def DoLeftSweep(self):
        for j in range(self.N,0,-1):
            this.DoLeftStep(j)
         
    def DoRightStep(self,N):
        H = self.BuildHeff(N)
        eigenvalue,eigenvec = eig(H)
        for i in range(self.states):
            
    def DoLeftStep(self,site):
        
    def BuildHeff(self,N):
        
        