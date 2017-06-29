import numpy as np

class heis_mpo:
    def __init__(self,nsite):
        self.L = nsite
        self.d = 2 # Dimension of local state space
        self.h = 1 # First interaction parameter
        self.J = 1 # Second interaction parameter
        # Construct operator matrices - W
        self.Wint = np.array([[[[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,0],[0,0,0,-self.J/2,1]],
                               [[0,0,0,0,0],[1,0,0,0,0],[-1j,0,0,0,0],[0,0,0,0,0],[-self.h,-self.J/2,self.J/2*1j,0,0]]],
                              [[[0,0,0,0,0],[1,0,0,0,0],[1j,0,0,0,0],[0,0,0,0,0],[-self.h,-self.J/2,self.J/2*1j,0,0]],
                               [[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[-1,0,0,0,0],[0,0,0,self.J/2,1]]]])
        # Construct M matrices (General versions of A and B matrices)
        self.M = []
        for cnt in range(self.L):
            if cnt < nsite/2:
                self.M.append(np.zeros([2,self.d**cnt,self.d**(cnt+1)]))
            else:
                self.M.append(np.zeros([2,self.d**(self.L-cnt),self.d**(self.L-cnt-1)]))
            self.M[cnt][:,0,0] = 1
        # Construct L and R arrays, or left and right blocks in DMRG parlance
        self.R = [None]*(self.L+1)
        self.F = [None]*(self.L+1)

    def W(self,site,sigma,sigmap):
        if sigmap is -1:
            sigmap = 0
        if sigma is -1:
            sigmap = 0
        if site == 1:
           return self.Wint[sigma,sigmap,4,:]
        if site == self.nsite:
            return self.Wint[sigma,sigmap,:,0]
        else:
            return self.Wint[sigma,sigmap,:,:]

    def initiate_r(self):
        self.R[-1] = np.array([1])
        for site in range(self.L)[::-1]:
            if site == self.L-1:
                self.R[site] = np.einsum('ijo,ink,k,jpm->nop',self.Wint[:,:,:,0],self.M[site],self.R[site+1],self.M[site])
            if site == 0:
                self.R[site] = np.array([1])
            else:
                self.R[site] = np.einsum('ijlo,ink,klm,jpm->nop',self.Wint,self.M[site],self.R[site+1],self.M[site])

    def run_calc(self):
        # Initialize arrays in R block
        self.initiate_r()
        # Overall Loop
        converged = False
        while not converged:
            # Sweep Right
            for l in range(self.L-1):
                # Solve eigenproblem for M(l)
                np.einsum('',)
                # Left-normalize M into A by SVD
                # Multiply remaining SVD Matrices into M(l+1)
                # Build the L expression by adding one more site
            # Sweep Left
            for l in range(1,L)[::-1]:
                # Solve eigenproblem for M(l)
                # Right-normalize M(l) into B by SVD
                # Multiply the remaining SVD Matrices into M(l-1)
                # Build the R expression by adding one more site
            # Check Convergence
            if (True):
                converged = True

if __name__ == "__main__":
    x = heis_mpo(4)
    x.run_calc()
