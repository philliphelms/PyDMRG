import numpy as np

class heis_mpo:
    def __init__(self,nsite):
        self.nsite = nsite
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
        for cnt in range(self.nsite):
            if cnt < nsite/2:
                self.M.append(np.zeros([2,self.d**cnt,self.d**(cnt+1)]))
            else:
                self.M.append(np.zeros([2,self.d**(self.nsite-cnt),self.d**(self.nsite-cnt-1)]))
            self.M[cnt][:,0,0] = 1
        # Construct L and R arrays, or left and right blocks in DMRG parlance
        self.R = [None]*(self.nsite+1)
        self.L = [None]*(self.nsite+1)
        self.L[0] = np.array([[1]])

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
    
    def shape_hamiltonian(self,H):
        sl,alm,al,slp,almp,alp = H.shape
        return H.reshape(sl*alm*al,-1)

    def shape_m(self,site,v):
        self.M[site]= v.reshape(self.M[site].shape)

    def initiate_r(self):
        self.R[-1] = np.array([[1]])
        for site in range(self.nsite)[::-1]:
            if site == self.nsite-1:
                self.R[site] = np.einsum('ijo,ink,km,jpm->nop',self.Wint[:,:,:,0],self.M[site],self.R[site+1],self.M[site])
            elif site == 0:
                self.R[site] = np.array([[1]])
            else:
                self.R[site] = np.einsum('ijlo,ink,klm,jpm->nop',self.Wint,self.M[site],self.R[site+1],self.M[site])

    def run_calc(self):
        # Initialize arrays in R block
        self.initiate_r()
        # Overall Loop
        converged = False
        while not converged:
            # Sweep Right
            for l in range(self.nsite-1):
                print(l)
                # Solve eigenproblem for M(l) (This should be replaced by iterative solver)
                if l == 0:
                    H = np.einsum('ik,lmn,onp->liomkp',self.L[l],self.Wint[:,:,-1,:],self.R[l+1])
                else:
                    H = np.einsum('ijk,lmjn,onp->liomkp',self.L[l],self.Wint,self.R[l+1])
                H = self.shape_hamiltonian(H)
                w,v = np.linalg.eig(H)
                w = np.sort(w)
                v = v[:,w.argsort()]
                self.shape_m(l,v[:,0])
                # Left-normalize M into A by SVD
                (U,S,V) = np.linalg.svd(self.M[l].reshape(-1,self.M[l][0,0,:].size))
                self.M[l][0,:,:] = U[0,:]
                self.M[l][1,:,:] = U[1,:]
                # Multiply remaining SVD Matrices into M(l+1)
                self.M[l+1] = np.einsum('i,ij,ljk->lik',S,V,self.M[l+1])
                # Build the L expression by adding one more site
                if l == 0:
                    self.L[l+1] = np.einsum('ijo,ink,km,jpm->nop',self.Wint[:,:,-1,:],self.M[l+1],self.L[l],self.M[l+1])
                else:
                    self.L[l+1] = np.einsum('ijlo,ink,klm,jpm->nop',self.Wint,self.M[l+1],self.L[l],self.M[l+1])
            # Sweep Left
            for l in range(1,self.nsite)[::-1]:
                # Solve eigenproblem for M(l)
                H = np.einsum('ijk,lmjn,opn->liomkp',self.L[site],self.Wint[site],self.R[site])
                # Right-normalize M(l) into B by SVD
                # Multiply the remaining SVD Matrices into M(l-1)
                # Build the R expression by adding one more site
            # Check Convergence
            if (True):
                converged = True

if __name__ == "__main__":
    x = heis_mpo(4)
    x.run_calc()
