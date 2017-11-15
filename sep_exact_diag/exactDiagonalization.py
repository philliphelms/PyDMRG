import numpy as np
from scipy.linalg import eig as fullEig

class exactDiag:

    def __init__(self, L=10, alpha=0.35, gamma=0,
                 beta=2/3, delta=0, s=-1, p=1, q=0,
                 clumpSize=10, maxIter=100):
        self.L = L
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.s = s
        self.p = p
        self.q = q
        self.clumpSize = clumpSize
        self.maxIter = maxIter

    def kernel(self):
        # Extract info
        L = self.L
        a = self.alpha
        g = self.gamma
        b = self.beta
        d = self.delta
        s = self.s
        p = self.p
        q = self.q
        clumpSize = self.clumpSize

        # Currents
        pw = p*np.exp(-s)
        qw = q*np.exp(-s)
        aw = a*np.exp(-s)
        bw = b*np.exp(-s)
        gw = g*np.exp(-s)
        dw = d*np.exp(-s)

        # Some Containers
        sproj = np.zeros((L/clumpSize, 2**clumpSize))
        isproj = np.zeros((L/clumpSize, 2**clumpSize))

        # Create Initial Guess
        nv = 0.1*np.ones(L)
        cdv = 0.1*np.ones(L)
        cv = 0.1*np.ones(L)

        # Create Operators
        m1 = np.array([[-a,gw],[aw,-g]])
        m1 = np.kron(m1,np.eye(2**(clumpSize-1)))

        mL = np.array([[-d,bw],[dw,-b]])
        mL = np.kron(np.eye(2**(clumpSize-1)),mL)

        mi = np.array([[0,0, 0, 0],
                       [0,-q,pw,0],
                       [0,qw,-p,0],
                       [0,0, 0, 0]])

        if clumpSize > 2:
            mc = np.zeros((2**clumpSize,2**clumpSize))
            for i in range(clumpSize-1):
                left = i-1
                right = clumpSize-(i+1)
                if left is 0:
                    mc += np.kron(mi,np.eye(2**right))
                elif right is 0:
                    mc += np.kron(np.eye(2**left),mi)
                else:
                    mc += kron(kron(np.eye(2**left),mi),np.eye(2**right))
        else:
            mc = mi

        # Individual Observables
        cop = np.zeros((2**clumpSize,2**clumpSize,clumpSize))
        cdop = np.zeros((2**clumpSize,2**clumpSize,clumpSize))
        nop = np.zeros((2**clumpSize,2**clumpSize,clumpSize))
        for i in range(clumpSize):
            im = i
            ip = clumpSize-(i+1)
            if im is 0:
                cop[:,:,i] = np.kron(np.array([[0,1],[0,0]]),np.eye(2**ip))
                cdop[:,:,i] = np.kron(np.array([[0,0],[1,0]]),np.eye(2**ip))
                nop[:,:,i] = np.kron(np.array([[0,0],[0,1]]),np.eye(2**ip))
            elif ip is clumpSize-1:
                cop[:,:,i] = np.kron(np.eye(2**im),np.array([[0,1],[0,0]]))
                cdop[:,:,i] = np.kron(np.eye(2**im),np.array([[0,0],[1,0]]))
                nop[:,:,i] = np.kron(np.eye(2**im),np.array([[0,0],[0,1]]))
            else:
                cop[:,:,i] = np.kron(np.kron(np.eye(2**im),np.array([[0,1],[0,0]])),np.eye(2**ip))
                cdop[:,:,i] = np.kron(np.kron(np.eye(2**im),np.array([[0,0],[1,0]])),np.eye(2**ip))
                nop[:,:,i] = np.kron(np.kron(np.eye(2**im),np.array([[0,0],[0,1]])),np.eye(2**ip))

        if L is clumpSize: self.maxIter = 1

        for iter in range(self.maxIter):
            print("Iteration {}".format(iter))
            nv_new = np.zeros(L)
            cv_new = np.zeros(L)
            cdv_new = np.zeros(L)
            lam = np.zeros((int(L/clumpSize),2**clumpSize))
            Ms = np.zeros((2**clumpSize,2**clumpSize,L/clumpSize))
            for clump in range(int(L/clumpSize)):
                leftClump = (clump)*clumpSize
                rightClump = (clump)*clumpSize+clumpSize-1
                # Couple left side
                if clump is 0:
                    leftSide = m1
                else:
                    leftSide = np.array([[-p*nv[leftClump],qw*cdv[leftClump]],
                                         [pw*cv[leftClump],-q*(1-nv[leftClump])]])
                # Couple Right Side
                rightSide = np.zeros((2,2))
                if clump is int(L/clumpSize-1):
                    rightSide = mL
                else:
                    rightSide = np.array([[-q*nv[rightClump+1],pw*cdv[rightClump+1]],
                                          [qw*cv[rightClump+1],-p*(1-nv[rightClump+1])]])
                    rightSide = np.kron(np.eye(2**(clumpSize-1)),rightSide)
                # Create Main Matrix
                print(mc)
                print(leftSide)
                print(rightSide)
                M = mc+leftSide+rightSide
                # Diagonalize Main Matrix
                u,v = fullEig(M)
                # Select largest eigenvalue
                v = v[:,np.argsort(u)]
                u = np.sort(u)
                # Check to see which one has a low imaginary value
                ind = -1
                for i in range(len(u)):
                    if np.abs(np.imag(u[i])) < 1e-8:
                        ind = i
                        break
                u = u[ind]
                lam[clump,0] = u
                # Calculate Expectation Values
                iv = np.linalg.inv(v)
                lpsi = iv[ind,:]
                rpsi = v[:,ind]
                for s in range(clumpSize):
                    nv_new[(clump)*clumpSize+s] = np.dot(lpsi,np.dot(nop[:,:,s],rpsi))
                    cv_new[(clump)*clumpSize+s] = np.dot(lpsi,np.dot(cop[:,:,s],rpsi))
                    cdv_new[(clump)*clumpSize+s] = np.dot(lpsi,np.dot(cdop[:,:,s],rpsi))
                sproj[clump,:] = v[:,ind]
                isproj[clump,:] = iv[ind,:]
            # Update Values
            nv = nv_new
            cv = cv_new
            cdv = cdv_new
        # Calculate Energy
        # Intra Clump
        E = 0
        for clump in range(int(L/clumpSize)):
            lpsi = isproj[clump,:]
            rpsi = sproj[clump,:]
            m = mc
            if clump is 0:
                m += m1
            if clump is L/clumpSize:
                m += mL
            E += np.dot(lpsi,np.dot(m,rpsi))
        # Inter Clump
        if clumpSize is not L:
            mi_x = np.kron(np.eye(2**(clumpSize-1)),mi)
            mi_x = np.kron(mi_x,np.eye(2**clumpSize-1))
            for clump in range(L/clumpSize-1):
                lpsi = np.kron(isproj[clump,:],isproj[clump+1,:])
                rpsi = np.kron(sproj[clump,:],sproj[clump+1,:]).transpose()
                E += np.dot(lpsi,np.dot(mi_x,rpsi))
        return E

if __name__ == "__main__":
    x = exactDiag(L=2,clumpSize=2)
    print(x.kernel())
