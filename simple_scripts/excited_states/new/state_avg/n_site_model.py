import numpy as np
import scipy.linalg as la

# To Do :
# - Take Energy from Center Site
# - Calculate observables (EE) after each state has converged
# - Return Gap
# - Left Eigenvectors

def create_sep_mpo(N,hamParams):
    # Unpack Ham Params ################################
    a = hamParams[0]
    g = hamParams[1]
    p = hamParams[2]
    q = hamParams[3]
    b = hamParams[4]
    d = hamParams[5]
    s = hamParams[6]
    exp_a = a*np.exp(-s)
    exp_g = g*np.exp(s)
    exp_p = p*np.exp(-s)
    exp_q = q*np.exp(s)
    exp_b = b*np.exp(-s)
    exp_d = d*np.exp(s)
    # Create MPO #######################################
    Sp = np.array([[0,1],[0,0]])
    Sm = np.array([[0,0],[1,0]])
    n = np.array([[0,0],[0,1]])
    v = np.array([[1,0],[0,0]])
    I = np.array([[1,0],[0,1]])
    z = np.array([[0,0],[0,0]])
    const = 1.
    W = []
    W.append(const*np.array([[exp_a*Sm-a*v+exp_g*Sp-g*n, Sp, -n, Sm, -v, I]]))
    for i in range(N-2):
        W.append(const*np.array([[I       ,  z,  z,  z,  z,  z],
                           [exp_p*Sm,  z,  z,  z,  z,  z],
                           [p*v     ,  z,  z,  z,  z,  z],
                           [exp_q*Sp,  z,  z,  z,  z,  z],
                           [q*n     ,  z,  z,  z,  z,  z],
                           [z       , Sp, -n, Sm, -v,  I]]))
    W.append(const*np.array([[I],[exp_p*Sm],[p*v],[exp_q*Sp],[q*n],[exp_d*Sm-d*v+exp_b*Sp-b*n]]))
    return W

def create_rand_mps(N,mbd):
    # Create MPS
    M = []
    for i in range(int(N/2)):
        M.insert(len(M),np.random.rand(2,min(2**(i),mbd),min(2**(i+1),mbd)))
    for i in range(int(N/2))[::-1]:
        M.insert(len(M),np.random.rand(2,min(2**(i+1),mbd),min(2**i,mbd)))
    return M

def load_mps(N,fname):
    npzfile = np.load(fname)
    M = []
    for i in range(N):
        M.append(npzfile['M'+str(i)])
    return M

def make_mps_right(M):
    N = len(M)
    for i in range(int(N)-1,0,-1):
        M_reshape = np.swapaxes(M[i],0,1)
        (n1,n2,n3) = M_reshape.shape
        M_reshape = np.reshape(M_reshape,(n1,n2*n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M_reshape = np.reshape(V,(n1,n2,n3))
        M[i] = np.swapaxes(M_reshape,0,1)
        M[i-1] = np.einsum('klj,ji,i->kli',M[i-1],U,s)
    return M

def calc_env(M,W,mbd):
    N = len(M)
    _,mbdW,_,_ = W[0].shape
    # Initialize Empty F
    F = []
    F.insert(len(F),np.array([[[1]]]))
    for i in range(int(N/2)):
        F.insert(len(F),np.zeros((min(2**(i+1),mbd),mbdW,min(2**(i+1),mbd))))
    for i in range(int(N/2)-1,0,-1):
        F.insert(len(F),np.zeros((min(2**(i),mbd),mbdW,min(2**i,mbd))))
    F.insert(len(F),np.array([[[1]]]))
    # Calculate Initial F (PH - Should speed up)
    for i in range(int(N)-1,0,-1):
        tmp1 = np.einsum('eaf,cdf->eacd',M[i],F[i+1])
        tmp2 = np.einsum('eacd,ydbe->acyb',tmp1,W[i])
        F[i] = np.einsum('acyb,bxc->xya',tmp2,np.conj(M[i]))
    return F

def calc_diag(M,W,F,site):
    # PH - Not Working!
    mpo_diag = np.einsum('abnn->anb',W[site])
    l_diag = np.einsum('lal->la',F[site])
    r_diag = np.einsum('rbr->rb',F[site+1])
    tmp = np.einsum('la,anb->lnb',l_diag,mpo_diag)
    diag = np.einsum('lnb,rb->nlr',tmp,r_diag)
    return(diag.ravel())

def calc_ham(M,W,F,site):
    tmp1 = np.einsum('lmin,kmq->linkq',W[site],F[site+1])
    H = np.einsum('jlp,linkq->ijknpq',F[site],tmp1)
    (n1,n2,n3,n4,n5,n6) = H.shape
    H = np.reshape(H,(n1*n2*n3,n4*n5*n6))
    return H

def calcRDM(M,swpDir):
    if swpDir == 'right':
        (n1,n2,n3) = M.shape
        M = np.swapaxes(M,0,1)
        M = np.reshape(M,(n2*n1,n3))
        return np.einsum('ij,kj->ik',M,np.conj(M))
    elif swpDir == 'left':
        (n1,n2,n3) = M.shape
        M = np.swapaxes(M,0,1)
        M = np.reshape(M,(n2,n1*n3))
        return np.einsum('ij,ik->jk',M,np.conj(M))

def renormalizeR(M,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = M[site].shape
    # Calculate the reduced density matrix
    for i in range(nStates):
        if nStates != 1: vtmp = v[:,i]
        vReshape = np.reshape(vtmp,(n1,n2,n3))
        w = 1./float(nStates)
        if i == 0:
            rdm = w*calcRDM(vReshape,'right')
        else:
            rdm +=w*calcRDM(vReshape,'right')
    # Take eigenvalues of the rdm
    vals,vecs = np.linalg.eig(rdm)
    # Sort Inds
    inds = np.argsort(vals)[::-1]
    # Keep only maxBondDim eigenstates
    inds = inds[:n3]
    vals = vals[inds]
    vecs = vecs[:,inds]
    # Make sure vecs are orthonormal
    vecs = la.orth(vecs)
    # Put resulting vectors into MPS
    M[site] = np.reshape(vecs,(n2,n1,n3))
    M[site] = np.swapaxes(M[site],0,1)
    #assert(np.all(np.isclose(np.einsum('ijk,ijl->kl',M[site],np.conj(M[site])),np.eye(n3),atol=1e-6)))
    if not np.all(np.isclose(np.einsum('ijk,ijl->kl',M[site],np.conj(M[site])),np.eye(n3),atol=1e-6)):
        print('\t\tNormalization Problem')
        # Check normalization:
        #for i in range(len(inds)):
        #    for j in range(len(inds)):
        #        print('\t\t\tNorm Check','i',i,'j',j,np.dot(vecs[:,i],np.conj(vecs[:,j])))
    # Calculate next site for guess
    vReshape = np.reshape(v[:,targetState],(n1,n2,n3))
    M[site+1] = np.einsum('ijk,ijl,mkn->mln',np.conj(M[site]),vReshape,M[site+1])
    return M

def renormalizeL(M,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = M[site].shape
    # Calculate the reduced density matrix
    for i in range(nStates):
        if nStates != 1: vtmp = v[:,i]
        vReshape = np.reshape(vtmp,(n1,n2,n3))
        w = 1./float(nStates)
        if i == 0:
            rdm = w*calcRDM(vReshape,'left') 
        else:
            rdm +=w*calcRDM(vReshape,'left')
    # Take eigenvalues of the rdm
    vals,vecs = np.linalg.eig(rdm) # Transpose here is useless...
    # Sort inds
    inds = np.argsort(vals)[::-1]
    # Keep only maxBondDim eigenstates
    inds = inds[:n2]
    vals = vals[inds]
    vecs = vecs[:,inds]
    # Make sure vecs are orthonormal
    vecs = la.orth(vecs)
    vecs = vecs.T
    # Put resulting vectors into MPS
    M[site] = np.reshape(vecs,(n2,n1,n3))
    M[site] = np.swapaxes(M[site],0,1)
    #assert(np.all(np.isclose(np.einsum('ijk,ilk->jl',M[site],np.conj(M[site])),np.eye(n2),atol=1e-6)))
    if not np.all(np.isclose(np.einsum('ijk,ilk->jl',M[site],np.conj(M[site])),np.eye(n2),atol=1e-6)):
        print('\t\tNormalization Problem')
    # Calculate next site's guess
    #M[site-1] = np.einsum('',
    return M

def calc_eigs(H,M,site,nStates):
    Mprev = M[site].ravel()
    vals,vecs = np.linalg.eig(H)
    inds = np.argsort(vals)[::-1]
    E = vals[inds[:nStates]]
    vecs = vecs[:,inds[:nStates]]
    return E,vecs

def calc_entanglement(S):
    EEspec = -S**2.*np.log2(S**2.)
    EE = np.sum(EEspec)
    return EE,EEspec

def update_envL(M,W,F,site):
    tmp1 = np.einsum('eaf,cdf->eacd',M[site],F[site+1])
    tmp2 = np.einsum('eacd,ydbe->acyb',tmp1,W[site])
    F[site] = np.einsum('acyb,bxc->xya',tmp2,np.conj(M[site]))
    return F

def update_envR(M,W,F,site):
    tmp1 = np.einsum('jlp,ijk->lpik',F[site],np.conj(M[site]))
    tmp2 = np.einsum('lmin,lpik->mpnk',W[site],tmp1)
    F[site+1] = np.einsum('npq,mpnk->kmq',M[site],tmp2)
    return F

def rightSweep(M,W,F,iterCnt,nStates=1):
    N = len(M)
    print('Right Sweep {}'.format(iterCnt))
    for site in range(N-1):
        #diag = calc_diag(M,W,F,site)
        H = calc_ham(M,W,F,site)
        E,v = calc_eigs(H,M,site,nStates)
        M = renormalizeR(M,v,site,nStates=nStates)
        F = update_envR(M,W,F,site)
        print('\tEnergy {}'.format(E))
    return E,M,F

def leftSweep(M,W,F,iterCnt,nStates=1):
    N = len(M)
    print('Left Sweep {}'.format(iterCnt))
    for site in range(N-1,0,-1):
        #diag = calc_diag(M,W,F,site)
        H = calc_ham(M,W,F,site)
        E,v = calc_eigs(H,M,site,nStates)
        M = renormalizeL(M,v,site,nStates=nStates)
        F = update_envL(M,W,F,site)
        print('\tEnergy {}'.format(E))
    return E,M,F

def checkConv(E_prev,E,tol,iterCnt,maxIter,nStates=1,targetState=0):
    if nStates != 1: E = E[targetState]
    if np.abs(E-E_prev) < tol:
        print('#'*75+'\nConverged at E = {}'.format(E)+'\n'+'#'*75)
        converged = True
    elif iterCnt > maxIter:
        print('Convergence not acheived')
        converged = True
    else:
        iterCnt += 1
        E_prev = E
        converged = False
    return converged,E_prev,iterCnt

def save_mps(M,fname):
    if fname is not None:
        Mdict = {}
        for i in range(len(M)):
            Mdict['M'+str(i)] = M[i]
        np.savez(fname,**Mdict)

def run_dmrg(N,hamParams,initGuess=None,mbd=100,tol=1e-5,maxIter=10,fname = None,nStates=1,targetState=0):
    W = create_sep_mpo(N,hamParams)
    if initGuess is None:
        M = create_rand_mps(N,mbd)
        M = make_mps_right(M)
    else:
        M = load_mps(N,initGuess)
    F = calc_env(M,W,mbd)
    # Run Sweeps
    converged = False
    iterCnt = 0
    E_prev = 0
    while not converged:
        E,M,F = rightSweep(M,W,F,iterCnt,nStates=nStates)
        E,M,F = leftSweep(M,W,F,iterCnt,nStates=nStates)
        converged,E_prev,iterCnt = checkConv(E_prev,E,tol,iterCnt,maxIter,nStates=nStates,targetState=targetState)
    save_mps(M,fname)
    if nStates != 1: E = E[targetState]
    return E

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 20
    p = 0.1
    mbd = 10
    sVec = np.linspace(-1,0.5,100)[::-1]
    E = np.zeros(sVec.shape)
    EE = np.zeros((len(sVec)))
    EEs = np.zeros((len(sVec),mbd))
    f = plt.figure()
    ax1 = f.add_subplot(151)
    ax2 = f.add_subplot(152)
    ax3 = f.add_subplot(153)
    ax4 = f.add_subplot(154)
    ax5 = f.add_subplot(155)
    for sind,s in enumerate(sVec):
        if sind == 0:
            print(s)
            E[sind] = run_dmrg(N,(0.5,0.5,p,1-p,0.5,0.5,s),mbd=mbd,fname='myMPS.npz',nStates=2)
        else:
            print(s)
            E[sind] = run_dmrg(N,(0.5,0.5,p,1-p,0.5,0.5,s),mbd=mbd,initGuess='myMPS.npz',fname='myMPS.npz',nStates=2)
        ax1.clear()
        ax1.plot(sVec[:sind],E[:sind])
        curr = (E[0:-1]-E[1:])/(sVec[0]-sVec[1])
        splt = sVec[1:]
        ax2.clear()
        ax2.plot(splt[:sind],curr[:sind])
        susc = (curr[0:-1]-curr[1:])/(sVec[0]-sVec[1])
        splt = sVec[1:-1]
        ax3.clear()
        ax3.plot(splt[:sind-1],susc[:sind-1])
        plt.pause(0.01)
