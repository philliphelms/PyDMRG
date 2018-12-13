import numpy as np
import scipy.linalg as la

# To Do :
# - Calculate observables (EE) after each state has converged
# - Return Gap
# - Left Eigenvectors
# - constant_mbd=True does not work

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

def increase_mbd(M,mbd,periodic=False,constant=False,d=2):
    N = len(M)
    if periodic == False:
        if constant == False:
            for site in range(int(N/2)):
                nx,ny,nz = M[site].shape
                sz1 = min(d**site,mbd)
                sz2 = min(d**(site+1),mbd)
                M[site] = np.pad(M[site], ((0,0), (0,sz1-ny), (0,sz2-ny)), 'constant', constant_values=0j)
            if N%2 is 1:
                site += 1
                nx,ny,nz = M[site].shape
                sz1 = min(d**(site),mbd)
                sz2 = min(d**(site),mbd)
                M[site] = np.pad(M[site], ((0,0), (0,sz1-ny), (0,sz2-nz)), 'constant', constant_values=0j)
            for i in range(int(N/2))[::-1]:
                site = N - i - 1
                nx,ny,nz = M[site].shape
                sz1 = min(d**(site+1),mbd)
                sz2 = min(d**(site),mbd)
                M[site] = np.pad(M[site], ((0,0), (0,sz1-ny), (0,sz2-nz)), 'constant', constant_values=0j)
        else:
            for site in range(N):
                nx,ny,nz = M[site].shape
                sz1 = mbd
                sz2 = mbd
                if site == 0: sz1 = 1
                if site == N-1: sz2 = 1
                M[site] = np.pad(M[site], ((0,0), (0,sz1-ny), (0,sz2-nz)), 'constant', constant_values=0j)
    else:
        for site in range(N):
            nx,ny,nz = M[site].shape
            sz1 = mbd
            sz2 = mbd
            M[site] = np.pad(M[site], ((0,0), (0,sz1-ny), (0,sz2-nz)), 'constant', constant_values=0j)
    return M

def create_rand_mps(N,mbd,d=2):
    # Create MPS
    M = []
    for i in range(int(N/2)):
        np.random.rand(d,d,d)
        min(d**(i),mbd)
        min(d**(i+1),mbd)
        np.random.rand(d,min(d**(i),mbd),min(d**(i+1),mbd))
        M.insert(len(M),np.random.rand(d,min(d**(i),mbd),min(d**(i+1),mbd)))
    if N%2 is 1:
        M.insert(len(M),np.random.rand(d,min(d**(i+1),mbd),min(d**(i+1),mbd)))
    for i in range(int(N/2))[::-1]:
        M.insert(len(M),np.random.rand(d,min(d**(i+1),mbd),min(d**i,mbd)))
    return M

def load_mps(N,fname):
    npzfile = np.load(fname+'.npz')
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
    if not np.all(np.isclose(np.einsum('ijk,ijl->kl',M[site],np.conj(M[site])),np.eye(n3),atol=1e-6)):
        print('\t\tNormalization Problem')
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
    if not np.all(np.isclose(np.einsum('ijk,ilk->jl',M[site],np.conj(M[site])),np.eye(n2),atol=1e-6)):
        print('\t\tNormalization Problem')
    # PH - Calculate next site's guess
    M[site-1] = np.einsum('',
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
        print('\tEnergy at Site {}: {}'.format(site,E))
        if site == int(N/2):
            Ereturn = E
    return Ereturn,M,F

def leftSweep(M,W,F,iterCnt,nStates=1):
    N = len(M)
    print('Left Sweep {}'.format(iterCnt))
    for site in range(N-1,0,-1):
        #diag = calc_diag(M,W,F,site)
        H = calc_ham(M,W,F,site)
        E,v = calc_eigs(H,M,site,nStates)
        M = renormalizeL(M,v,site,nStates=nStates)
        F = update_envL(M,W,F,site)
        print('\tEnergy at Site {}: {}'.format(site,E))
        if site == int(N/2):
            Ereturn = E
    return Ereturn,M,F

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
        np.savez(fname+'.npz',**Mdict)

def observable_sweep(M,F):
    # Going to the right
    # PH - Only calculates Entanglement Currently
    N = len(M)
    for site in range(N-1):
        (n1,n2,n3) = M[site].shape
        M_reshape = np.reshape(M[site],(n1*n2,n3))
        (U,S,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M[site] = np.reshape(U,(n1,n2,n3))
        M[site+1] = np.einsum('i,ij,kjl->kil',S,V,M[site+1])
        if site == int(N/2):
            EE,EEs = calc_entanglement(S)
    return EE,EEs

def run_sweeps(M,W,F,initGuess=None,maxIter=10,tol=1e-5,fname = None,nStates=1,targetState=0):
    converged = False
    iterCnt = 0
    E_prev = 0
    while not converged:
        E,M,F = rightSweep(M,W,F,iterCnt,nStates=nStates)
        E,M,F = leftSweep(M,W,F,iterCnt,nStates=nStates)
        converged,E_prev,iterCnt = checkConv(E_prev,E,tol,iterCnt,maxIter,nStates=nStates,targetState=targetState)
    save_mps(M,fname)
    EE,EEs = observable_sweep(M,F)
    if nStates != 1: E = E[targetState]
    return E,EE

def run_dmrg(N,hamParams,initGuess=None,mbd=[2,4,8,16],tol=1e-5,maxIter=3,fname=None,nStates=1,targetState=0,constant_mbd=False):
    # Determine Hamiltonian MPO
    W = create_sep_mpo(N,hamParams)
    # Make sure everything is a vector
    if not hasattr(mbd,'__len__'): mbd = np.array([mbd])
    if not hasattr(tol,'__len__'):
        tol = tol*np.ones(len(mbd))
    else:
        assert(len(mbd) == len(tol))
    if not hasattr(maxIter,'__len__'):
        maxIter = maxIter*np.ones(len(mbd))
    else:
        assert(len(maxIter) == len(mbd))
    # Vector to store results
    Evec = np.zeros(len(mbd))
    EEvec= np.zeros(len(mbd))
    for mbdInd,mbdi in enumerate(mbd):
        if initGuess is None:
            M = create_rand_mps(N,mbdi)
            M = make_mps_right(M)
            if constant_mbd: M = increase_mbd(M,mbdi,constant=True)
        else:
            M = load_mps(N,initGuess+'_mbd'+str(mbdInd))
        F = calc_env(M,W,mbdi)
        E,EE = run_sweeps(M,W,F,
                         maxIter=maxIter[mbdInd],
                         tol=tol[mbdInd],
                         fname=fname+'_mbd'+str(mbdInd),
                         nStates=nStates,
                         targetState=targetState)
        Evec[mbdInd] = E
        EEvec[mbdInd]= EE
    return Evec,EEvec

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 10
    p = 0.1
    mbd = [2,4,6]
    sVec = np.linspace(-1,0.5,100)[::-1]
    E = np.zeros((len(sVec),len(mbd)))
    EE = np.zeros((len(sVec),len(mbd)))
    f = plt.figure()
    ax1 = f.add_subplot(151)
    ax2 = f.add_subplot(152)
    ax3 = f.add_subplot(153)
    ax4 = f.add_subplot(154)
    ax5 = f.add_subplot(155)
    for sind,s in enumerate(sVec):
        if sind == 0:
            print(s)
            E[sind,:],EE[sind,:] = run_dmrg(N,(0.5,0.5,p,1-p,0.5,0.5,s),mbd=mbd,fname='mps/myMPS_N'+str(N),nStates=2)
        else:
            print(s)
            E[sind,:],EE[sind,:] = run_dmrg(N,(0.5,0.5,p,1-p,0.5,0.5,s),mbd=mbd,initGuess='mps/myMPS_N'+str(N),fname='mps/myMPS_N'+str(N),nStates=2)
        # Print Results
        ax1.clear()
        for i in range(len(mbd)):
            ax1.plot(sVec[:sind],E[:sind,i])
        ax2.clear()
        for i in range(len(mbd)):
            ax2.semilogy(sVec[:sind],np.abs(E[:sind,i]-E[:sind,-1]))
        curr = (E[0:-1,:]-E[1:,:])/(sVec[0]-sVec[1])
        splt = sVec[1:]
        ax3.clear()
        for i in range(len(mbd)):
            ax3.plot(splt[:sind],curr[:sind,i])
        susc = (curr[0:-1,:]-curr[1:,:])/(sVec[0]-sVec[1])
        splt = sVec[1:-1]
        ax4.clear()
        for i in range(len(mbd)):
            ax4.plot(splt[:sind-1],susc[:sind-1,i])
        ax5.clear()
        for i in range(len(mbd)):
            ax5.plot(sVec[:sind],EE[:sind,i])
        plt.pause(0.01)
        # Save Results
        np.savez('results/asep_psweep_N'+str(N)+'_Np1_Ns'+str(len(sVec)),N=N,p=p,mbd=mbd,s=sVec,E=E,EE=EE)
    plt.show()
