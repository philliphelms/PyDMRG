import numpy as np
from pydmrg.pydmrg.dmrg import run_dmrg
from pydmrg.pydmrg.mpo.asep import return_mpo, curr_mpo
from pydmrg.pydmrg.idmrg import return_bulk_mpo, return_edge_mpo
from pydmrg.pydmrg.tools.mpo_tools import mpo2mat
from pydmrg.pydmrg.tools.mps_tools import svd_right, calc_entanglement
from pyscf.lib import eig as davidson
from pyscf.lib import einsum
np.set_printoptions(linewidth=1000)

# A simple script to run the iDMRG algorithm 
# (sec 10, schollwock, DMRG in the age of MPS)
# 
# To Do:
# - Local Energy
# - Local Current
# - Correct Sing Vals
# - Efficient Observable Contractions

# Start by running a finite DMRG to get an initial guess
Ninit= 20
mbd = 1000
tol = 1e-16
alg = 'davidson'

# Hamiltonian Parameters
p = 0.1
alpha = 0.5
gamma = 1.-alpha
q     = 1.-p
beta  = 0.5
delta = 1.-beta
s = -0.5

# Get mpo
hamParams = np.array([alpha,gamma,p,q,beta,delta,s])
finiteMPO = return_mpo(Ninit,hamParams)
impo = return_mpo(4,hamParams)
icmpo = curr_mpo(4,hamParams)
mpoEdge = return_edge_mpo(impo)
mpoBulk = return_bulk_mpo(impo)
currMPO = return_bulk_mpo(icmpo)

# Run finite DMRG
centerSite = int(Ninit/2)-1
(E,_,_,mps,fullenv) = run_dmrg(finiteMPO,
                             mbd=mbd,
                             nStates=1,
                             returnState=True,
                             returnEnv=True,
                             gaugeSiteSave=centerSite)

# Move gauge to center
(n1,n2,n3) = mps[0][centerSite].shape
nStates = len(mps)
for state in range(nStates):
    (U,S,V) = svd_right(mps[state][centerSite])
    mps[state][centerSite] = np.reshape(U,(n1,n2,n3))
    mps[state][centerSite+1] = einsum('ij,kjl->kil',V,mps[state][centerSite+1])

# Contract MPS to get environment
env = [np.array([[[1.]]]),np.array([[[1.]]])]
for site in range(centerSite+1):
    tmp = einsum('ijk,lim->jklm',env[0],mps[0][site])
    tmp = einsum('jklm,jnol->kmno',tmp,finiteMPO[0][site])
    env[0] = einsum('kmno,okp->mnp',tmp,np.conj(mps[0][site]))
for site in range(centerSite+1,Ninit)[::-1]:
    tmp = einsum('ijk,lmi->jklm',env[1],mps[0][site])
    tmp = einsum('jklm,njol->kmno',tmp,finiteMPO[0][site])
    env[1] = einsum('kmno,opk->mnp',tmp,np.conj(mps[0][site]))

# Check energy
print('Energy from Environment: {}'.format(einsum('ijk,i,k,ijk->',env[0],S,S.conj(),env[1])))

# Get iMPS from full MPS
mps = [mps[0][centerSite],mps[0][centerSite+1]]


# Run growth procedure
converged = False
iterCnt = 0
N = Ninit + 2
Sm = S
while not converged:
    printStr = ''
    # Predict Next Guess
    (n1,n2,n3) = mps[0].shape
    LB = einsum('j,ijk->ijk',S,mps[1])
    LB = np.swapaxes(LB,0,1)
    LB = np.reshape(LB,(n1*n3,n2))
    (U_,S_,V_) = np.linalg.svd(LB,full_matrices=False)
    Aguess = np.reshape(U_,(n3,n1,n2))
    Aguess = np.swapaxes(Aguess,0,1)
    LR = einsum('i,ij->ij',S_,V_)
    LambdaPad = min(n3*2,mbd)-n2
    LR = np.pad(LR,((0,LambdaPad),(0,LambdaPad)),'constant')
    (n1_,n2_,n3_) = Aguess.shape
    Aguess = np.pad(Aguess,((0,0),
                             (0,n3-n2_),
                             (0,min(n3*2,mbd)-n3_)),
                             'constant')
    AL = einsum('ijk,k->ijk',mps[0],S)
    AL = np.swapaxes(AL,0,1)
    AL = np.reshape(AL,(n2,n1*n3))
    (U_,S_,V_) = np.linalg.svd(AL,full_matrices=False)
    Bguess = np.reshape(V_,(n2,n1,n3))
    Bguess = np.swapaxes(Bguess,0,1)
    LL = einsum('ij,j->ij',U_,S_)
    LL = np.pad(LL,((0,LambdaPad),(0,LambdaPad)),'constant')
    Bguess = np.pad(Bguess,((0,0),
                            (0,min(n3*2,mbd)-n3_),
                            (0,n3-n2_)),
                            'constant')
    Sm_inv = np.linalg.inv(np.diag(Sm))
    LambdaPad = min(n3*2,mbd)-len(Sm_inv)
    Sm_inv = np.pad(Sm_inv,((0,LambdaPad),(0,LambdaPad)),'constant')
    Lguess = np.dot(LR,np.dot(Sm_inv,LL))
    mps[0] = Aguess
    mps[1] = Bguess

    # Absorb guesses into environment tensors
    tmp = einsum('ijk,lim->jklm',env[0],mps[0])
    tmp = einsum('jklm,jnol->kmno',tmp,mpoBulk[0][0])
    env[0] = einsum('kmno,okp->mnp',tmp,np.conj(mps[0]))
    tmp = einsum('ijk,lmi->jklm',env[1],mps[1])
    tmp = einsum('jklm,njol->kmno',tmp,mpoBulk[0][1])
    env[1] = einsum('kmno,opk->mnp',tmp,np.conj(mps[1]))

    # Solve Eigenproblem
    if alg == 'exact':
        #HL = einsum('ijk,lim,jnol,okp->mnp',env[0],mps[0],mpoBulk[0][0],mps[0].conj())
        #HR = einsum('IJK,LMI,NJOL,OPK->MNP',env[1],mps[1],mpoBulk[0][1],mps[1].conj())
        H = einsum('mnp,MnP->pPmM',env[0],env[1])
        (n1,n2,n3,n4) = H.shape
        H = np.reshape(H,(n1*n2,n3*n4))
        (nz,_) = H.shape
        E,v = np.linalg.eig(H)
        inds = np.argsort(E)[::-1]
        E = E[inds[0]]
        v = v[:,inds[0]]
    elif alg == 'davidson':
        # Define Hamiltonian Function
        def Hfun(x):
            (_,_,n1) = mps[0].shape
            (_,n2,_) = mps[1].shape
            x_reshape = np.reshape(x,(n1,n2))
            fin_sum = np.zeros(x_reshape.shape,dtype=np.complex_)
            for mpoInd in range(len(mpoBulk)):
                tmp1 = einsum('ijk,il->ljk',env[0],x_reshape)
                fin_sum += einsum('ljm,ljk->km',env[1],tmp1)
            return -np.ravel(fin_sum)
        # Define Preconditioner Function
        def precond(dx,e,x0): return dx
        # Solve eigenproblem
        vals,vecso = davidson(Hfun,np.ravel(Lguess),precond,nroots=1,follow_state=False,tol=1e-10,max_cycle=1000)
        E = -vals
        v = vecso
    printStr += 'N={}\t'.format(N)
    printStr += 'E={}\t'.format(E)
    printStr += 'ovlp={}\t'.format(np.abs(np.dot(np.ravel(Lguess),v.conj())))

    # Save old Sing Val vector
    Sm = S

    # Put result into MPS
    dim = int(np.sqrt(len(v)))
    v = np.reshape(v,(dim,dim))
    (U,S,V) = np.linalg.svd(v)
    mps[0] = einsum('ijk,kl->ijl',mps[0],U)
    mps[1] = einsum('ij,kjl->kil',V,mps[1])
    env[0] = einsum('ijk,il->ljk',env[0],U)
    env[0] = einsum('ljk,km->ljm',env[0],U.conj())
    env[1] = einsum('ijk,li->ljk',env[1],V)
    env[1] = einsum('ljk,mk->ljm',env[1],V.conj())

    # Evaluate Observables
    psi = einsum('ijk,k,lkm->iljm',mps[0],S,mps[1])
    norm = einsum('ijkl,ijkl->',psi,np.conj(psi))
    # Global Energy
    envE = np.real(einsum('ijk,i,k,ijk->',env[0],S,S.conj(),env[1]))/norm
    if np.abs(envE-E) > 1e-3:
        print('Uh oh...')
    # Local Energy
    Eloc = einsum('ijkl,mnoi,npqj,oqkl->',psi,
                                          np.expand_dims(mpoBulk[0][0][-1,:,:,:],axis=0),
                                          np.expand_dims(mpoBulk[0][1][: ,0,:,:],axis=1),
                                          np.conj(psi))/norm
    printStr += 'Eloc={}\t'.format(np.real(Eloc))
    # Local Current
    Jloc = einsum('ijkl,mnoi,npqj,oqkl->',psi,
                                          np.expand_dims(currMPO[0][0][-1,:,:,:],axis=0),
                                          np.expand_dims(currMPO[0][1][: ,0,:,:],axis=1),
                                          np.conj(psi))/norm
    printStr += 'Jloc={}\t'.format(np.real(Jloc))
    # Local Density
    n = np.array([[[[0.,0.],[0.,1.]]]])
    I = np.array([[[[1.,0.],[0.,1.]]]])
    rhoL = einsum('ijkl,mnoi,npqj,oqkl->',psi,n,I,np.conj(psi))/norm
    rhoR = rhoR = einsum('ijkl,mnoi,npqj,oqkl->',psi,I,n,np.conj(psi))/norm
    printStr += 'rhoL={}\t'.format(np.real(rhoL))
    printStr += 'rhoR={}\t'.format(np.real(rhoR))
    # Calculate/Print Entanglement Entropy
    EE = calc_entanglement(S)[0]
    printStr += 'EE={}\t'.format(EE)

    # Check Convergence
    mat = np.dot(LL,np.diag(Sm))
    (_,sConv,_) = np.linalg.svd(mat)
    fidelity = np.sum(sConv)
    printStr += 'fid={}\t'.format(1.-fidelity)
    if np.abs(1.-fidelity) < tol:
        converged = True
    else:
        # Update iteration stuff
        N += 2
        iterCnt += 1
    print(printStr)
