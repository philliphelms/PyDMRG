import numpy as np
from pydmrg.pydmrg.dmrg import run_dmrg
from pydmrg.pydmrg.mpo.asep import return_mpo
from pydmrg.pydmrg.idmrg import return_bulk_mpo, return_edge_mpo
from pydmrg.pydmrg.tools.mpo_tools import mpo2mat
from pydmrg.pydmrg.tools.mps_tools import svd_right, calc_entanglement
np.set_printoptions(linewidth=1000)

# A simple script to run the iDMRG algorithm 
# (sec 10, schollwock, DMRG in the age of MPS)

# Start by running a finite DMRG to get an initial guess
Ninit= 10
mbd = 10
tol = 1e-10

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
mpoEdge = return_edge_mpo(impo)
mpoBulk = return_bulk_mpo(impo)

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
    mps[state][centerSite+1] = np.einsum('ij,kjl->kil',V,mps[state][centerSite+1])

# Contract MPS to get environment
env = [np.array([[[1.]]]),np.array([[[1.]]])]
for site in range(centerSite+1):
    env[0] = np.einsum('ijk,lim,jnol,okp->mnp',
                       env[0],
                       mps[0][site],
                       finiteMPO[0][site],
                       np.conj(mps[0][site]))
for site in range(centerSite+1,Ninit)[::-1]:
    env[1] = np.einsum('ijk,lmi,njol,opk->mnp',
                       env[1],
                       mps[0][site],
                       finiteMPO[0][site],
                       np.conj(mps[0][site]))
# Check energy
print('Energy from Environment: {}'.format(np.einsum('ijk,i,k,ijk->',env[0],S,S.conj(),env[1])))

# Get iMPS from full MPS
mps = [mps[0][centerSite],mps[0][centerSite+1]]


# Run growth procedure
converged = False
iterCnt = 0
N = Ninit + 2
while not converged:
    print('\n\nIteration {}'.format(iterCnt))
    # Predict Next Guess
    print('\tPreparing Guess')
    Sm = S
    (n1,n2,n3) = mps[0].shape
    LB = np.einsum('j,ijk->ijk',S,mps[1])
    LB = np.swapaxes(LB,0,1)
    LB = np.reshape(LB,(n1*n3,n2))
    (U_,S_,V_) = np.linalg.svd(LB,full_matrices=False)
    Aguess = np.reshape(U_,(n3,n1,n2))
    Aguess = np.swapaxes(Aguess,0,1)
    LR = np.einsum('i,ij->ij',S_,V_)
    LambdaPad = min(n3*2,mbd)-n2
    LR = np.pad(LR,((0,LambdaPad),(0,LambdaPad)),'constant')
    (n1_,n2_,n3_) = Aguess.shape
    Aguess = np.pad(Aguess,((0,0),
                             (0,n3-n2_),
                             (0,min(n3*2,mbd)-n3_)),
                             'constant')
    AL = np.einsum('ijk,k->ijk',mps[0],S)
    AL = np.swapaxes(AL,0,1)
    AL = np.reshape(AL,(n2,n1*n3))
    (U_,S_,V_) = np.linalg.svd(AL,full_matrices=False)
    Bguess = np.reshape(V_,(n2,n1,n3))
    Bguess = np.swapaxes(Bguess,0,1)
    LL = np.einsum('ij,j->ij',U_,S_)
    LL = np.pad(LL,((0,LambdaPad),(0,LambdaPad)),'constant')
    Bguess = np.pad(Bguess,((0,0),
                            (0,min(n3*2,mbd)-n3_),
                            (0,n3-n2_)),
                            'constant')
    Sm_inv = np.linalg.inv(np.diag(Sm))
    LambdaPad = min(n3*2,mbd)-len(Sm_inv)
    Sm_inv = np.pad(Sm_inv,((0,LambdaPad),(0,LambdaPad)),'constant')
    Lguess = np.dot(LR,np.dot(Sm_inv,LL))

    # Solve Eigenproblem
    print('\tSolving Eigenproblem')
    HL = np.einsum('ijk,lim,jnol,okp->mnp',env[0],Aguess,mpoBulk[0][0],Aguess.conj())
    HR = np.einsum('IJK,LMI,NJOL,OPK->MNP',env[1],Bguess,mpoBulk[0][1],Bguess.conj())
    H = np.einsum('mnp,MnP->pPmM',HL,HR)
    (n1,n2,n3,n4) = H.shape
    H = np.reshape(H,(n1*n2,n3*n4))
    E,v = np.linalg.eig(H)
    inds = np.argsort(E)[::-1]
    E = E[inds]
    v = v[:,inds]
    print('\t\tEnergy = {}'.format(E[0]))
    print('\t\tE/{} = {}'.format(N,E[0]/N))
    print('\t\tOverlap = {}'.format(np.dot(np.ravel(Lguess),v[:,0])))

    # Put result into MPS
    v = v[:,0]
    dim = int(np.sqrt(len(v)))
    v = np.reshape(v,(dim,dim))
    (U,S,V) = np.linalg.svd(v)
    mps[0] = np.einsum('ijk,kl->ijl',Aguess,U)
    mps[1] = np.einsum('ij,kjl->kil',V,Bguess)
    
    # Update Environment
    env[0] = np.einsum('ijk,lim,jnol,okp->mnp',
                       env[0],
                       mps[0],
                       mpoBulk[0][0],
                       np.conj(mps[0]))
    env[1] = np.einsum('ijk,lmi,njol,opk->mnp',
                       env[1],
                       mps[1],
                       mpoBulk[0][1],
                       np.conj(mps[1]))

    # Check energy
    envE = np.einsum('ijk,i,k,ijk->',env[0],S,S.conj(),env[1])
    print('\t\tEnergy from Environment: {}'.format(envE))
    if np.abs(envE-E[0]) > 1e-3:
        print('Uh oh...')

    # Calculate/Print Entanglement Entropy
    EE = calc_entanglement(S)[0]
    print('\t\tEntanglement Entropy = {}'.format(EE))

    # Check Convergence
    mat = np.dot(LL,np.diag(Sm))
    (_,sConv,_) = np.linalg.svd(mat)
    fidelity = np.sum(sConv)
    print('\t\t1-Fidelity = {}'.format(1.-fidelity))
    if np.abs(1.-fidelity) < tol:
        converged = True
    else:
        # Update iteration stuff
        N += 2
        iterCnt += 1
