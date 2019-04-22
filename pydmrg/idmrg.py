import numpy as np
import scipy.linalg as sla
from pyscf.lib import eig as davidson
from pyscf.lib import einsum
from scipy.sparse.linalg import eigs as arnoldi
from scipy.sparse.linalg import LinearOperator
from tools.mps_tools import *
from tools.mpo_tools import *
from tools.diag_tools import *
from tools.env_tools import *
from tools.contract import *
import warnings
import copy

VERBOSE = 10

def return_bulk_mpo(mpo):
    mpoBulk = []
    nOps = len(mpo)
    for opInd in range(nOps):
        op = [None]*2
        op[0] = mpo[opInd][1]
        op[1] = mpo[opInd][2]
        mpoBulk.append(op)
    return mpoBulk

def return_edge_mpo(mpo):
    mpoEdge = []
    nOps = len(mpo)
    for opInd in range(nOps):
        op = [None]*2
        op[0] = mpo[opInd][0]
        op[1] = mpo[opInd][3]
        mpoEdge.append(op)
    return mpoEdge

def printResults(converged,E,EE,EEspec,gap):
    if VERBOSE > 1: print('#'*75)
    if converged:
        if VERBOSE > 0: print('Converged at E = {}'.format(E))
    else:
        if VERBOSE > 0: print('Convergence not acheived, E = {}'.format(E))
    if VERBOSE > 2: print('\tGap = {}'.format(gap))
    if VERBOSE > 2: print('\tEntanglement Entropy  = {}'.format(EE))
    if VERBOSE > 3: print('\tEntanglement Spectrum =')
    for i in range(len(EEspec)):
        if VERBOSE > 3: print('\t\t{}'.format(EEspec[i]))
    if VERBOSE > 1: print('#'*75)

def make_next_guess(mps,Sl,Slm,mbd=10):
    (n1_,n2_,n3_) = mps[0][0].shape
    (n4_,n5_,n6_) = mps[0][1].shape
    for state in range(len(mps)):
        # Get Left Side Prediction
        LB = np.einsum('j,ijk->ijk',Sl,mps[state][1])
        (U,S,V) = svd_right(LB)
        Alp1 = np.reshape(U,LB.shape)
        lambdaR = np.einsum('i,ij->ij',S,V)
        # Get Right Side Prediction
        AL = np.einsum('ijk,k->ijk',mps[state][0],Sl)
        (U,S,V) = svd_left(AL)
        Blp1 = np.reshape(V,AL.shape)
        lambdaL = np.einsum('ij,j->ij',U,S)
        # Put into MPS
        mps[state][0] = np.einsum('ijk,kl,lm->ijm',Alp1,lambdaR,np.linalg.inv(np.diag(Slm))) # PH - How to take inverse of vector
        mps[state][1] = np.einsum('ij,kjm->kim',lambdaL,Blp1)
    # Increase Bond Dim if needed
    nStates = len(mps)
    (n1,n2,n3) = mps[0][0].shape
    (n4,n5,n6) = mps[0][1].shape
    for state in range(nStates):
        (n1,n2,n3) = mps[state][0].shape
        mps[state][0] = np.pad(mps[state][0],((0,0),(0,min(mbd,n3_)-n2),(0,min(mbd,n3_*n1)-n3)),'constant')
        mps[state][1] = np.pad(mps[state][1],((0,0),(0,min(mbd,n5_*n4)-n5),(0,min(mbd,n5_)-n6)),'constant')
    return mps,lambdaL

def calc_conv_fidelity(lL,l):
    (lL_dim,_) = lL.shape
    (l_dim,) = l.shape
    # Calculate fidelity of rdms, page 109 of schollwock
    l = np.diag(l)
    if l_dim > lL_dim:
        lL = np.pad(lL,((0,l_dim-lL_dim),(0,l_dim-lL_dim)),'constant')
    mat = np.dot(lL,np.conj(l).T)
    (U,S,V) = np.linalg.svd(mat)
    fidelity = np.sum(S)
    return fidelity

def checkConv(lambdaL,S,iterCnt,
              tol=1e-10,maxIter=100,minIter=None,
              nStates=1,targetState=0):
    # Get fidelity of rdms
    fidelity = calc_conv_fidelity(lambdaL,S)
    # Check for convergence
    if (np.abs(fidelity-1.) < tol) and (iterCnt > minIter-3):
        cont = False
        conv = True
    elif iterCnt > maxIter - 1:
        cont = False
        conv = False
    else:
        cont = True
        conv = False
    return cont,conv,fidelity

def calc_local_energy(mps,mpo,state=0,mpsl=None):
    if mpsl is None: mpsl = conj_mps(mps)
    from mpo.asep import return_mpo
    from tools.contract import inf_contract as contract
    p = 0.1 
    alpha = 0.5      # in at left
    gamma = 1.-alpha  # Out at left
    q     = 1.-p      # Jump left
    beta  = 0.5     # Out at right
    delta = 1.-beta   # In at right
    s0 = -0.5
    hamParams = np.array([alpha,gamma,p,q,beta,delta,s0])
    MPO_Eloc = return_mpo(4,hamParams,singleBond=True,bond=1)
    MPO_Eloc = return_bulk_mpo(MPO_Eloc)
    norm = contract(mps=mps,lmps=mpsl,state=0,lstate=0)
    Eloc = -contract(mpo=MPO_Eloc,mps=mps,lmps=mpsl,state=0,lstate=0)/norm
    return Eloc

def single_iter(N,mps,mpo,
                env,Sprev,E_prev,
                iterCnt=0,maxIter=1000,
                minIter=10,targetState=0,local_energy=True,
                nStates=1,tol=1e-10,alg='davidson',mbd=10):
    # Solve Eigenproblem
    E,vecs,_ = calc_eigs(mps,mpo,env,0,nStates,
                         alg=alg,oneSite=False,edgePreserveState=False)
    # Do SVD
    mps,EE,EEs,S = renorm_inf(mps,mpo,env,vecs,mbd)
    # Calculate Local Energy
    print('Recontracted Energy = {}'.format(1./N*einsum('ijk,lkm,m,nmo,jpql,prsn,qit,t,stu,uro->',
                                            env[0][0],mps[0][0],S,mps[0][1],mpo[0][0],mpo[0][1],mps[0][0].conj(),S.conj(),mps[0][1].conj(),env[0][1])))
    # Update Environments
    env = update_env_inf(mps[0],mpo,env,mpsl=None)
    # Update MPS (increase bond dim)
    if Sprev is None: Sprev = S
    mps,lambdaL = make_next_guess(mps,S,Sprev,mbd=mbd)
    # Check for convergence
    cont,conv,fidelity = checkConv(lambdaL,Sprev,iterCnt,
                                   tol=tol,maxIter=maxIter,minIter=minIter,
                                   nStates=nStates,targetState=targetState)
    # Print Results
    print('N={}\tEnergy = {:f}\tLocal E={:f}\tdiff={:e}\tfidelity={:e}\tEE={:f}'.format(N,np.real(E/N),np.real(Eloc),np.real(np.abs(E/N-E_prev/N)),1.-fidelity,EE))
    # Update IterCnt and Energies
    iterCnt += 1
    return (N,E,EE,EEs,mps,env,S,cont,conv,iterCnt)

def run_iters(mps,mpo,env,mbd=10,maxIter=100,minIter=None,
               tol=1e-10,fname=None,nStates=1,
               targetState=0,alg='davidson',
               preserveState=False,returnState=False,
               returnEnv=False,returnEntSpec=False,
               orthonormalize=False):
    # Get Edge and bulk mpos separated
    # PH - How to handle this is initial environment is already loaded...
    mpoEdge = return_edge_mpo(mpo)
    mpoBulk = return_bulk_mpo(mpo)

    # Do first step (using edge mpo)
    N = 2
    if True: # PH - Find way to determine if we have a new environment
        output = single_iter(2,mps,mpoEdge,
                             env,None,0.,
                             iterCnt=0,maxIter=maxIter,
                             minIter=minIter,targetState=targetState,
                             nStates=nStates,tol=tol,alg=alg,mbd=mbd)
        (N,E,EE,EEs,mps,env,S,cont,conv,iterCnt) = output

    # Run iterations
    cont = True
    while cont:
        # Increase System Size
        N += 2
        # Run single update
        output = single_iter(N,mps,mpoBulk,
                             env,S,E,
                             iterCnt=iterCnt,maxIter=maxIter,
                             minIter=minIter,targetState=targetState,
                             nStates=nStates,tol=tol,alg=alg,mbd=mbd)
        (N,E,EE,EEs,mps,env,S,cont,conv,iterCnt) = output

    # Save Resulting MPS
    save_mps(mps,fname,gaugeSite=0)

    # Determine which vars to return
    if nStates != 1: 
        gap = E[0]-E[1]
    else:
        gap = None
    if hasattr(E,'__len__'): E = E[targetState]
    printResults(conv,E,EE,EEs,gap)
    output = [E,EE,gap]
    if returnEntSpec:
        output.append(EEs)
    if returnState:
        output.append(mps)
    if returnEnv:
        output.append(env)
    return output

def run_idmrg(mpo,initEnv=None,initGuess=None,mbd=[2,4,8,16],
             tol=1e-16,maxIter=1000,minIter=10,fname=None,
             nStates=1,targetState=0,alg='davidson',
             preserveState=False,edgePreserveState=False,
             returnState=False,returnEnv=False,
             returnEntSpec=False,orthonormalize=False,
             calcLeftState=False):

    # Check Data Structures to make sure they are correct
    if not hasattr(mbd,'__len__'): mbd = np.array([mbd])
    if not hasattr(tol,'__len__'):
        tol = tol*np.ones(len(mbd))
    else:
        assert(len(mbd) == len(tol))
    if not hasattr(maxIter,'__len__'):
        maxIter = maxIter*np.ones(len(mbd))
    else:
        assert(len(maxIter) == len(mbd))
    if not hasattr(minIter,'__len__'):
        minIter = minIter*np.ones(len(mbd))
    else:
        assert(len(minIter) == len(mbd))

    # Get mpo for calculating left state
    if calcLeftState: mpol = mpo_conj_trans(mpo)

    # Create data structures to save results
    Evec  = np.zeros(len(mbd),dtype=np.complex_)
    EEvec = np.zeros(len(mbd),dtype=np.complex_)
    gapvec= np.zeros(len(mbd),dtype=np.complex_)
    if calcLeftState: EEvecl = np.zeros(len(mbd),dtype=np.complex_)

    # Loop over all maximum bond dimensions, running dmrg for each one
    for mbdInd,mbdi in enumerate(mbd):
        if VERBOSE > 1: print('Starting Calc for MBD = {}'.format(mbdi))

        # Set up initial MPS
        if initGuess is None:
            # Generate Random Guess
            mps = create_all_mps(2,mbdi,nStates)
            if calcLeftState: mpsl = create_all_mps(2,mbdi,nStates)
        else: 
            # Load user provided Guess    
            if mbdInd == 0:
                mps,_ = load_mps(initGuess+'_mbd'+str(mbdInd))
                if calcLeftState: mpsl,_ = load_mps(initGuess+'_mbd'+str(mbdInd)+'_left')

        # Calc initial environment (or load if provided)
        if initEnv is None: 
            env = calc_env_inf(mps,mpo,mbdi)
            if calcLeftState: envl = calc_env_inf(mpsl,mpol,mbdi)
        else:
            env = initEnv
            if calcLeftState: env,envl = initEnv[0],initEnv[1]

        # Add an index to the MPS filename saving to indicate its bond dimension
        fname_mbd = None
        if fname is not None: fname_mbd = fname + '_mbd' + str(mbdInd)

        # Run DMRG Sweeps (right eigenvector)
        if VERBOSE > 0: print('Calculating Right Eigenstate')
        output = run_iters(mps,mpo,env,
                          mbd=mbdi,
                          maxIter=maxIter[mbdInd],
                          minIter=minIter[mbdInd],
                          tol=tol[mbdInd],
                          fname=fname_mbd,
                          nStates=nStates,
                          alg=alg,
                          targetState=targetState,
                          preserveState=preserveState,
                          returnState=returnState,
                          returnEnv=returnEnv,
                          returnEntSpec=returnEntSpec,
                          orthonormalize=orthonormalize)

        # Extract Results
        E = output[0]
        EE = output[1]
        gap = output[2]
        Evec[mbdInd]  = output[0]
        EEvec[mbdInd] = output[1]
        gapvec[mbdInd]= output[2]
        # Extract Extra results
        if returnEntSpec and returnState and returnEnv:
            EEs = output[3]
            mps = output[4]
            env = output[5]
        elif returnEntSpec and returnState:
            EEs = output[3]
            mps = output[4]
        elif returnEntSpec and returnEnv:
            EEs = output[3]
            env = output[4]
        elif returnState and returnEnv:
            mps = output[3]
            env = output[4]
        elif returnEntSpec:
            EEs = output[3]
        elif returnState:
            mps = output[3]
        elif returnEnv:
            env = output[3]

        if calcLeftState:
            # Run DMRG Sweeps (left eigenvector)
            if VERBOSE > 0: print('Calculating Left Eigenstate')
            output = run_iters(mpsl,mpol,envl,
                              mbd=mbdi,
                              maxIter=maxIter[mbdInd],
                              minIter=minIter[mbdInd],
                              tol=tol[mbdInd],
                              fname=fname_mbd+'_left',
                              nStates=nStates,
                              alg=alg,
                              targetState=targetState,
                              preserveState=preserveState,
                              returnState=returnState,
                              returnEnv=returnEnv,
                              returnEntSpec=returnEntSpec,
                              orthonormalize=orthonormalize)
            # Extract left state specific Results
            EEl = output[1]
            EEvecl[mbdInd]  = output[1]
            # Extra potential extra data
            if returnEntSpec and returnState and returnEnv:
                EEsl = output[3]
                mpsl = output[4]
                envl = output[5]
            elif returnEntSpec and returnState:
                EEsl = output[3]
                mpsl = output[4]
            elif returnEntSpec and returnEnv:
                EEsl = output[3]
                envl = output[4]
            elif returnState and returnEnv:
                mpsl = output[3]
                envl = output[4]
            elif returnEntSpec:
                EEsl = output[3]
            elif returnState:
                mpsl = output[3]
            elif returnEnv:
                envl = output[3]

    # Lump right and left results
    if calcLeftState:
        EE = [EE,EEl]
        if returnEntSpec: EEs = [EEs,EEsl]
        if returnState: mpsList = [mps,mpsl]
        if returnEnv: env = [env,envl]

    # Return Results
    if len(Evec) == 1:
        output = [E,EE,gap]
    else:
        output = [Evec,EEvec,gapvec]
    if returnEntSpec:
        output.append(EEs)
    if returnState:
        output.append(mpsList)
    if returnEnv:
        output.append(env)
    return output

if __name__ == "__main__":
    from mpo.asep import return_mpo
    # Hamiltonian Parameters
    p = 0.1 
    alpha = 0.5      # in at left
    gamma = 1.-alpha  # Out at left
    q     = 1.-p      # Jump left
    beta  = 0.5     # Out at right
    delta = 1.-beta   # In at right
    s = -0.5
    # Get MPO
    hamParams = np.array([alpha,gamma,p,q,beta,delta,s])
    mpo = return_mpo(4,hamParams)
    # Run idmrg
    output = run_idmrg(mpo,mbd=20)
