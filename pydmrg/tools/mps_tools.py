import numpy as np
from pyscf.lib import einsum
import scipy.linalg as sla
import copy
import h5py
import sys

# General Tools ########################################################

def conj_mps(mps,copyMPS=True):
    # Return the complex conjugate of the provided MPS
    if copyMPS: mps_c = copy.deepcopy(mps)
    else: mps_c = mps
    nStates = len(mps)
    for state in range(nStates):
        N = len(mps[state])
        for site in range(N):
            mps_c[state][site] = np.conj(mps[state][site])
    return mps_c

def nSites(mpsL):
    # Return the number of sites in an mps list
    return len(mpsL[0])

def maxBondDim(mpsL,state=0):
    # Return the maximum bond dimension of an mps list
    mbd = 0
    for site in range(len(mpsL[state])):
        mbd = np.max(np.array([mbd,np.max(mpsL[0][site].shape)]))
    return mbd

# Save and Load ########################################################

def load_mps_npz(fname):
    # Create a list to contain all mps
    mpsL = []
    moreStates = True
    state = 0
    while moreStates:
        try:
            npzfile = np.load(fname+'state'+str(state)+'.npz')
            # List to hold a single MPS
            M = []
            moreSites = True
            site = 0
            while moreSites:
                try:
                    M.append(npzfile['M'+str(site)])
                    site += 1
                except:
                    moreSites=False
            # Add full MPS to MPS List
            mpsL.append(M)
            state += 1
        except:
            moreStates = False
    # Specifies where the gauge is located in the MPS
    try:
        gaugeSite = npzfile['site']
    except:
        print('Fname {} not found'.format(fname+'state'+str(state)+'.npz'))
        gaugeSite = npzfile['site']
    return mpsL,gaugeSite

def load_mps_hdf5(fname):
    mpsL = []
    moreStates = True
    state = 0
    try:
        with h5py.File(fname+'.hdf5','r') as f:
            while moreStates:
                # List to hold a single MPS
                mps = []
                moreSites = True
                site = 0
                while moreSites:
                    M_datasetR = f.get('state'+str(state)+'/M'+str(site)+'/real')
                    M_datasetI = f.get('state'+str(state)+'/M'+str(site)+'/imag')
                    if M_datasetR is None:
                        if site == 0:
                            moreSites = False
                            moreStates=False
                        else:
                            moreSites=False
                    else:
                        mps.append(np.array(M_datasetR)+1.j*np.array(M_datasetI))
                        site += 1
                if moreStates:
                    mpsL.append(mps)
                state += 1
            gaugeSite = np.array(f.get('state0/site'))
    except Exception:
        print('Error: {}'.format(sys.exc_info()[0]))
    return mpsL,gaugeSite

def load_mps(fname,fformat='hdf5'):
    if fformat == 'npz':
        return load_mps_npz(fname)
    elif fformat == 'hdf5':
        return load_mps_hdf5(fname)

def save_mps_npz(mpsL,fname,gaugeSite=0):
    nStates = len(mpsL)
    nSites = len(mpsL[0])
    for state in range(nStates):
        Mdict = {}
        for site in range(nSites):
            Mdict['M'+str(site)] = mpsL[state][site]
        np.savez(fname+'state'+str(state)+'.npz',site=gaugeSite,**Mdict)

def save_mps_hdf5(mpsL,fname,gaugeSite=0,comp_opts=4):
    nStates = len(mpsL)
    nSites = len(mpsL[0])
    with h5py.File(fname+'.hdf5','w') as f:
        for state in range(nStates):
            stateGroup = f.create_group('state'+str(state))
            stateGroup.create_dataset('site',data=gaugeSite)
            for site in range(nSites):
                stateGroup.create_dataset('M'+str(site)+'/real',data=np.real(mpsL[state][site]),compression='gzip',compression_opts=comp_opts)
                stateGroup.create_dataset('M'+str(site)+'/imag',data=np.imag(mpsL[state][site]),compression='gzip',compression_opts=comp_opts)
            

def save_mps(mpsL,fname,gaugeSite=0,fformat='hdf5',comp_opts=4):
    if fname is not None:
        if fformat == 'npz':
            save_mps_npz(mpsL,fname,gaugeSite=gaugeSite)
        elif fformat == 'hdf5':
            save_mps_hdf5(mpsL,fname,gaugeSite=gaugeSite,comp_opts=comp_opts)

# Entanglement Calculations ############################################

def calc_entanglement(S):
    # Calculate von neumann entanglement entropy
    # given a vector of singular values
    #
    # Ensure correct normalization
    S /= np.sqrt(np.dot(S,np.conj(S)))
    #assert(np.isclose(np.abs(np.sum(S*np.conj(S))),1.))
    EEspec = -S*np.conj(S)*np.log2(S*np.conj(S))
    EE = np.sum(EEspec)
    return EE,EEspec

def calc_ent_right(M,v,site):
    # Given an MPS, and the vectors that go in it, 
    # Calculate the von neumann entanglement entropy
    # in the bond to the right of specified site
    (n1,n2,n3) = M[site].shape
    Mtmp = np.reshape(v,(n1,n2,n3))
    M_reshape = np.reshape(Mtmp,(n1*n2,n3))
    (_,S,_) = np.linalg.svd(M_reshape,full_matrices=False)
    EE,EEs = calc_entanglement(S)
    print('\t\tEE = {}'.format(EE))
    return EE, EEs 

def calc_ent_left(M,v,site):
    # Given an MPS, and the vectors that go in it, 
    # Calculate the von neumann entanglement entropy
    # in the bond to the left of specified site
    (n1,n2,n3) = M[site].shape
    Mtmp = np.reshape(v,(n1,n2,n3))
    M_reshape = np.swapaxes(Mtmp,0,1)
    M_reshape = np.reshape(M_reshape,(n2,n1*n3))
    (_,S,_) = np.linalg.svd(M_reshape,full_matrices=False)
    EE,EEs = calc_entanglement(S)
    print('\t\tEE = {}'.format(EE))
    return EE, EEs

# Canonicalization and gauging #######################################

def calcRDM(M,swpDir):
    # Calculate the reduced density matrix
    if swpDir == 'right':
        (n1,n2,n3) = M.shape
        M = np.swapaxes(M,0,1)
        M = np.reshape(M,(n2*n1,n3))
        return einsum('ij,kj->ik',M,np.conj(M))
    elif swpDir == 'left':
        (n1,n2,n3) = M.shape
        M = np.swapaxes(M,0,1)
        M = np.reshape(M,(n2,n1*n3))
        return einsum('ij,ik->jk',M,np.conj(M))

def make_mps_right(M):
    # Make the entire mps right canonical
    N = len(M)
    for i in range(int(N)-1,0,-1):
        M_reshape = np.swapaxes(M[i],0,1)
        (n1,n2,n3) = M_reshape.shape
        M_reshape = np.reshape(M_reshape,(n1,n2*n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M_reshape = np.reshape(V,(n1,n2,n3))
        M[i] = np.swapaxes(M_reshape,0,1)
        M[i-1] = einsum('klj,ji,i->kli',M[i-1],U,s)
    return M

def make_all_mps_right(mpsList):
    # Make a list of mps right canonical
    nStates = len(mpsList)
    for state in range(nStates):
        mpsList[state] = make_mps_right(mpsList[state])
    return mpsList

def svd_right(ten):
    # Singular value decomposition to the right
    (n1,n2,n3) = ten.shape
    ten_reshape = np.reshape(ten,(n1*n2,n3))
    (u,s,v) = np.linalg.svd(ten_reshape,full_matrices=False)
    return (u,s,v)

def move_gauge_right(mps,site,returnEE=False,returnEEs=False,targetState=0):
    (n1,n2,n3) = mps[0][site].shape
    nStates = len(mps)
    for state in range(nStates):
        (u,s,v) = svd_right(mps[state][site])
        if (state == targetState): 
            EE,EEs = calc_entanglement(s)
        mps[state][site] = np.reshape(u,(n1,n2,n3))
        mps[state][site+1] = np.einsum('i,ij,kjl->kil',s,v,mps[state][site+1])
    if returnEE and returnEEs: return mps,EE,EEs
    elif returnEE: return mps,EE
    else: 
        if returnEEs: return mps,EEs
        else: return mps

def svd_left(ten):
    # Singular value decomposition to the left
    ten_reshape = np.swapaxes(ten,0,1)
    (n1,n2,n3) = ten_reshape.shape
    ten_reshape = np.reshape(ten_reshape,(n1,n2*n3))
    (u,s,v) = np.linalg.svd(ten_reshape,full_matrices=False)
    return (u,s,v)

def move_gauge_left(mps,site,returnEE=False,returnEEs=False,targetState=0):
    (n1,n2,n3) = mps[0][site].shape
    nStates = len(mps)
    for state in range(nStates):
        (u,s,v) = svd_left(mps[state][site])
        if (returnEE or returnEEs) and (state == targetState): 
            EE,EEs = calc_entanglement(s)
        M_reshape = np.reshape(v,(n2,n1,n3))
        mps[state][site] = np.swapaxes(M_reshape,0,1)
        mps[state][site-1] = np.einsum('klj,ji,i->kli',mps[state][site-1],u,s)
    if returnEE and returnEEs: return mps,EE,EEs
    elif returnEE: return mps,EE
    else: 
        if returnEEs: return mps,EEs
        else: return mps

def move_gauge(mps,init_site,fin_site):
    if init_site < fin_site:
        # Right Sweep
        for site in range(init_site,fin_site):
            mps = move_gauge_right(mps,site)
    else:
        # Left Sweep
        for site in range(fin_site,init_site,-1):
            mps = move_gauge_left(mps,site)
    return mps

def renormR(mpsL,v,site,nStates=1,targetState=0,stateAvg=True):
    if stateAvg:
        return renormR_avg(mpsL,v,site,nStates=nStates,targetState=targetState)
    else:
        return renormR_svd(mpsL,v,site,nStates=nStates,targetState=targetState)

def renormR_svd(mpsL,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = mpsL[0][site].shape
    # Put v into mpsL
    _,nStatesCalc = v.shape
    nStatesAvailable = min(nStates,nStatesCalc)
    for state in range(nStatesAvailable):
        if nStatesAvailable != 1:
            vtmp = v[:,state]
        else:
            vtmp = v
        mpsL[state][site] = np.reshape(vtmp,(n1,n2,n3))
    mpsL,EE,EEs = move_gauge_right(mpsL,site,returnEE=True,returnEEs=True,targetState=targetState)
    return mpsL,EE,EEs

def renormR_avg(mpsL,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = mpsL[0][site].shape
    # Calculate Entanglement
    EE,EEs = calc_ent_right(mpsL[0],v[:,targetState],site)
    # Calculate the reduced density matrix
    _,nStatesCalc = v.shape
    nStatesAvg = min(nStates,nStatesCalc)
    for i in range(nStatesAvg):
        if nStatesAvg != 1: 
            vtmp = v[:,i]
        else:
            vtmp  = v
        vReshape = np.reshape(vtmp,(n1,n2,n3))
        w = 1./float(nStatesAvg)
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
    vecs = sla.orth(vecs)
    # Loop through all MPS in list
    for state in range(nStates):
        # Put resulting vectors into MPS
        mpsL[state][site] = np.reshape(vecs,(n2,n1,n3))
        mpsL[state][site] = np.swapaxes(mpsL[state][site],0,1)
        if not np.all(np.isclose(einsum('ijk,ijl->kl',mpsL[state][site],np.conj(mpsL[state][site])),np.eye(n3),atol=1e-6)):
            if VERBOSE > 4: print('\t\tNormalization Problem')
        # Calculate next site for guess
        if nStates != 1:
            vReshape = np.reshape(v[:,min(nStatesAvg-1,state)],(n1,n2,n3))
        else:
            vReshape = np.reshape(v,(n1,n2,n3))
        # PH - This next line is incorrect!!!
        mpsL[state][site+1] = einsum('lmn,lmk,ikj->inj',np.conj(mpsL[state][site]),vReshape,mpsL[state][site+1])
    return mpsL,EE,EEs

def renormL(mpsL,v,site,nStates=1,targetState=0,stateAvg=False):
    if stateAvg:
        return renormL_avg(mpsL,v,site,nStates=nStates,targetState=targetState)
    else:
        return renormL_svd(mpsL,v,site,nStates=nStates,targetState=targetState)

def renormL_svd(mpsL,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = mpsL[0][site].shape
    # Put v into mpsL
    _,nStatesCalc = v.shape
    nStatesAvailable = min(nStates,nStatesCalc)
    for state in range(nStatesAvailable):
        if nStatesAvailable != 1:
            vtmp = v[:,state]
        else:
            vtmp = v
        mpsL[state][site] = np.reshape(vtmp,(n1,n2,n3))
    mpsL,EE,EEs = move_gauge_left(mpsL,site,returnEE=True,returnEEs=True,targetState=targetState)
    return mpsL,EE,EEs

def renormL_avg(mpsL,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = mpsL[0][site].shape
    # Try to calculate Entanglement?
    EE,EEs = calc_ent_left(mpsL[0],v[:,targetState],site)
    # Calculate the reduced density matrix
    _,nStatesCalc = v.shape
    nStatesAvg = min(nStates,nStatesCalc)
    for i in range(nStatesAvg):
        if nStatesAvg != 1: 
            vtmp = v[:,i]
        else: 
            vtmp = v
        vReshape = np.reshape(vtmp,(n1,n2,n3))
        w = 1./float(nStatesAvg)
        if i == 0:
            rdm = w*calcRDM(vReshape,'left') 
        else:
            rdm +=w*calcRDM(vReshape,'left')
    # Take eigenvalues of the rdm
    vals,vecs = np.linalg.eig(rdm) 
    # Sort inds
    inds = np.argsort(vals)[::-1]
    # Keep only maxBondDim eigenstates
    inds = inds[:n2]
    vals = vals[inds]
    vecs = vecs[:,inds]
    # Make sure vecs are orthonormal
    vecs = sla.orth(vecs)
    vecs = vecs.T
    # Loops through all MPSs in list
    for state in range(nStates):
        # Put resulting vectors into MPS
        mpsL[state][site] = np.reshape(vecs,(n2,n1,n3))
        mpsL[state][site] = np.swapaxes(mpsL[state][site],0,1)
        if not np.all(np.isclose(einsum('ijk,ilk->jl',mpsL[state][site],np.conj(mpsL[state][site])),np.eye(n2),atol=1e-6)):
            if VERBOSE > 4: print('\t\tNormalization Problem')
        # Calculate next site's guess
        if nStates != 1:
            vReshape = np.reshape(v[:,min(nStatesAvg-1,state)],(n1,n2,n3))
        else:
            vReshape = np.reshape(v,(n1,n2,n3))
        # Push gauge onto next site
        mpsL[state][site-1] = einsum('ijk,lkm,lnm->ijn',mpsL[state][site-1],vReshape,np.conj(mpsL[state][site]))
    return mpsL,EE,EEs

# Infinite iDMRG tools ############################################

def svd_right_inf(ten):
    (n1,n2,n3) = ten.shape
    ten = np.swapaxes(ten,0,1)
    ten_reshape = np.reshape(ten,(n1*n2,n3))
    (u,s,v) = np.linalg.svd(ten_reshape,full_matrices=False)
    return (u,s,v)

#(mpsL,v,site,nStates=1,targetState=0,stateAvg=True):
def renormInf(mpsList,mpoList,envList,
               vecs,mbd,
               targetState=0,stateAvg=True):
    (n1,n3,_) = mpsList[0][0].shape
    (n2,_,n4) = mpsList[0][1].shape
    nStates = len(mpsList)
    for state in range(nStates):
        # Reshape Matrices
        psi = np.reshape(vecs[:,state],(n1,n2,n3,n4))
        psi = np.transpose(psi,(2,0,1,3))
        psi = np.reshape(psi,(n3*n1,n4*n2))
        # Do SVD
        (U,S,V) = np.linalg.svd(psi,full_matrices=False)
        U = np.reshape(U,(n3,n1,-1))
        U = U[:,:,:mbd]
        mpsList[state][0] = np.swapaxes(U,0,1)
        V = np.reshape(V,(-1,n2,n4))
        V = V[:mbd,:,:]
        mpsList[state][1] = np.swapaxes(V,0,1)
        S = S[:mbd]
        EE,EEs = calc_entanglement(S)
        if state == targetState:
            EEr,EEsr,Sr = EE,EEs,S
    return mpsList,EEr,EEsr,Sr

# Create MPS ##########################################################

def create_rand_mps(N,mbd,d=2,const=1e-2):
    # Create MPS
    M = []
    for i in range(int(N/2)):
        M.insert(len(M),const*np.random.rand(d,min(d**(i),mbd),min(d**(i+1),mbd)))
    if N%2 is 1:
        M.insert(len(M),const*np.random.rand(d,min(d**(i+1),mbd),min(d**(i+1),mbd)))
    for i in range(int(N/2))[::-1]:
        M.insert(len(M),const*np.random.rand(d,min(d**(i+1),mbd),min(d**i,mbd)))
    return M

def increase_all_mbd(mpsL,mbd,periodic=False,constant=False,d=2):
    # Increase the max bond dimension for all sites
    nStates = len(mpsL)
    for state in range(nStates):
        mpsL[state] = increase_mbd(mpsL[state],mbd,periodic=periodic,constant=constant,d=d)
    return mpsL

def create_const_mps(N,mbd,d=2,const=1e-2):
    # Create MPS
    M = []
    for i in range(int(N/2)):
        M.insert(len(M),const*np.ones((d,min(d**(i),mbd),min(d**(i+1),mbd))))
    if N%2 is 1:
        M.insert(len(M),const*np.ones((d,min(d**(i+1),mbd),min(d**(i+1),mbd))))
    for i in range(int(N/2))[::-1]:
        M.insert(len(M),const*np.ones((d,min(d**(i+1),mbd),min(d**i,mbd))))
    return M

def create_all_mps(N,mbd,nStates,rand=True,const=1e-2,d=2):
    mpsList = []
    for state in range(nStates):
        if rand:
            mpsList.append(create_rand_mps(N,mbd,d=d,const=const))
        else:
            mpsList.append(create_const_mps(N,mbd,d=d,const=const))
    return mpsList

def increase_mbd(M,mbd,periodic=False,constant=False,d=2):
    N = len(M)
    if periodic == False:
        if constant == False:
            for site in range(int(N/2)):
                nx,ny,nz = M[site].shape
                sz1 = min(d**site,mbd)
                sz2 = min(d**(site+1),mbd)
                M[site] = np.pad(M[site], ((0,0), (0,sz1-ny), (0,sz2-nz)), 'constant', constant_values=0j)
            if N%2 is 1:
                site += 1
                nx,ny,nz = M[site].shape
                sz1 = min(d**(site),mbd)
                sz2 = min(d**(site),mbd)
                M[site] = np.pad(M[site], ((0,0), (0,sz1-ny), (0,sz2-nz)), 'constant', constant_values=0j)
            for i in range(int(N/2))[::-1]:
                site = N - i - 1
                nx,ny,nz = M[site].shape
                sz1 = min(d**(N-(site)),mbd)
                sz2 = min(d**(N-(site+1)),mbd)
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

# Other Functions ########################################################

def orthonormalize_states(mps,mpo=None,gSite=None,printEnergies=False):
    # Orthonormalize matrix product states
    from tools.contract import full_contract as contract
    # Load matrix product states
    if isinstance(mps,str):
        mps,gSite = load_mps(mps)
    N = len(mps[0])
    nStates = len(mps)
    # Calculate energy of states before orthonormalization
    if mpo is not None:
        E0 = [0]*len(mps)
        for i in range(len(E0)):
            E0[i] = contract(mps=mps,mpo=mpo,gSite=gSite,state=i)/contract(mps=mps,gSite=gSite,state=i)
        if printEnergies: print('Initial Energies = {}'.format(E0))
    # Get shape of center matrix
    (n1,n2,n3) = mps[0][gSite].shape
    # Put the vector of MPS center sites into a matrix
    vecsList = []
    for state in range(nStates):
        vecsList.append(np.reshape(mps[state][gSite],-1))
    # Put those lists into a matrix
    vecs = np.zeros((len(vecsList[0]),nStates),dtype=np.complex_)
    for state in range(nStates):
        vecs[:,state] = vecsList[state]
    # Orthonormalize the states
    vecs = sla.orth(vecs,rcond=1e-100) # PH - Something is wrong here?
    # Put back into the MPS
    for state in range(nStates):
        mps[state][gSite] = np.reshape(vecs[:,state],(n1,n2,n3))
    # Check if the orthonormalization altered the energy
    if mpo is not None:
        Ef = [0]*len(mps)
        for i in range(len(E0)):
            Ef[i] = contract(mps=mps,mpo=mpo,gSite=gSite,state=i)/contract(mps=mps,gSite=gSite,state=i)
        if printEnergies: print('Final Energies = {}'.format(Ef))
    return mps

def list_occs(N):
    occ = np.zeros((2**N,N),dtype=int)
    for i in range(2**N):
        occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
    return occ

def return_mat_dims(N,mbd):
    fbd_site = []
    mbd_site = []
    fbd_site.insert(0,1)
    mbd_site.insert(0,1)
    for i in range(int(N/2)):
        fbd_site.insert(-1,2**i)
        mbd_site.insert(-1,min(2**i,mbd))
    for i in range(int(N/2))[::-1]:
        fbd_site.insert(-1,2**(i+1))
        mbd_site.insert(-1,min(2**(i+1),mbd))
    return fbd_site,mbd_site

def state2mps(N,psi,mbd,return_ee=True):
    psi = np.reshape(psi,[2]*N)
    fbd_site,mbd_site = return_mat_dims(N,mbd)
    mps = []
    EE = np.zeros(N-1)
    for i in range(N,1,-1):
        psi = np.reshape(psi,(2**(i-1),-1))
        (u,s,v) = np.linalg.svd(psi,full_matrices=False)
        B = np.reshape(v,(-1,2,mbd_site[i]))
        B = B[:mbd_site[i-1],:,:mbd_site[i]]
        B = np.swapaxes(B,0,1)
        mps.insert(0,B)
        psi = np.einsum('ij,j->ij',u[:,:mbd_site[i-1]],s[:mbd_site[i-1]])
        # Calculate & Print Entanglement Entropy
        EE[i-2] = -np.dot(s**2.,np.log2(s**2.))
    #assert(np.isclose(eer[int(N/2)-1],old_entanglement))
    mps.insert(0,np.reshape(psi,(2,1,min(2,mbd))))
    if return_ee:
        return mps,EE[int(N/2)-1]
    else:
        return mps

def contract_config(mps,config,norm='L1',state=0,gSite=None):
    # Contract a given configuration
    from tools.contract import full_contract as contract
    # Load matrix product states
    if isinstance(mps,str):
        mps,gSite = load_mps(mps)
    N = len(mps[0])
    nStates = len(mps)
    # Check Normalization
    if norm == 'L1':
        mps[state][gSite-1] /= np.einsum('ijk->',mps[state][gSite-1])
        #print('L1 Norm',np.einsum('ijk->',mps[state][gSite-1]))
    else:
        mps[state][gSite-1] /= np.einsum('ijk,ijk->',mps[state][gSite-1],np.conj(mps[state][gSite-1]))
        #print('L2 Norm',np.einsum('ijk,ijk->',mps[state][gSite-1],np.conj(mps[state][gSite-1])))
    # Contract MPS
    res = np.array([[1]])
    for site in range(N):
        res = np.dot(res,mps[state][site][config[site],:,:])
    return res[0,0]

def all_config_prob(mps,norm='L2',state=0):
    # Find the probability of all possible configurations
    # 
    # Load MPS
    if isinstance(mps,str):
        mps,gSite = load_mps(mps)
    N = len(mps[0])
    nStates = len(mps)
    prob = np.zeros(2**N,dtype=np.complex_)
    for i in range(2**N):
        config = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
        prob[i] = contract_config(mps,config,norm=norm,state=state,gSite=gSite)
        print(str(i)+"\t'"+'0'*(N-len(bin(i)[2:]))+bin(i)[2:]+'\t'+str(np.real(prob[i]))+'\t'+str(np.imag(prob[i])))
    return prob
