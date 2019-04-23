from idmrg import *
from mpo.asep import return_mpo, curr_mpo
import time
import os

# Set Calculation Parameters
mbd = 6 # Can only be a single value currently

# Jumping Rates
p = 0.1
alpha = 0.5      # in at left
gamma = 1.-alpha  # Out at left
q     = 1.-p      # Jump left
beta  = 0.5     # Out at right
delta = 1.-beta   # In at right
s0 = -0.5
hamParams = np.array([alpha,gamma,p,q,beta,delta,s0])

# Calculation Settings
leftState = True
nStates = 1
alg = 'davidson'

# Create directory for storing states
dirid = str(int(time.time()))
path = 'saved_states/simpleInfinite_'+'mbd'+str(mbd)+'_'+dirid+'/'
os.mkdir(path)
fname = path+'MPS_'


# Set up MPO
mpo = return_mpo(4,hamParams)
MPOc = curr_mpo(4,hamParams,singleBond=True,bond=1)
MPOc = return_bulk_mpo(MPOc)

# Create callback function
def callFunc(output,outputl):
    # Extract outputs
    (N ,E ,Eloc ,EE ,EEs ,vecs ,mps ,mpo ,env ,S ,cont ,conv ,iterCnt ,fid ) = output
    (Nl,El,Elocl,EEl,EEsl,vecsl,mpsl,mpol,envl,Sl,contl,convl,iterCntl,fidl) = outputl
    # Reshape results in vecs
    (_,_,n1,_) = mpo[0][0].shape
    (_,_,n2,_) = mpo[0][1].shape
    (n3,_,_) = env[0][0].shape
    (n4,_,_) = env[0][1].shape
    sz = vecs[:,0].size
    bond_dim = int(np.sqrt(sz/4))
    psi = np.reshape(vecs[:,0],(2,2,bond_dim,bond_dim))
    psil = np.reshape(vecsl[:,0],(2,2,bond_dim,bond_dim))
    # Calculate norm
    norm = einsum('ijkl,ijkl',psi,psil)
    # Calculate local energy
    Eloc = einsum('ijkl,mnoi,npqj,oqkl->',psi,np.expand_dims(mpo[0][0][-1,:,:,:],axis=0),np.expand_dims(mpo[0][1][:,0,:,:],axis=1),psil)
    Eloc = np.real(Eloc/norm)
    # Calculate local density
    n = np.array([[[[0.,0.],
                    [0.,1.]]]])
    I = np.array([[[[1.,0.],
                    [0.,1.]]]])
    rhoL = einsum('ijkl,mnoi,npqj,oqkl->',psi,n,I,psil)
    rhoR = einsum('ijkl,mnoi,npqj,oqkl->',psi,I,n,psil)
    rhoL = np.real(rhoL/norm)
    rhoR = np.real(rhoR/norm)
    # Calculate local current
    curr = einsum('ijkl,mnoi,npqj,oqkl->',psi,MPOc[0][0],MPOc[0][1],psil)
    curr = np.real(curr/norm)
    return '\tEloc2={:f}\trho=({:f},{:f})\tcurr={:f}'.format(Eloc,rhoL,rhoR,curr)

# Run Optimizaton
Etmp,EEtmp,gaptmp = run_idmrg(mpo,
                                 mbd=mbd,
                                 minIter = 1,
                                 maxIter = 1000,
                                 fname=fname+'s0',
                                 nStates=nStates,
                                 alg=alg,
                                 callFunc=callFunc,
                                 calcLeftState=leftState)
