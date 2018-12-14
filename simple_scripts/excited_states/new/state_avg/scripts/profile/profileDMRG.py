import cProfile
from dmrg import *
from mpo.asep import return_mpo
import pstats

# Set Calculation Parameters
N = 50
p = 0.1 
mbd = np.array([16])
s = 0.5

# Set up calculation
mpo = return_mpo(N,(0.5,0.5,p,1-p,0.5,0.5,s))
cProfile.run('E,EE,gap = run_dmrg(mpo,mbd=mbd,nStates=2,fname="tmp")','mps_stats')

p = pstats.Stats('mps_stats')
p.sort_stats('cumulative').print_stats(20)
