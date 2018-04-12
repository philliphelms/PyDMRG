import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt 
from sys import argv

#-----------------------------------------------------------------------------
# In this script, we use the mpo module to calculation the hamiltonian
# for various models (1D SEP, 2D SEP, 1D Heis, 2D Heis) and print each of
# these to the terminal
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('fivethirtyeight') #'fivethirtyeight') #'ggplot'

#-----------------------------------------------------------------------------
# 1D SEP
#-----------------------------------------------------------------------------
N = 8
x = mps_opt.MPS_OPT(N=N,
                    hamType = "sep",
                    hamParams = (np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand()))
x.initialize_containers()
x.generate_mpo()
full_ham = x.mpo.return_full_ham()
E_ed,_ = np.linalg.eig(full_ham)
print('Energy via Exact Diagonalization:\n{}'.format(np.sort(E_ed)[-1]))
plt.figure(1)
plt.subplot(221)
plt.spy(full_ham)
plt.title('1D SEP')
plt.figure(2)
plt.subplot(221)
plt.imshow(full_ham)
plt.colorbar()
plt.title('1D SEP')
print('1D SEP Hamiltonian Size: {}'.format(full_ham.shape))
if (np.conj(full_ham).transpose() == full_ham).all():
    print('Hamiltonian is Hermitian')
else:
    print('Hamiltonian is not Hermitian')
print('\n\nRun DMRG Calculation for Comparison')
E_dmrg = x.kernel()



#-----------------------------------------------------------------------------
# 2D SEP
#-----------------------------------------------------------------------------
N=2
x = mps_opt.MPS_OPT(N = [N,N],
                    hamType    ="sep_2d",
                    hamParams  = (np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand(),
                                  np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand()))
x.initialize_containers()
x.generate_mpo()
full_ham = x.mpo.return_full_ham()
E_ed,_ = np.linalg.eig(full_ham)
print('Energy via Exact Diagonalization: {}'.format(np.sort(E_ed)[-1]))
plt.figure(1)
plt.subplot(222)
plt.spy(full_ham)
plt.title('2D SEP')
plt.figure(2)
plt.subplot(222)
plt.imshow(full_ham)
plt.colorbar()
plt.title('2D SEP')
print('2D SEP Hamiltonian Size: {}'.format(full_ham.shape))
if (np.conj(full_ham).transpose() == full_ham).all():
    print('\tHamiltonian is Hermitian')
else:
    print('\tHamiltonian is not Hermitian')
E_dmrg = x.kernel()
"""
#-----------------------------------------------------------------------------
# 1D Heis
#-----------------------------------------------------------------------------
x = mps_opt.MPS_OPT(N = N**2,
                    hamType = "heis",
                    hamParams = (np.random.rand(),np.random.rand()))
x.generate_mpo()
full_ham = x.mpo.return_full_ham()
plt.figure(1)
plt.subplot(223)
plt.spy(full_ham)
plt.title('1D HEIS')
plt.figure(2)
plt.subplot(223)
plt.imshow(full_ham)
plt.colorbar()
plt.title('1D HEIS')
print('1D Heisenberg Hamiltonian Size: {}'.format(full_ham.shape))
if (np.conj(full_ham).transpose() == full_ham).all():
    print('\tHamiltonian is Hermitian')
else:
    print('\tHamiltonian is not Hermitian')

#-----------------------------------------------------------------------------
# 2D Heis
#-----------------------------------------------------------------------------
x = mps_opt.MPS_OPT(N=N**2,
                    hamType = "heis_2d",
                    hamParams = (np.random.rand(),np.random.rand()))
x.generate_mpo()
full_ham = x.mpo.return_full_ham()
plt.figure(1)
plt.subplot(224)
plt.spy(full_ham)
plt.title('2D HEIS')
plt.figure(2)
plt.subplot(224)
plt.imshow(full_ham)
plt.colorbar()
plt.title('2D HEIS')
print('1D Heisenberg Hamiltonian Size: {}'.format(full_ham.shape))
if (np.conj(full_ham).transpose() == full_ham).all():
    print('\tHamiltonian is Hermitian')
else:
    print('\tHamiltonian is not Hermitian')
plt.show()
"""
