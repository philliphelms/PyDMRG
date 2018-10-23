import unittest
import numpy as np
import time
import mps_opt

# ADDITIONAL TESTS NEEDED ---------------------------
# Check for excited states
# Check left eigenstates
# Check densities are correct
# ---------------------------------------------------

class pydmrg_test(unittest.TestCase):

    def test_simple_tasep(self):
        # Just test a general tasep calculation
        x = mps_opt.MPS_OPT(N=10,
                            hamType='tasep',
                            verbose=100,
                            calc_psi=True,
                            maxBondDim = [10,20,30],
                            tol = [1e-2,1e-5,1e-8],
                            maxIter = [5,3,2],
                            hamParams=(0.35,-1.,2./3.))
        E = x.kernel()
        Ecomp = 5.378
        self.assertTrue(np.isclose(E,Ecomp))

    def test_init_guess_tasep(self):
        # Test all types of initial guesses
        print('entering this spot')
        x = mps_opt.MPS_OPT(N=10,
                            hamType='tasep',
                            verbose=1,
                            maxBondDim = 5,
                            tol = 1e-4,
                            maxIter = 10,
                            initialGuess = 'rand',
                            hamParams=(0.35,-1.,2./3.))
        Erand = x.kernel()
        Ecomp = 5.378
        x = mps_opt.MPS_OPT(N=10,
                            hamType='tasep',
                            verbose=1,
                            maxBondDim = 5,
                            tol = 1e-4,
                            maxIter = 10,
                            initialGuess = 'ones',
                            hamParams=(0.35,-1.,2./3.))
        Eones = x.kernel()
        Ecomp = 5.378
        x = mps_opt.MPS_OPT(N=10,
                            hamType='tasep',
                            verbose=1,
                            maxBondDim = 5,
                            tol = 1e-4,
                            maxIter = 10,
                            initialGuess = 0.25,
                            hamParams=(0.35,-1.,2./3.))
        Eval = x.kernel()
        Ecomp = 5.378
        self.assertTrue(np.isclose(Eval,Ecomp,rtol=1e-3))
        self.assertTrue(np.isclose(Erand,Ecomp,rtol=1e-3))
        self.assertTrue(np.isclose(Eones,Ecomp,rtol=1e-3))

    def test_general_sep(self):
        # test general sep calculations for multiple lattice sizes
        N_vec = np.array([5,10,15,20])
        E = np.zeros(N_vec.shape)
        for i in range(len(N_vec)):
            N = int(N_vec[i])
            x = mps_opt.MPS_OPT(N=N,
                                hamType='sep',
                                maxBondDim = 10,
                                tol = 1e-4,
                                verbose = 4,
                                hamParams = (0.5,0.5,0.2,0.8,0.8,0.5,-5))
            E[i] = x.kernel()
        print('Here is E {}'.format(E))
        self.assertTrue(np.isclose(E,np.array([84.850,126.914,171.224,216.122]),rtol=1e-3).all())

    def test_1d_heis(self):
        # Check 1D Heis and check on adding noise, max_eig_iter, and outputFile
        x = mps_opt.MPS_OPT(N=int(10),
                            add_noise=True,
                            max_eig_iter=100,
                            outputFile='tests/heisTestResult',
                            hamType = "heis",
                            periodic_x = True,
                            hamParams = (1,0))
        E = x.kernel()
        Ecomp = -4.515394791198144
        self.assertTrue(np.isclose(E,Ecomp,rtol=1e-3))

    def test_reverse_tasep(self):
        # Check that tasep, sep and reverse sep give same results
        # Check plotting
        x = mps_opt.MPS_OPT(N=20,
                            hamType = "sep",
                            plotExpVals = True,
                            plotConv = True,
                            hamParams = (2/3,0,1,0,0,0.35,-1))
        Esep = x.kernel()
        x = mps_opt.MPS_OPT(N=20,
                            hamType = "sep",
                            plotExpVals = True,
                            plotConv = True,
                            hamParams = (0,0.35,0,1,2/3,0,1))
        Ersep = x.kernel()
        x = mps_opt.MPS_OPT(N=20,
                            plotExpVals = True,
                            plotConv = True,
                            hamType='tasep',
                            hamParams=(0.35,-1.,2./3.))
        E = x.kernel()
        self.assertTrue(np.isclose(E,Esep,rtol=1e-5))
        self.assertTrue(np.isclose(E,Ersep,rtol=1e-5))

    def test_2d_heis(self):
        x = mps_opt.MPS_OPT(N=[3,7],
                            hamType = "heis_2d",
                            verbose = 4,
                            periodic_x = True,
                            periodic_y = True,
                            maxBondDim=10,
                            hamParams = (1,0))
        E = x.kernel()
        Ecomp = -11.110057069936254
        self.assertTrue(np.isclose(E,Ecomp,rtol=1e-3))

    def test_ising(self):
        x = mps_opt.MPS_OPT(N=10,
                            hamType = "ising",
                            periodic_x = True,
                            hamParams = (1,0))
        E = x.kernel()
        Ecomp = -2.5
        self.assertTrue(np.isclose(E,Ecomp,rtol=1e-3))

    def test_ed(self):
        x = mps_opt.MPS_OPT(N=30,
                            hamType = "sep",
                            hamParams = (0.9,0.1,0.5,0.5,0.1,0.9,-0.5),
                            usePyscf = False)
        E1 = x.exact_diag()
        x = mps_opt.MPS_OPT(N=30,
                            hamType = "tasep",
                            hamParams = (0.8,-0.5,0.2),
                            usePyscf = False)
        E2 = x.exact_diag()
        #self.assertTrue(np.isclose(E1,,rtol=1e-3))
        #self.assertTrue(np.isclose(E2,,rtol=1e-3))

        






















if __name__ == "__main__":
    unittest.main()
