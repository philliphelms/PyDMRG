import unittest
import numpy as np
import sys
sys.path.append("..")

class pydmrg_test(unittest.TestCase):

    def test_01_simple_tasep(self):
        from simple_tasep import run_test
        E = run_test()
        self.assertTrue(np.isclose(E,5.378072046779028))

    def test_02_vary_tasep_system_size(self):
        from vary_tasep_system_size import run_test
        Current = run_test()
        self.assertTrue(np.isclose(np.array(Current),np.array([-0.26720331,-0.26315928,-0.26064439]),atol=1e-3,rtol=1e-3).all())

    def test_03_vary_sep_system_size(self):
        from vary_sep_system_size import run_test
        Current = run_test()
        print(Current)
        self.assertTrue(np.isclose(np.array(Current),np.array([0.134814,0.136255,0.137151]),atol=1e-3,rtol=1e-3).all())

    def test_04_current_vs_s_tasep(self):
        from current_vs_s_tasep import run_test
        Current = run_test()
        self.assertTrue(np.isclose(np.array(Current),np.array([6.86269,3.89343,0.69562,0.00437]),atol=1e-3,rtol=1e-3).all())

    def test_05_vary_tasep_bond_dim(self):
        from vary_tasep_bond_dim import run_test
        E_diff = run_test()
        self.assertTrue(E_diff[1]<E_diff[0])
        self.assertTrue(E_diff[2]<E_diff[1])
        self.assertTrue(E_diff[3]<E_diff[2])

    def test_06_simple_heis(self):
        from simple_heis import run_test
        E = run_test()
        self.assertTrue(np.isclose(np.real(E),-4.515446354))
    
    def test_07_simple_sep(self):
        from simple_sep import run_test
        E1,E2 = run_test()
        self.assertTrue(np.isclose(E1,E2))
    
    def test_08_heis_2d(self):
        from heis_2d import run_test
        E = run_test()
        self.assertTrue(np.isclose(E,-4.393656))

    def test_09_simple_ising(self):
        from simple_ising import run_test
        E = run_test()
        self.assertTrue(np.isclose(E,-2.5))



if __name__ == "__main__":
    unittest.main()
