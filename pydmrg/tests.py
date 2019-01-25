import unittest
import numpy as np

class pydmrg_test(unittest.TestCase):
    def test_arnoldiCheck(self):
        import tests.asep.arnoldiCheck as eigCheck
        E1,E2 = eigCheck.run_test()
        self.assertTrue(np.isclose(E1,E2,atol=1e-4,rtol=1e-4),
                        'Exact ({}) and Arnoldi ({}) energies do not agree'.format(E1,E2))

    def test_davidsonCheck(self):
        import tests.asep.davidsonMultipleCheck as eigCheck
        E1,E2 = eigCheck.run_test()
        self.assertTrue(np.isclose(E1,E2,atol=1e-4,rtol=1e-4),
                        'Exact ({}) and Davidson ({}) energies do not agree'.format(E1,E2))

    def tests_multipleMBD(self):
        import tests.asep.multipleBondDim as mbd
        Ee1,Ee2,Ea1,Ea2,Ed1,Ed2 = mbd.run_test()
        self.assertTrue(np.isclose(Ee1[0],Ee2[0]),'Exact Energies do not agree for d=2 ({},{})'.format(Ee1[0],Ee2[0]))
        self.assertTrue(np.isclose(Ee1[1],Ee2[1]),'Exact Energies do not agree for d=4 ({},{})'.format(Ee1[1],Ee2[1]))
        self.assertTrue(np.isclose(Ee1[2],Ee2[2]),'Exact Energies do not agree for d=6 ({},{})'.format(Ee1[2],Ee2[2]))
        self.assertTrue(np.isclose(Ea1[0],Ea2[0]),'Arnoldi Energies do not agree for d=2 ({},{})'.format(Ea1[0],Ea2[0]))
        self.assertTrue(np.isclose(Ea1[1],Ea2[1]),'Arnoldi Energies do not agree for d=4 ({},{})'.format(Ea1[1],Ea2[1]))
        self.assertTrue(np.isclose(Ea1[2],Ea2[2]),'Arnoldi Energies do not agree for d=6 ({},{})'.format(Ea1[2],Ea2[2]))
        self.assertTrue(np.isclose(Ed1[0],Ed2[0]),'Davidson Energies do not agree for d=2 ({},{})'.format(Ed1[0],Ed2[0]))
        self.assertTrue(np.isclose(Ed1[1],Ed2[1]),'Davidson Energies do not agree for d=4 ({},{})'.format(Ed1[1],Ed2[1]))
        self.assertTrue(np.isclose(Ed1[2],Ed2[2]),'Davidson Energies do not agree for d=6 ({},{})'.format(Ed1[2],Ed2[2]))

    def test_contract_calc(self):
        print('Running Contraction Calculation Test')
        import tests.asep.contractCheck as contractCheck
        E1,E2,E3,E4 = contractCheck.run_test()
        self.assertTrue(np.isclose(E1,E2))
        self.assertTrue(np.isclose(E1,E3))
        self.assertTrue(np.isclose(E1,E4))

    def test_periodic2D(self):
        import tests.asep2D.periodicCheck as perCheck
        E1,E2,E3,E4,E5,E6,E7,E8,E9 = perCheck.run_test()
        self.assertTrue(np.isclose(E1,E2))
        self.assertTrue(np.isclose(E1,E3))
        self.assertTrue(np.isclose(E1,E4))
        self.assertTrue(np.isclose(E1,E5))
        self.assertTrue(np.isclose(E1,E6))
        self.assertTrue(np.isclose(E1,E7))
        self.assertTrue(np.isclose(E1,E8))
        self.assertTrue(np.isclose(E1,E9))

    def test_current_calc(self):
        print('Running Current Calculation Test')
        import tests.asep.currentCheck as currCheck
        opCur,opCurS,derCur = currCheck.run_test()
        self.assertTrue(np.isclose(opCur,derCur,atol=1e-3))
        self.assertTrue(np.isclose(opCurS,derCur,atol=1e-3))

if __name__ == "__main__":
    unittest.main()
