import unittest
import numpy as np
import time
import mps_opt

class pydmrg_test(unittest.TestCase):

    def test_01_tasep_current(self):
        N = 8
        a = 0.35
        b = 2/3
        s = -1
        ds = .001
        # Create MPS object
        x = mps_opt.MPS_OPT(N = N,
                            maxBondDim = 20,
                            hamType = 'tasep',
                            verbose = 1,
                            leftMPS = True,
                            hamParams = (a,s,b))
        x.kernel()
        current = x.current
        # Compare to actual current
        x1  = mps_opt.MPS_OPT(N = N,
                              maxBondDim = 100,
                              hamType = 'tasep',
                              verbose = 1,
                              leftMPS = False,
                              hamParams = (a,s+ds,b))
        E1 = x1.kernel()
        x2  = mps_opt.MPS_OPT(N = N,
                              maxBondDim = 100,
                              hamType = 'tasep',
                              verbose = 1,
                              leftMPS = False,
                              hamParams = (a,s-ds,b))
        E2 = x2.kernel()
        der_current = (E2-E1)/(2*ds)
        print('Operator Current: {}'.format(current))
        print('CGF Der  Current: {}'.format(der_current))
        self.assertTrue(np.abs(der_current-current) < 1e-3)

    def test_02_sep_current(self):
        N = 8
        a = 0.35
        g = 0.
        p = 1.
        q = 0.
        b = 0.
        d = 2/3
        s = -1
        ds = .001
        # Create MPS object
        x = mps_opt.MPS_OPT(N = N,
                            maxBondDim = 20,
                            hamType = 'sep',
                            verbose = 1,
                            leftMPS = True,
                            hamParams = (a,g,p,q,b,d,s))
        x.kernel()
        current = x.current
        # Compare to actual current
        x1  = mps_opt.MPS_OPT(N = N,
                              maxBondDim = 100,
                              hamType = 'sep',
                              verbose = 1,
                              leftMPS = False,
                              hamParams = (a,g,p,q,b,d,s+ds))
        E1 = x1.kernel()
        x2  = mps_opt.MPS_OPT(N = N,
                              maxBondDim = 100,
                              hamType = 'sep',
                              verbose = 1,
                              leftMPS = False,
                              hamParams = (a,g,p,q,b,d,s-ds))
        E2 = x2.kernel()
        der_current = (E2-E1)/(2*ds)
        print('Operator Current: {}'.format(current))
        print('CGF Der  Current: {}'.format(der_current))
        self.assertTrue(np.abs(der_current-current) < 1e-3)

    def test_03_sep_current(self):
        N = 8
        a = 0.
        g = 2/3
        p = 0.
        q = 1.
        b = 0.35
        d = 0.
        s = 1
        ds = .001
        # Create MPS object
        x = mps_opt.MPS_OPT(N = N,
                            maxBondDim = 20,
                            hamType = 'sep',
                            verbose = 1,
                            leftMPS = True,
                            hamParams = (a,g,p,q,b,d,s))
        x.kernel()
        current = x.current
        # Compare to actual current
        x1  = mps_opt.MPS_OPT(N = N,
                              maxBondDim = 100,
                              hamType = 'sep',
                              verbose = 1,
                              leftMPS = False,
                              hamParams = (a,g,p,q,b,d,s+ds))
        E1 = x1.kernel()
        x2  = mps_opt.MPS_OPT(N = N,
                              maxBondDim = 100,
                              hamType = 'sep',
                              verbose = 1,
                              leftMPS = False,
                              hamParams = (a,g,p,q,b,d,s-ds))
        E2 = x2.kernel()
        der_current = (E2-E1)/(2*ds)
        print('Operator Current: {}'.format(current))
        print('CGF Der  Current: {}'.format(der_current))
        self.assertTrue(np.abs(der_current-current) < 1e-3)

if __name__ == "__main__":
        unittest.main()
