import unittest
import numpy as np
import time
import mps_opt

class pydmrg_test(unittest.TestCase):

    def test_01_tasep_current(self):
        N = 4
        a = 0.35
        b = 2/3
        s = -1
        ds = .01
        # Create MPS object
        x0 = mps_opt.MPS_OPT(N = N,
                            maxBondDim = 20,
                            hamType = 'tasep',
                            verbose = 1,
                            leftMPS = True,
                            hamParams = (a,s,b))
        E0 = x0.kernel()
        current = x0.current
        susc = x0.susc
        print('Susc 0 {}'.format(x0.susc))
        # Compare to actual current
        x1  = mps_opt.MPS_OPT(N = N,
                              maxBondDim = 100,
                              hamType = 'tasep',
                              verbose = 1,
                              leftMPS = False,
                              hamParams = (a,s+ds,b))
        E1 = x1.kernel()
        print('Susc 1 {}'.format(x1.susc))
        x2  = mps_opt.MPS_OPT(N = N,
                              maxBondDim = 100,
                              hamType = 'tasep',
                              verbose = 1,
                              leftMPS = False,
                              hamParams = (a,s-ds,b))
        E2 = x2.kernel()
        print('Susc 2 {}'.format(x2.susc))
        print('D1 = {}'.format((E2-E0)/ds))
        print('D2 = {}'.format((E0-E1)/ds))
        print('D3 = {}'.format( (((E2-E0)/ds) - ((E0-E1)/ds) )/ds ))
        der_current = (E2-E1)/(2*ds)
        der_susc = (E1-2*E0+E2)/(ds**2)
        print('Operator Current: {}'.format(current))
        print('CGF Der  Current: {}'.format(der_current))
        print('Operator Suscept: {}'.format(susc))
        print('CGF Der  Suscept: {}'.format(der_susc))
        self.assertTrue(np.abs(der_current-current) < 1e-3)
        self.assertTrue(np.abs(der_susc-susc) < 1e-3)

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
