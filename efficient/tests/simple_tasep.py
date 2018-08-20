import numpy as np
import time
from pydmrg.efficient import mps_opt

def run_test():
    # Create MPS object
    x = mps_opt.MPS_OPT(N = 10,
                        hamType = 'tasep',
                        maxBondDim = 50,
                        tol = 1e-5,
                        maxIter = 5,
                        periodic_x = False,
                        periodic_y = False,
                        plotConv = False,
                        plotExpVals = False,
                        add_noise = False,
                        hamParams = (0.35,-1,2/3))
    # Run optimization
    return x.kernel()

if __name__ == "__main__":
    run_test()
