import cProfile
import mps_opt

x = mps_opt.MPS_OPT(N = 4**2,
                    hamType = 'sep_2d',
                    maxBondDim = 30,
                    verbose=2,
                    plotExpVals=False,
                    plotConv=False,
                    hamParams = (0.5,0.5,0.9,0.2,0.2,0.8,
                                 0.5,0.5,0.9,0.2,0.2,0.8,0))
cProfile.run('x.kernel()')
