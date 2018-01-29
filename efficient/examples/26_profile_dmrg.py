import cProfile
import mps_opt
import pstats

x = mps_opt.MPS_OPT(N = 10**2,
                    hamType = 'sep_2d',
                    maxBondDim = 20,
                    maxIter = 1,
                    verbose=5,
                    usePyscf = False,
                    plotExpVals = True,
                    plotConv = True,
                    hamParams = (0.5,0.5,0.9,0.2,0.2,0.8,
                                 0.5,0.5,0.9,0.2,0.2,0.8,-1))

#cProfile.run('x.kernel()','mps_stats')
x.kernel()
#p = pstats.Stats('mps_stats')
#p.sort_stats('cumulative').print_stats(20)
