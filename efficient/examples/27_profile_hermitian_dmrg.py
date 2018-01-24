import cProfile
import mps_opt
import pstats

x = mps_opt.MPS_OPT(N=10**2,
                    hamType = "heis_2d",
                    maxBondDim=100,
                    maxIter = 3,
                    verbose = 5,
                    hamParams = (1,0))

cProfile.run('x.kernel()','mps_stats')

p = pstats.Stats('mps_stats')
p.sort_stats('cumulative').print_stats(20)
