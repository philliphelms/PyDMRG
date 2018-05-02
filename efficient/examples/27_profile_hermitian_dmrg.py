import cProfile
import mps_opt
import pstats

x = mps_opt.MPS_OPT(N=[10,10],
                    hamType = "heis_2d",
                    maxBondDim=[500,1000,2000,3000,4000,5000],
                    maxIter = 2,
                    verbose = 5,
                    max_eig_iter = 2,
                    hamParams = (1,1))

cProfile.run('x.kernel()','mps_stats')

p = pstats.Stats('mps_stats')
p.sort_stats('cumulative').print_stats(20)
