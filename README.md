# PyDMRG

This code was originally developed as a simple implementation of the Density Matrix Renormalization Group (DMRG) algorithm using the Matrix Product States (MPS) and Matrix Product Operator (MPO) formalisms.
As this code is further developed, I am seeking to maintain the initial simple scripts for those seeking to see how a DMRG algorithm works for simple models such as the Ising and Heisenberg Models.
In addition, a major focus of this project is the application of the DMRG algorithm to stochastic classical processes such as the simple exclusion process and its derivatives. Because of this, many of the Hamiltonians currently implemented in this work are centered in these driven diffusive processes.

## Code Organization
The code is currently divided into three different sections. First is a directory named *efficient* which contains the most efficient implementation of the mps/dmrg algorithm for a few models. Currently, the supported models are:
* Heisenberg
* Ising
* Totally Assymetric Simple Exclusion Process
In this, and all, implementations, I seek to follow the algorithm and terminology as discussed in the paper [**Density Matrix Renormalization Group in the Age of Matrix Product States**](https://arxiv.org/abs/1008.3477) (Schollwoeck, 2011).

In the second directory, *sep_exact_diagonalization*, I have adapted calculations by [Ushish Ray](http://www.stochasticphysics.org/) to perform exact diagonalization calculations. These are currently implemented in Matlab scripts, but as time progresses I will translate these to python and allow them to be run from the *efficient* code to allow comparisons.

The final directory is named *verificationCalcs* and contains the most simple implementations of the DMRG algorithm. 
This contains scripts that are only approximately 100 lines long an perform DMRG calculations in the simplest implementation I include. 
Currently, there are three scripts which run the calculations for two, four, or an arbitrary number of sites. 

## Running Calculations
### *efficient*
To run the simple code, navigate into the *simple* directory and do the following in python:
```python
import numpy as np
import simple_dmrg
x = simple_dmrg.simpleHeisDMRG(L=20, 
                               d=2, 
                               D=8, 
                               tol=1e-5, 
                               max_sweep_cnt=100, 
                               J=1, 
                               h=0)
x.calculate_ground_state()
```

### *sep_exact_diagonalization*


### *verificationCalcs*

## Future Work
* Efficiency Related
    * Improve Arnoldi Diagonalization Algorithm
    * Use Cyclops for tensor algebra
* Applications Related
    * Implement higher dimensional tensor network methods
        * PEPS
    * Implement time evolution
        * Using TEBD Algorithm
    * Implement infinite algorithm
    * Implement fully general SEP (instead of only TASEP)    

## Known Issues
There are currently no known issues - Please contact me if you come accross any problems. 
