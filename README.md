# MPS_DMRG

This code is meant to be a simple  implementation of the Density Matrix Renormalization Group (DMRG) algorithm using the Matrix Product States (MPS) and Matrix Product Operators (MPOs) formalisms. 

## Code Organization
The code in general is currently divided into two sections. First is a directory named *simple* which contains a one-page implementation of the mps/dmrg algorithm for the Heisenberg Model. 
It is meant to be extremely simple to read through and follows the algorithm and terminology as discussed in the paper [**Density Matrix Renormalization Group in the Age of Matrix Product States**](https://arxiv.org/abs/1008.3477) (Schollwoeck, 2011).

The second directory is named *efficient* where I have sought to create a more efficient and flexible implementation of the DMRG algorithm, still following the Schollwoeck paper. 
This code is currently 

## Running the Calculations
### *simple*
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

### *efficient*
I will update this section as the efficient calculation is developed further

## Future Work
* Implement time evolution
    * Using TEBD Algorithm
* Calculate expectation values of observables
    * Locally
    * Globally
* Implement infinite algorithm
* Begin implementation of 2D DMRG algorithm 

## Known Issues
* For applications of DMRG that involve non-hermitian operators, there are two key problems:
    * 
    * The left and right eigenvectors are not identical.
        * I have not yet learned how to deal with this issue.
        * DMRG has been dome for systems with non-hermitian hamiltonians in the classical formulation of DMRG, but it seems that it is difficult to translate this to the MPS formalism.
