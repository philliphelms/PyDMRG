// contract_mps.cpp : Returns the weight from the contraction of a 
//                    single configuration of an MPS. 

#include "cnpy.h"
#include <complex>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h> // has rand
#include <time.h>
//#include <cblas.h> // Matrix multiplication

void print_mps_elem(std::vector<cnpy::NpyArray> mps, int site)
{
    std::cout << "M"+std::to_string(site) << ":\n";
    // Print Dimensions
    size_t ndims_st = mps[site].shape.size();
    int ndims = static_cast<int>(ndims_st);
    std::cout << "\tDimensions: " << ndims << "\n";
    // Print Shape
    std::cout << "\tShape: (";
    int len_arr = 1;
    for (int dim = 0; dim < ndims; dim++)
    {
        std::cout << " " << mps[site].shape[dim];
        len_arr *= mps[site].shape[dim];
    }
    std::cout << ")\n";
    // Print Tensor
    std::cout << "\tNumber of Elements: " << len_arr << "\n";
    std::cout << "\tData:\n";
    int elem = 0;
    for (int i = 0; i < mps[site].shape[0]; i++)
    {
        for (int j = 0; j < mps[site].shape[1]; j++)
        {
            for (int k = 0; k < mps[site].shape[2]; k++)
            {
                std::cout << "(" << i << "," << j << "," << k << ")";
                std::cout << "\t\t" <<  mps[site].data<std::complex<double>>()[elem] << "\n";
                elem += 1;
            };
        };
    };
    std::cout << "\n";
}

// load_mps:
//
//     Inputs:
//      N - Number of lattice sites
//      fname - "path/to/file.npz"
//
//     Returns:
//      mps - a vector of numpy arrays composing the mps
std::vector<cnpy::NpyArray> load_mps(int N, std::string fname)
{
    // Create a vector to hold the NpyArrays
    std::vector<cnpy::NpyArray> mps;
    // Loop through all N
    std::string arrName;
    for (int site = 0; site < N; site++)
    {
        arrName = "M"+std::to_string(site);
        mps.push_back(cnpy::npz_load(fname,arrName));
        print_mps_elem(mps,site);
    }
    return mps;
}

/*
std::complex<double>* mat_multiply(std::complex<double>* A, int nr_A, int nc_A, std::complex<double>* B, int nr_B, int nc_B)
{
    std::complex<double>* C = new std::complex<double>[nr_A*nc_B];
    std::complex<double> alpha(1.,0.);
    std::complex<double> beta(1.,0.);
    std::cout << "Sending to blas\n";
    std::cout << nr_A << "x" << nc_A;
    //cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,nr_A,nc_B,nc_A,&alpha,A,nr_A,B,nr_B,&beta,C,nr_A);
    cgemm_("N","N",nr_A,nc_B,nc_A,&alpha,A,nr_A,B,nr_B,&beta,C,nr_A)//CblasRowMajor, CblasNoTrans, CblasNoTrans,nr_A,nc_B,nc_A,&alpha,A,nr_A,B,nr_B,&beta,C,nr_A);
    std::cout << "Back from blas\n";
    return C;
}
*/

int mbd(std::vector<cnpy::NpyArray> mps, int N)
{
    int max = 0;
    int localMax = 0;
    for (int site = 0; site < N; site++)
    {
        std::cout << "mbd " << mps[site].shape[1];
        //max = std::max(localMax,mps[site].shape[1]);
    }
    return max;
}

std::complex<double> contract_mps(int N, std::string fname, std::vector<int> config)
{
    std::cout << "In contract_mps\n";
    std::complex<double> result;
    auto mps = load_mps(N,fname);
    std::cout << "Completed Loading MPS\n";
    std::cout << "Multiplying MPS\n";
    int maxBondDim = mbd(mps,N);
    std::complex<double>* C = new std::complex<double>[maxBondDim*maxBondDim];
    C = mps[0].data<std::complex<double>>();
    int nr_C = mps[0].shape[1];
    int nc_C = mps[0].shape[2];
    for (int site = 0; site < N-1; site++)
    {
        //C = mat_multiply(C,nr_C,nc_C,mps[site+1].data<std::complex<double>>(),mps[site+1].shape[1],mps[site+1].shape[2]);
        nr_C = mps[site+1].shape[1];
        nc_C = mps[site+1].shape[2];
    }
    // Now that it's loaded, contract matrices
    return result;
}

int main()
{
    // Initialize random seed
    srand(time(NULL));
    // Print out that we have started
    std::cout << "Running\n";
    // Set length of chain
    int N = 10;
    // Specify filename
    std::string fname = "../pydmrg/saved_states/singleLane_manyStates_N10mbd30_1550018526/MPS_s0_mbd0state0.npz";
    // Generate random configuration
    std::vector<int> config;
    for (int i = 0; i < N; i++)
        config.push_back(rand() % 2); // generate rand 0 or 1
    // Print out Configurations
    std::cout << "Configurations: \n";
    for (auto const& c : config)
        std::cout << c << " ";
    std::cout << "\n";
    // Contract the MPS
    std::complex<double> result = contract_mps(N,fname,config);
    return 0;
}
