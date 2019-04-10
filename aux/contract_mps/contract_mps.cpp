#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <H5Cpp.h>
#include <complex>

using namespace H5;
using namespace std;

int mbd(int N, H5File file){
    int mbd = 0;
    for (int site = 0; site < N; ++site){
        // Get tensor as dataset
        DataSet dataset = file.openDataSet("state0/M"+std::to_string(site)+"/real");

        // Get dataspace of dataset
        DataSpace dataspace = dataset.getSpace();

        // Find size of Tensor
        hsize_t dims[3];
        int ndims = dataspace.getSimpleExtentDims(dims,NULL);
        cout << "Site: " << site << ", ndims: " << ndims << ", dims: (" << dims[0] << "," << dims[1] << "," << dims[2] << ")" << endl;

        // Specify mbd
        mbd = max( max( mbd,static_cast<int>(dims[1])), static_cast<int>(dims[2]));
    }
    return mbd;
}

vector<vector<vector<complex<double> > > > load_mat(H5File file,int site){
    // Load real part
    DataSet dataset_r = file.openDataSet("state0/M"+std::to_string(site)+"/real");
    DataSpace dataspace_r = dataset_r.getSpace();
    hsize_t dims_r[3];
    int ndims_r = dataspace_r.getSimpleExtentDims(dims_r,NULL);
    double tensor_r[dims_r[0]][dims_r[1]][dims_r[2]];
    dataset_r.read(tensor_r,H5::PredType::NATIVE_DOUBLE);
    // Load imag part
    DataSet dataset_i = file.openDataSet("state0/M"+std::to_string(site)+"/imag");
    DataSpace dataspace_i = dataset_i.getSpace();
    hsize_t dims_i[3];
    int ndims_i = dataspace_i.getSimpleExtentDims(dims_i,NULL);
    double tensor_i[dims_i[0]][dims_i[1]][dims_i[2]];
    dataset_i.read(tensor_i,H5::PredType::NATIVE_DOUBLE);
    // Transfer to complex vector
    vector< vector< vector< complex<double> > > > tensor(dims_i[0], vector<vector<complex<double> > >(dims_i[1], vector<complex<double> >(dims_i[2],0.0)));
    for (int i = 0; i<dims_i[0]; ++i){
        for (int j = 0; j<dims_i[1]; ++j){
            for (int k = 0; k<dims_i[2]; ++k){
                tensor[i][j][k] = {tensor_r[i][j][k], tensor_i[i][j][k]};
            }
        }
    }
    return tensor;
}

vector<vector<complex<double> > > mat_mult(vector<vector<complex<double> > > mat1, vector<vector<complex<double> > > mat2){
    /*
     * Multiply two matrices
     */
    // Figure out Matrix Dimensions
    size_t dims1[2] = {mat1.size(), mat1[0].size()};
    size_t dims2[2] = {mat2.size(), mat2[0].size()};
    // Declare new tensor of these dims
    vector< vector< complex<double> > > res(dims1[0], vector<complex<double> >(dims2[1],0.));
    // Loop through for multiplication
    for (int i = 0; i<dims1[0]; ++i){
        for (int j = 0; j<dims2[1]; ++j){
            for (int k = 0; k<dims1[1]; ++k){
                res[i][j] += mat1[i][k]*mat2[k][j];
            }
        }
    }
    return res;
}

complex<double> contract_mps(int N, string fname, vector<int> config){
    // Open File
    H5File file(fname,H5F_ACC_RDONLY);
    // Set up for initial Calculations
    auto ten1 = load_mat(file,0);
    auto ten2 = load_mat(file,1);
    auto mat1 = ten1[config[0]];
    auto mat2 = ten2[config[1]];
    auto res = mat_mult(mat1,mat2);
    for (int i = 1; i<N-1; ++i){
        // Load next tensor
        auto ten2 = load_mat(file,i+1);
        // Convert to matrix for local config
        vector<vector<complex<double> > > mat2 = ten2[config[i+1]];
        // Multiply Matrices
        res = mat_mult(res,mat2);
    }
    // Check sizes of result
    return res[0][0];
}
            
int main(void){
    /*
     * Load a saved MPS in an hdf5 file
     * and compute the probabilities of being in the 
     * various states.
     *
     * Compile with: h5c++ -o contract_mps contract_mps.cpp -lhdf5_cpp -lhdf5 -std=c++17
     */

    // Initialize random seed
    srand(time(NULL));

    // Inputs
    int N = 10;
    srand(time(NULL));
    string fname = "../../pydmrg/saved_states/singleLane_N10mbd10_1548458130/MPS_s0_mbd0.hdf5";

    // Generate Random Configuration
    cout << "Config: ";
    vector<int> config;
    for (int i = 0; i < N; i++){
        config.push_back(rand() % 2); // generate rand 0 or 1
        cout << config[i] << " ";
    }
    cout << endl;

    // Calculate Probability of Configuration
    complex<double> result = contract_mps(N,fname,config);
    
    // Print Result
    cout << "Result: " << result << endl;
    return 0;
}
