import numpy as np
from tools.mps_tools import save_mps,load_mps
import h5py

# Purpose of this module:
#   Take MPS as stored in .npz file format and 
#   convert into hdf5 format to be used in outside
#   applications (i.e. c++ contraction code)

def npz2hdf(fname):
    # Load file
    mps,gSite = load_mps(fname,fformat='npz')
    # Create hdf5 file to save
    save_mps(mps,fname,gaugeSite=gSite,fformat='hdf5')

def test_npz2hdf(fname,tol=1e-8):
    # Load file
    mps_npz,gSite = load_mps(fname,fformat='npz')
    # Create hdf5 file to save
    save_mps(mps_npz,fname,gaugeSite=gSite,fformat='hdf5')
    # Load the new file
    mps_hdf5,gSite = load_mps(fname,fformat='hdf5')
    # Check that they are the same
    total_diff = 0
    for state in range(len(mps_npz)):
        for site in range(len(mps_npz[0])):
            total_diff += np.sum(np.sum(np.sum(np.abs(mps_npz[state][site]-mps_hdf5[state][site]))))
    if total_diff < 1e-8:
        print('Passed Test')
    else:
        print('Failed Test')
