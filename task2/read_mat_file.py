import scipy.io
import h5py
import numpy as np
from scipy.sparse import csr_matrix

def load_mat_file(filename):
    """
    Load a .mat file from sparse.tamu.edu and return the sparse matrix A.
    Handles both v7 and v7.3 (HDF5) formats.
    """
    mat_data = scipy.io.loadmat(filename)

    problem = mat_data['Problem'][0, 0]
    A = problem['A']
    
    return A

# Example usage
if __name__ == "__main__":
    filename = "lp_capri.mat"  # Replace with your .mat file
    A = load_mat_file(filename)
    
    if A is not None:
        # Display matrix info
        print(f"Matrix type: {type(A)}")
        print(f"Shape: {A.shape}")
        print(f"Non-zero elements: {A.nnz if scipy.sparse.issparse(A) else np.count_nonzero(A)}")