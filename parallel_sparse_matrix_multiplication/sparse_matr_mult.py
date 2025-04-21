# Sparse Matrix Multiplication: https://www.cs.cmu.edu/~scandal/cacm/node9.html

from mpi4py import MPI
from typing import List
import scipy.io
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


if rank == 0:
    # A = [[(0, 2.0), (1, -1.0)],
    #     [(0, -1.0), (1, 2.0), (2, -1.0)],
    #     [(1, -1.0), (2, 2.0), (3, -1.0)],
    #     [(2, -1.0), (3, 2.0)]]
    # x = [1.0, 0.0, -1.0, 0.0,1.0, 0.0, -1.0, 0.0]
    
    mat_data = scipy.io.loadmat("wiki-Talk.mat")
    problem = mat_data['Problem'][0, 0]
    A_sparse = problem['A']

    num_cols = A_sparse.shape[1]
    x = np.array([1, 0, -1] * (num_cols // 3) + [1, 0, -1][:num_cols % 3]) 

    print(f"A Shape: {A_sparse.shape}")
    print(f"Non-zero elements in A: {A_sparse.nnz}")

    # Convert sparse matrix to list of lists format: [[(col_idx, value), ...], ...]
    A = []
    A_csr = A_sparse.tocsr()
    for i in range(A_csr.shape[0]):
        row = []
        for j, v in zip(A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i+1]],
                       A_csr.data[A_csr.indptr[i]:A_csr.indptr[i+1]]):
            row.append((j, float(v)))
        A.append(row)
    
    A_splited = [A[i::size] for i in range(size)]
else:
    A_splited = None
    x = None

A_selected = comm.scatter(A_splited, root=0)
x = comm.bcast(x, root=0)

def sparse_matr_mult(A: List[List[float]], x: List[float]) -> List[float]:
    return [sum((v * x[i] for i, v in row)) for row in A]

mult_res = sparse_matr_mult(A_selected, x)

gathered_results = comm.gather(mult_res, root=0)
import numpy as np
if rank == 0:
    final_result = np.concatenate(gathered_results)
    print(f"Mult Result: {final_result}")