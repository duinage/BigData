# Sparse Matrix Multiplication: https://www.cs.cmu.edu/~scandal/cacm/node9.html

from mpi4py import MPI
from typing import List
from pathlib import Path

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

x = [1.0, 0.0, -1.0, 0.0,1.0, 0.0, -1.0, 0.0]
if rank == 0:
    A = [[(0, 2.0), (1, -1.0)],
        [(0, -1.0), (1, 2.0), (2, -1.0)],
        [(1, -1.0), (2, 2.0), (3, -1.0)],
        [(2, -1.0), (3, 2.0)],
        [(0, 2.0), (1, -1.0)],
        [(0, -1.0), (1, 2.0), (2, -1.0)],
        [(1, -1.0), (2, 2.0), (3, -1.0)],
        [(2, -1.0), (3, 2.0)]]
    A_splited = [A[i::size] for i in range(size)]
else:
    A_splited = None

A_selected = comm.scatter(A_splited, root=0)

def sparse_matr_mult(A: List[List[float]], x: List[float]) -> List[float]:
    return [sum((v * x[i] for i, v in row)) for row in A]

mult_res = sparse_matr_mult(A_selected, x)

gathered_results = comm.gather(mult_res, root=0)
if rank == 0:
    result_queues = [list(res) for res in gathered_results]
    final_result = []
    
    while any(result_queues):
        for queue in result_queues:
            if queue:
                final_result.append(queue.pop(0))
    print(f"Mult Result: {final_result}")