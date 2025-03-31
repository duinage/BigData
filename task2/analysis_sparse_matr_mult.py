import subprocess
import time
import matplotlib.pyplot as plt

execution_times = []
process_counts = range(1, 9)

for n in process_counts:
    command = f"mpiexec -n {n} python sparse_matr_mult.py"
    print(f"Running: {command}")
    
    start_time = time.perf_counter()
    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    total_time = time.perf_counter() - start_time

    execution_times.append(total_time)
    print(f"time for {n} processes: {total_time:.2f} s.")

# Calculate speedup: S(n) = T(1) / T(n)
baseline_time = execution_times[0]  # time with 1 process
speedups = [baseline_time / t for t in execution_times]

plt.figure()
plt.plot(process_counts, speedups, marker='o', linestyle='-', color='b', label=f'Baseline T(1) = {baseline_time:.2f}s')
plt.title("Speedup vs. Number of Processes")
plt.xlabel("Number of Processes")
plt.ylabel("Speedup (S(n) = T(1) / T(n))")
plt.xticks(process_counts)
plt.axhline(y=1, color='r', linestyle='--', label='No Speedup (S=1)')
plt.grid(True)
plt.legend()
plt.savefig("t_vs_s.png")
plt.show()