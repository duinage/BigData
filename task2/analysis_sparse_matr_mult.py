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


plt.figure()
plt.plot(process_counts, execution_times, marker='o', linestyle='-', color='b')
plt.title("Execution Time vs. Number of Processes")
plt.xlabel("Number of Processes")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.savefig("time_vs_processes.png")
plt.show()