import numpy as np
import threading
import sys
import os


def read_matrix(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        rows = int(lines[0])
        cols = int(lines[1])
        matrix = np.zeros((rows, cols))
        for i in range(2, len(lines)):
            values = lines[i].split()
            for j in range(len(values)):
                matrix[i - 2, j] = float(values[j])
    return matrix


def multiply(A, B, result_matrix, start, end, sum_lock):
    partial_sum = 0
    partial_squares_sum = 0
    for i in range(start, end):
        for j in range(B.shape[1]):
            result_matrix[i, j] += np.dot(A[i, :], B[:, j])
            partial_sum += result_matrix[i, j]
            partial_squares_sum += result_matrix[i, j] ** 2
    with sum_lock:
        global global_sum
        global global_squares_sum
        global_sum += partial_sum
        global_squares_sum += partial_squares_sum


def validate_args():
    MAX_THREADS = 4
    if len(sys.argv) != 4:
        print("Usage: python matrix.py filename_A filename_B num_threads")
        sys.exit(1)

    filename_A = sys.argv[1]
    filename_B = sys.argv[2]
    num_threads = int(sys.argv[3])

    if not (filename_A.endswith(".txt") and filename_B.endswith(".txt")):
        print("Error: Both filenames must have .txt extension")
        sys.exit(1)
    if not os.path.isfile(filename_A):
        print("Error: First file does not exist")
        sys.exit(1)
    if not os.path.isfile(filename_B):
        print("Error: Second file does not exist")
        sys.exit(1)

    if num_threads <= 0 or num_threads > MAX_THREADS:
        print("Error: Number of threads must be between 1 and", MAX_THREADS)
        sys.exit(1)

    return filename_A, filename_B, num_threads


filename_A, filename_B, num_threads = validate_args()

A = read_matrix(filename_A)
B = read_matrix(filename_B)

if A.shape[1] != B.shape[0]:
    print(
        "Error: Number of columns in first matrix must be equal to the number of rows in second matrix"
    )
    sys.exit(1)

result_matrix = np.zeros((A.shape[0], B.shape[1]))
global_sum = 0.0
global_squares_sum = 0.0
sum_lock = threading.Lock()

num_rows = result_matrix.shape[0]
data_fragments = np.array_split(np.arange(num_rows), num_threads)
data_fragments = [(fragment[0], fragment[-1] + 1) for fragment in data_fragments]

threads = []
for data_fragment in data_fragments:
    thread = threading.Thread(
        target=multiply,
        args=(
            A,
            B,
            result_matrix,
            data_fragment[0],
            data_fragment[1],
            sum_lock,
        ),
    )
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

frobenius_norm = np.sqrt(global_squares_sum)
print("Result matrix:")
print(result_matrix, "\n")

print("Sum:", global_sum)
print("Frobenius norm:", frobenius_norm)

print("\nCheck results using numpy:")
numpy_matrix = np.dot(A, B)
print("Sum (numpy):", np.sum(numpy_matrix))
print("Frobenius norm (numpy):", np.linalg.norm(numpy_matrix))
