from multiprocessing.managers import BaseManager
import numpy as np
import sys
import time


class QueueManager(BaseManager):
    pass


def main(ip, port, matrix_file, vector_file, chunk_size=10):
    QueueManager.register("in_queue")
    QueueManager.register("out_queue")
    manager = QueueManager(address=(ip, int(port)), authkey=b"password")
    manager.connect()

    in_queue = manager.in_queue()
    out_queue = manager.out_queue()

    try:
        matrix = np.loadtxt(matrix_file)
    except Exception as e:
        print("Error loading matrix file:", e)
        sys.exit(1)
    try:
        vector = np.loadtxt(vector_file)
    except Exception as e:
        print("Error loading vector file:", e)
        sys.exit(1)

    if vector.ndim != 1 or matrix.shape[1] != vector.shape[0]:
        print("Invalid dimensions.")
        sys.exit(1)

    start_time = time.time()

    num_rows = len(matrix)
    num_chunks = (num_rows + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_rows)
        in_queue.put((i, matrix[start:end], vector))

    results = {}

    for _ in range(num_chunks):
        task_id, partial_result = out_queue.get()
        results[task_id] = partial_result

    result = np.concatenate([results[i] for i in range(num_chunks)])

    end_time = time.time()

    print("Result:", result)
    print("Time:", end_time - start_time)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "Usage: python client.py <ip> <port> <matrix_file> <vector_file> <chunk_size>"
        )
        sys.exit(1)
    chunk_size = int(sys.argv[5]) if len(sys.argv) == 6 else 10
    main(*sys.argv[1:5], chunk_size)
