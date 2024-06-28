from multiprocessing.managers import BaseManager
import numpy as np
import multiprocessing
import sys


class QueueManager(BaseManager):
    pass


def multiply(matrix, vector):
    return np.dot(matrix, vector)


def worker_loop(in_queue, out_queue):
    while True:
        task = in_queue.get()
        if task is None:
            break
        i, matrix, vector = task
        result = multiply(matrix, vector)
        out_queue.put((i, result))


def main(ip, port):
    QueueManager.register("in_queue")
    QueueManager.register("out_queue")
    manager = QueueManager(address=(ip, int(port)), authkey=b"password")
    manager.connect()

    in_queue = manager.in_queue()
    out_queue = manager.out_queue()

    processes = []
    for _ in range(multiprocessing.cpu_count()):
        process = multiprocessing.Process(
            target=worker_loop, args=(in_queue, out_queue)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python worker.py <ip> <port>")
        sys.exit(1)
    main(*sys.argv[1:])
