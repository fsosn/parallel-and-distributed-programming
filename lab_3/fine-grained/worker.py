from multiprocessing.managers import BaseManager
import numpy as np
import sys


class QueueManager(BaseManager):
    pass


def multiply(matrix_row, vector):
    return np.dot(matrix_row, vector)


def worker_loop(in_queue, out_queue):
    while True:
        task = in_queue.get()
        if task is None:
            break
        i, matrix_row, vector = task
        result = multiply(matrix_row, vector)
        out_queue.put((i, np.array([result])))


def main(ip, port):
    QueueManager.register("in_queue")
    QueueManager.register("out_queue")
    manager = QueueManager(address=(ip, int(port)), authkey=b"password")
    manager.connect()

    in_queue = manager.in_queue()
    out_queue = manager.out_queue()

    worker_loop(in_queue, out_queue)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python worker.py <ip> <port>")
        sys.exit(1)
    main(*sys.argv[1:])
