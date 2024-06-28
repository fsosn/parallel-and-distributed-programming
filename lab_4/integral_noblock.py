from mpi4py import MPI
import sys
import numpy as np
import time


def f(x):
    return x ** 2


def trapezoidal_rule(a, b, n, f):
    h = (b - a) / n
    integral = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        integral += f(a + i * h)
    integral *= h
    return integral


def main(begin, end, num_points):
    start_time = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    begin = float(begin)
    end = float(end)
    num_points = int(num_points)

    if num_points < size:
        if rank == 0:
            print("Num of points cannot be less than num of processes.")
        MPI.Finalize()
        sys.exit(1)
    if num_points <= 0:
        if rank == 0:
            print("Num of points cannot be less than 0.")
        MPI.Finalize()
        sys.exit(1)

    points_per_process = num_points // size
    remainder = num_points % size
    local_points = points_per_process + (1 if rank < remainder else 0)

    local_a = begin + rank * (end - begin) / size
    local_b = local_a + local_points * (end - begin) / num_points

    local_integral = trapezoidal_rule(local_a, local_b, local_points, f)
    integrals = comm.gather(local_integral, root=0)

    send_buffer = np.array([local_integral])
    recv_buffer = np.zeros(1)

    send_request = None
    recv_request = None
    send_status = None
    recv_status = None

    if rank == 0:
        total_integral = local_integral
        for source_rank in range(1, size):
            recv_request = comm.Irecv(recv_buffer, source=source_rank)
            recv_request.Wait(recv_status)
            total_integral += recv_buffer[0]
        print("Result:", total_integral)
        end_time = time.time()
        print("Time:", end_time - start_time)
    else:
        send_request = comm.Isend(send_buffer, dest=0)
        send_request.Wait(send_status)

    MPI.Finalize()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: mpiexec -np <num_processes> python integral.py <begin> <end> <num_points>"
        )
        sys.exit(1)

    main(*sys.argv[1:])
