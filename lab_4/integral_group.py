from mpi4py import MPI
import sys
import time


def f(x):
    return x**2


def integrate(a, b, n, f):
    if n == 0:
        return 0.0

    h = (b - a) / n
    integral = (f(a) + f(b)) / 2.0

    for i in range(1, n):
        integral += f(a + i * h)
    integral *= h

    return integral


def main(begin: float, end: float, num_points: int):
    start_time = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    begin = float(begin)
    end = float(end)
    num_points = int(num_points)

    if num_points < size:
        print("Num of points cannot be less than num of proceses.")
        sys.exit(1)
    if num_points < 2:
        print("Num of points cannot be less than 2.")
        sys.exit(1)

    points_per_process = num_points // size
    remainder = num_points % size

    data_to_scatter = []

    for i in range(size):
        local_points = points_per_process + (1 if i < remainder else 0)
        local_a = begin + i * (end - begin) / size
        local_b = local_a + local_points * (end - begin) / num_points
        data_to_scatter.append((local_a, local_b, local_points))

    local_begin, local_end, local_num_points = comm.scatter(data_to_scatter, root=0)

    local_integral = integrate(local_begin, local_end, local_num_points, f)
    local_integrals = comm.gather(local_integral, root=0)

    if rank == 0:
        # print("Local integrals:", local_integrals)
        total_integral = sum(local_integrals)
        print("Result:", total_integral)
        end_time = time.time()
        print("Time:", end_time - start_time)

    MPI.Finalize()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: mpiexec -np <num_processes> python integral_group.py <begin> <end> <num_points>"
        )
        sys.exit(1)

    main(*sys.argv[1:])
