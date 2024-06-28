from mpi4py import MPI
import sys

def local_gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def main(data: list):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_value = data[rank % len(data)]
    
    if size == 1:
        gcd_result = data[0]
        for i in range(1, len(data)):
            gcd_result = local_gcd(gcd_result, data[i])
        print("NWD:", gcd_result)
        return
    
    for i in range(1, size):
        dest = (rank + i) % size
        source = (rank - i) % size
        comm.send(local_value, dest=dest)
        local_value = comm.recv(source=source)
        
        local_value = local_gcd(local_value, data[(rank - i) % len(data)])
        
    if rank != 0:
        comm.send(local_value, dest=0)
    else:
        gcd_values = [local_value]
        for i in range(1, size):
            gcd_values.append(comm.recv(source=i))
        
        result = gcd_values[0]
        for value in gcd_values[1:]:
            result = local_gcd(result, value)
        print("NWD:", result)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: mpiexec -np <num_processes> python integral_group.py <num_1> <num_2> ... <num_n>"
        )
        sys.exit(1)

    data = list(map(int, sys.argv[1].strip('[]').split(',')))

    main(data)