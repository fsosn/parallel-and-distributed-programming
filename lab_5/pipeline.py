from mpi4py import MPI
import numpy as np

DATA_SIZE = 1_000_000
CHUNK_SIZE = 1_000

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

cart = comm.Create_cart(dims=(size,), periods=(False,))
left, right = cart.Shift(direction=0, disp=1)

if rank == 0:
    data = np.ones(DATA_SIZE, dtype=int)
    print(f"Process {rank}: Total sum = {np.sum(data)}")

buf = np.empty(CHUNK_SIZE, dtype=int)
local_sum = 0

for i in range(0, DATA_SIZE, CHUNK_SIZE):
    if rank == 0:
        buf = data[i : i + CHUNK_SIZE] if i + CHUNK_SIZE <= len(data) else data[i:]
        cart.Send(buf, dest=right)
        # print(f"Sent data from process {rank} to {right}")
    elif rank == size - 1:
        cart.Recv(buf, source=left)
        local_sum += np.sum(buf)
        # print(f"Chunk {i+1}-{i+CHUNK_SIZE}: sum =", np.sum(buf))
        # print(f"Received data in process {rank} from process {left}")
    else:
        cart.Recv(buf, source=left)
        cart.Send(buf, dest=right)
        # print(f"Received data in process {rank} from {left} and sent to process {right}")

if rank == size - 1:
    print(f"Process {rank}: Total sum = {local_sum}")

cart.Free()
MPI.Finalize()
