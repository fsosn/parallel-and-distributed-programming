from mpi4py import MPI
import numpy as np
import sys
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

ALIVE = 1
DEAD = 0

PYGAME_CELL_SIZE = 1
PYGAME_FPS = 60


def init_grid(rows, cols):
    np.random.seed(123)
    return np.random.randint(2, size=(rows, cols), dtype=np.int32)


def count_neighbours(grid, row, col):
    rows, cols = grid.shape
    count = 0
    if row > 0 and col > 0:
        count += grid[row - 1, col - 1]
    if row > 0:
        count += grid[row - 1, col]
    if row > 0 and col < cols - 1:
        count += grid[row - 1, col + 1]
    if col > 0:
        count += grid[row, col - 1]
    if col < cols - 1:
        count += grid[row, col + 1]
    if row < rows - 1 and col > 0:
        count += grid[row + 1, col - 1]
    if row < rows - 1:
        count += grid[row + 1, col]
    if row < rows - 1 and col < cols - 1:
        count += grid[row + 1, col + 1]

    return count


def update_grid(local_grid, rows, cols):
    new_local_grid = np.copy(local_grid)
    for r in range(1, rows + 1):
        for c in range(cols):
            num_neighbours = count_neighbours(local_grid, r, c)
            if local_grid[r, c] == ALIVE:
                if num_neighbours < 2 or num_neighbours > 3:
                    new_local_grid[r, c] = DEAD
            else:
                if num_neighbours == 3:
                    new_local_grid[r, c] = ALIVE
    return new_local_grid[1:-1, :]


def draw_grid(screen, grid, rows, cols):
    for i in range(rows):
        for j in range(cols):
            color = (255, 255, 255) if grid[i, j] == ALIVE else (0, 0, 0)
            pygame.draw.rect(
                screen,
                color,
                (
                    j * PYGAME_CELL_SIZE,
                    i * PYGAME_CELL_SIZE,
                    PYGAME_CELL_SIZE,
                    PYGAME_CELL_SIZE,
                ),
            )


def exchange_edges(local_grid, rank, size, cols):
    top_row = np.zeros(cols, dtype=np.int32)
    bottom_row = np.zeros(cols, dtype=np.int32)
    comm = MPI.COMM_WORLD

    if rank > 0:
        comm.Sendrecv(local_grid[1, :], dest=rank - 1, recvbuf=top_row, source=rank - 1)
    if rank < size - 1:
        comm.Sendrecv(
            local_grid[-2, :], dest=rank + 1, recvbuf=bottom_row, source=rank + 1
        )
    if rank > 0:
        local_grid[0, :] = top_row
    if rank < size - 1:
        local_grid[-1, :] = bottom_row

    return local_grid


def main(rows, cols, generations):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rows % size != 0:
        if rank == 0:
            print("Number of rows is not divisible by the number of processes.")
        MPI.Finalize()
        return

    rows_per_process = rows // size

    grid = None
    if rank == 0:
        grid = init_grid(rows, cols)

    local_grid = np.zeros((rows_per_process + 2, cols), dtype=np.int32)

    if rank == 0:
        pygame.init()
        screen_size = cols * PYGAME_CELL_SIZE, rows * PYGAME_CELL_SIZE
        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Conway's Game of Life")
        clock = pygame.time.Clock()
        start_time = MPI.Wtime()

    for _ in range(generations):
        comm.Scatter(grid, local_grid[1:-1, :], root=0)
        local_grid = exchange_edges(local_grid, rank, size, cols)
        new_local_grid = update_grid(local_grid, rows_per_process, cols)

        if rank == 0:
            updated_grid = np.zeros((rows, cols), dtype=np.int32)
        else:
            updated_grid = None

        comm.Gather(new_local_grid, updated_grid, root=0)

        if rank == 0:
            grid = updated_grid
            screen.fill((0, 0, 0))
            draw_grid(screen, grid, rows, cols)
            pygame.display.flip()
            clock.tick(PYGAME_FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    if rank == 0:
        end_time = MPI.Wtime()
        time = end_time - start_time
        print("Time:", time, "s")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            clock.tick(PYGAME_FPS)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: mpiexec -n <num_processes> python game_of_life.py <num_rows> <num_cols> <num_generations>"
        )
        sys.exit(1)

    try:
        rows = int(sys.argv[1])
        cols = int(sys.argv[2])
        generations = int(sys.argv[3])
    except ValueError:
        print("All arguments must be integers.")
        sys.exit(1)

    if rows <= 0 or cols <= 0 or generations <= 0:
        print("All arguments must be positive integers.")
        sys.exit(1)

    main(rows, cols, generations)
    MPI.Finalize()