#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <string.h>
#include <errno.h>
#include <sys/wait.h>
#include <time.h>

#define BUFFER_SIZE 1024

double *vector;
double *results;

void on_usr1(int signal)
{
    printf("Otrzymałem USR1 (PID=%d)\n", getpid());
}

int main(int argc, char **argv)
{
    // ustalenie liczby procesów potomnych
    int children = argc > 1 ? atoi(argv[1]) : 4;
    if (children <= 0)
    {
        fprintf(stderr, "Nieprawidłowa liczba dzieci, musi być wyższa od 0.\n");
        exit(EXIT_FAILURE);
    }

    // odczytanie wektora z pliku i zapisanie do pamięci współdzielonej
    FILE *file = fopen("vector.dat", "r");
    if (!file)
    {
        perror("Blad otwarcia pliku");
        exit(EXIT_FAILURE);
    }

    char buffer[BUFFER_SIZE + 1];
    fgets(buffer, BUFFER_SIZE, file);
    int n = atoi(buffer);

    key_t key1 = ftok("shm1", 65);
    key_t key2 = ftok("shm2", 66);

    // utworzenie pamięci współdzielonej
    int shmid1 = shmget(key1, n * sizeof(double), IPC_CREAT | 0666);
    int shmid2 = shmget(key2, children * sizeof(double), IPC_CREAT | 0666);

    // dołączenie pamięci współdzielonej
    vector = (double *)shmat(shmid1, NULL, 0);
    results = (double *)shmat(shmid2, NULL, 0);

    if (vector == (double *)-1 || results == (double *)-1)
    {
        perror("Blad dolaczania pamieci wspoldzielonej");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++)
    {
        fgets(buffer, BUFFER_SIZE, file);
        vector[i] = atof(buffer);
    }
    fclose(file);

    // ustalenie zakresu indeksów dla poszczególnych procesów potomnych
    int range[children + 1];
    for (int i = 0; i < children; i++)
    {
        range[i] = i * (n / children);
    }
    range[children] = n;

    // tworzenie procesów potomnych
    pid_t child_pids[children];
    for (int i = 0; i < children; i++)
    {
        pid_t pid = fork();

        if (pid == 0)
        {
            struct sigaction usr1;
            sigemptyset(&usr1.sa_mask);
            usr1.sa_flags = 0;
            usr1.sa_handler = on_usr1;
            sigaction(SIGUSR1, &usr1, NULL);

            pause();

            int start_index = range[i];
            int end_index = range[i + 1];

            double partial_sum = 0.0;
            for (int j = start_index; j < end_index; j++)
            {
                partial_sum += vector[j];
            }
            results[i] = partial_sum;

            exit(0);
        }
        else if (pid > 0)
        {
            child_pids[i] = pid;
        }
        else
        {
            perror("Błąd podczas tworzenia procesu potomnego");
            exit(EXIT_FAILURE);
        }
    }
    sleep(1);

    // wysłanie sygnału do każdego z procesów potomnych
    for (int i = 0; i < children; i++)
    {
        kill(child_pids[i], SIGUSR1);
    }

    // oczekiwanie na zakończenie wszystkich procesów potomnych
    while (wait(NULL) > 0)
        ;

    double sum = 0;
    for (int i = 0; i < children; i++)
    {
        printf("Wynik cząstkowy w procesie %d: %f\n", child_pids[i], results[i]);
        sum += results[i];
    }
    printf("\nSuma całkowita wyników cząstkowych: %f\n", sum);

    // zwolnienie pamięci współdzielonej
    shmdt(vector);
    shmdt(results);

    return 0;
}
