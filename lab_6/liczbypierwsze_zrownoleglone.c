#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N 100000
#define S (int)sqrt(N)
#define M N / 10
#define NUM_OF_THREADS 8

int main(int argc, char **argv)
{
    long int a[S + 1];
    long int pierwsze[M];
    long i, k, liczba, reszta;
    long int lpodz;
    long int llpier = 0;
    FILE *fp;

    omp_set_num_threads(NUM_OF_THREADS);
    printf("Liczba watkow: %d\n", NUM_OF_THREADS);

    double start, end;
    start = omp_get_wtime();

#pragma omp parallel for
    for (i = 2; i <= S; i++)
        a[i] = 1;

    for (i = 2; i <= S; i++)
        if (a[i] == 1)
        {
            pierwsze[llpier++] = i;
            for (k = i + i; k <= S; k += i)
                a[k] = 0;
        }

    lpodz = llpier;

#pragma omp parallel for private(liczba, k, reszta) shared(pierwsze, llpier)
    for (liczba = S + 1; liczba <= N; liczba++)
    {
        for (k = 0; k < lpodz; k++)
        {
            reszta = (liczba % pierwsze[k]);
            if (reszta == 0)
                break;
        }
        if (reszta != 0)
        {
#pragma omp critical
            {
                pierwsze[llpier++] = liczba;
            }
        }
    }

    end = omp_get_wtime();
    printf("Czas wykonania: %f s\n", end - start);

    if ((fp = fopen("primes_2.txt", "w")) == NULL)
    {
        printf("Nie moge otworzyc pliku do zapisu\n");
        exit(1);
    }
    
    for (i = 0; i < llpier; i++)
        fprintf(fp, "%ld ", pierwsze[i]);
    fclose(fp);
    
    return 0;
}
