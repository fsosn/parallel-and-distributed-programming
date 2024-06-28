#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

struct thread_data {
    int thread_id;
    int start_row;
    int end_row;
    int a_cols;
    int b_cols;
    double **A;
    double **B;
    double **C;
    double *partial_sums;
};

void *multiply(void *threadarg) {
    struct thread_data *my_data;
    my_data = (struct thread_data *) threadarg;

    int i, j, k;
    double sum;
    
    for (i = my_data->start_row; i < my_data->end_row; i++) {
        for (j = 0; j < my_data->b_cols; j++) {
            sum = 0;
            for (k = 0; k < my_data->a_cols; k++) {
                sum += my_data->A[i][k] * my_data->B[k][j];
            }
            my_data->C[i][j] = sum;
            my_data->partial_sums[my_data->thread_id] += sum;
        }
    }
    
    pthread_exit(NULL);
}

double frobenius_norm(double **C, int m, int n) {
    int i, j;
    double norm = 0.0;
    for(i = 0; i < m; i++) {
        for(j = 0; j < n; j++) {
            norm += C[i][j] * C[i][j];
        }
    }
    return sqrt(norm);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Wrong amount of arguments\n");
        return EXIT_FAILURE;
    }

    FILE *fpa, *fpb;
    double **A, **B, **C;
    int ma, mb, na, nb;
    int i, j, rc;
    double x;

    fpa = fopen(argv[1], "r");
    fpb = fopen(argv[2], "r");
    if( fpa == NULL || fpb == NULL ) {
        perror("Error while opening file");
        exit(EXIT_FAILURE);
    }

    fscanf(fpa, "%d %d", &ma, &na);
    fscanf(fpb, "%d %d", &mb, &nb);


    if(na != mb) {
        printf("Wrong matrix size\n");
        return EXIT_FAILURE;
    }
    
    // Alokacja pamięci
    A = malloc(ma * sizeof(double *));
    for(i = 0; i < ma; i++) {
        A[i] = malloc(na * sizeof(double));
    }

    B = malloc(mb * sizeof(double *));
    for(i = 0; i < mb; i++) {
        B[i] = malloc(nb * sizeof(double));
    }

    C = malloc(ma * sizeof(double *));
    for(i = 0; i < ma; i++) {
        C[i] = malloc(nb * sizeof(double));
    }

    // Wczytanie macierzy
    
    for(i = 0; i < ma; i++) {
        for(j = 0; j < na; j++) {
            fscanf(fpa, "%lf", &x);
            A[i][j] = x;
        }
    }

    for(i = 0; i < mb; i++) {
        for(j = 0; j < nb; j++) {
            fscanf(fpb, "%lf", &x);
            B[i][j] = x;
        }
    }

    // Wątki
    int num_threads = atoi(argv[3]);
    pthread_t threads[num_threads];
    struct thread_data thread_data_array[num_threads];
    double partial_sums[num_threads];

    for (i = 0; i < num_threads; i++) {
        partial_sums[i] = 0;
        thread_data_array[i].thread_id = i;
        thread_data_array[i].start_row = i * (ma / num_threads);
        thread_data_array[i].end_row = (i == (num_threads - 1)) ? ma : (i + 1) * (ma / num_threads);
        thread_data_array[i].a_cols = na;
        thread_data_array[i].b_cols = nb;
        thread_data_array[i].A = A;
        thread_data_array[i].B = B;
        thread_data_array[i].C = C;
        thread_data_array[i].partial_sums = partial_sums;
        
        rc = pthread_create(&threads[i], NULL, multiply, (void *)&thread_data_array[i]);
    }

    // Oczekiwanie na zakończenie wątków
    for (i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Sumowanie i norma Frobeniusa
    double total_sum = 0;
    for (i = 0; i < num_threads; i++) {
        total_sum += partial_sums[i];
    }
    double frobenius = frobenius_norm(C, ma, nb);

    printf("[\n");
    for(i = 0; i < ma; i++) {
        for(j = 0; j < nb; j++) {
            printf("%f ", C[i][j]);
        }
        printf("\n");
    }
    printf("]\n");
    printf("Sum: %f\n", total_sum);
    printf("Frobenius Norm: %f\n", frobenius);

    // Zwolnienie pamięci
    for(i = 0; i < ma; i++) {
        free(A[i]);
    }
    free(A);

    for(i = 0; i < mb; i++) {
        free(B[i]);
    }
    free(B);

    for(i = 0; i < ma; i++) {
        free(C[i]);
    }
    free(C);

    fclose(fpa);
    fclose(fpb);

    return 0;
}