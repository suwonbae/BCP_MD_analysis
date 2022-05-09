#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>

#include "mpi.h"

// forward decelration
// memory allocation helper functions
void *smalloc(int nbytes);
void *srelloc(void *ptr, int nbyte);
double **create(double **&array, int n1, int n2);
double **grow(double **&array, int n1, int n2);

int main(int argc, char *argv[]) {
    
    // variables for MPI
    int numprocs, myid, tag;
    int start_row, end_row, average_rows_per_processor, num_rows;
    tag = 1;
    MPI_Status msegStatus;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm world = MPI_COMM_WORLD;

    // variables for computation
    double **x = nullptr; // xyz array
    double **q = nullptr; // q array
    double res, cc, ss;
    double Lx, Ly, Lz;
    int n;
    int line = 0;
    FILE *fxyz, *fparams, *fout;

    if (myid == 0) {
	
	// 1st dim of x increases from 0 to 1
	x = grow(x, line + 1, 3);
	
	// read in the xyz file
	fxyz = fopen("input.xyz", "r");
	while (fscanf(fxyz, "%lf %lf %lf\n", &x[line][0], &x[line][1], &x[line][2]) == 3) {
	    line++;
	    x = grow(x, line + 1, 3);
	}
	fclose(fxyz);
	
	// read in the parameters
	fparams = fopen("inc.params", "r");
	fscanf(fparams, "%lf %lf %lf\n", &Lx, &Ly, &Lz);
	fscanf(fparams, "%d\n", &n);
	fclose(fparams);
    }
    MPI_Bcast(&line, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (myid > 0 ) {
	x = grow(x, line + 1, 3);
    }
    MPI_Bcast(&x[0][0], line*3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);

    q = grow(q, n*n*n, 4);
    average_rows_per_processor = n*n*n/numprocs;
    start_row = myid * average_rows_per_processor;
    end_row = start_row + average_rows_per_processor;
    if (myid == numprocs-1) end_row = n*n*n;
    num_rows = end_row - start_row;

    for (int i = start_row; i < end_row; i++) {
	// vector to each point in the reciprocal space
        q[i][0] = i/(n*n) * 2 * M_PI / Lx;
        q[i][1] = (i%(n*n))/n * 2 * M_PI / Ly;
        q[i][2] = (i%(n*n))%n * 2 * M_PI / Lz;
	
	cc = ss = 0;
        for (int j = 0; j < line; j++) {
	    res = x[j][0]*q[i][0] + x[j][1]*q[i][1] + x[j][2]*q[i][2];
	    cc += cos(res);
	    ss += sin(res);
        }

	q[i][3] = (pow(cc, 2) + pow(ss, 2))/line;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (myid == 0) {
        
        for (int i = 1; i < numprocs; i++) {
            start_row = i*average_rows_per_processor;
            end_row = start_row + average_rows_per_processor;
            if (i == numprocs-1) end_row = n*n*n;

            num_rows = end_row - start_row;

            MPI_Recv(&q[start_row][0], num_rows*4, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &msegStatus);
        }
    } else {
        MPI_Send(&q[start_row][0], num_rows*4, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    }
    
    if (myid == 0) {
	fout = fopen("sq.txt", "w");
	for (int i = 0; i < n*n*n; i++) {
	    fprintf(fout, "%lf %lf %lf %lf\n", q[i][0], q[i][1], q[i][2], q[i][3]);
	}
    }
    MPI_Finalize();

    free(&(x[0][0]));
    free(x);
    free(&(q[0][0]));
    free(q);
}

void *smalloc(int nbytes){
    void *ptr = malloc(nbytes);
    
    return ptr;
}

void *srealloc(void *ptr, int nbytes) {
    ptr = realloc(ptr, nbytes);
    
    return ptr;
}

double **create(double **&array, int n1, int n2) {
    
    int nbytes = sizeof(double) * n1 * n2;
    double *data = (double *) smalloc(nbytes);
    nbytes = sizeof(double *) * n1;
    array = (double **) smalloc(nbytes);
    
    int n = 0;
    for (int i = 0; i < n1; i++) {
        array[i] = &data[n];
        n += n2;
    }
        
    return array;
}

double **grow(double **&array, int n1, int n2) {
    
    if (array == nullptr) return create(array, n1, n2);
    
    int nbytes = sizeof(double) * n1 * n2;
    double *data = (double *) srealloc(array[0], nbytes);
    nbytes = sizeof(double *) * n1;
    array = (double **) srealloc(array, nbytes);
    
    int n = 0;
    for (int i = 0; i < n1; i++) {
        array[i] = &data[n];
        n += n2;
    }
    
    return array;
}
