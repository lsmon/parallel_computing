//
// Created by Salomon Lee on 11/29/16.
//

#ifndef PARALLEL_COMPUTING_ATOMIC_H
#define PARALLEL_COMPUTING_ATOMIC_H

#include <stdio.h>
#include "gpu_timer.h"

#define NUM_THREADS 10000000
#define ARRAY_SIZE  100

#define BLOCK_WIDTH 1000

void print_array(int *array, int size)
{
    printf("{ ");
    for (int i = 0; i < size; i++)  { printf("%d ", array[i]); }
    printf("}\n");
}

__global__ void increment_naive(int *g)
{
    // which thread is this?
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread to increment consecutive elements, wrapping at ARRAY_SIZE
    i = i % ARRAY_SIZE;
    g[i] = g[i] + 1;
}

__global__ void increment_atomic(int *g)
{
    // which thread is this?
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread to increment consecutive elements, wrapping at ARRAY_SIZE
    i = i % ARRAY_SIZE;
    atomicAdd(& g[i], 1);
}

class atomic {
public:
    void naive_increment(int *h_array){
        GpuTimer timer;
        const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

        // declare, allocate, and zero out GPU memory
        int * d_array;
        cudaMalloc((void **) &d_array, ARRAY_BYTES);
        cudaMemset((void *) d_array, 0, ARRAY_BYTES);

        // launch the kernel - comment out one of these
        timer.Start();
        // increment_naive<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
        increment_naive<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
        timer.Stop();

        // copy back the array of sums from GPU and print
        cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
        print_array(h_array, ARRAY_SIZE);
        printf("Time elapsed = %g ms\n", timer.Elapsed());

        // free GPU memory allocation and exit
        cudaFree(d_array);
    }

    void atomic_increment(int *h_array){
        GpuTimer timer;
        const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

        // declare, allocate, and zero out GPU memory
        int * d_array;
        cudaMalloc((void **) &d_array, ARRAY_BYTES);
        cudaMemset((void *) d_array, 0, ARRAY_BYTES);

        // launch the kernel - comment out one of these
        timer.Start();
        // increment_naive<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
        increment_atomic<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
        timer.Stop();

        // copy back the array of sums from GPU and print
        cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
        print_array(h_array, ARRAY_SIZE);
        printf("Time elapsed = %g ms\n", timer.Elapsed());

        // free GPU memory allocation and exit
        cudaFree(d_array);
    }
};

#endif //PARALLEL_COMPUTING_ATOMIC_H
