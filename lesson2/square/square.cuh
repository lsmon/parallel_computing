//
// Created by lsmon on 11/28/16.
//

#ifndef CUDA_SQUARE_H
#define CUDA_SQUARE_H

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void square_on_gpu(double *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) a[idx] = a[idx] * a[idx];
    printf("blockIdx.x %d, blockIdx.y %d, blockIdx.z %d\nblockDim.x %d, blockDim.y %d, blockDim.z %d\nthreadIdx.x %d, threadIdx.y %d, threadIdx.z %d\nGPU print: idx = %d, a=%f\n"
            , blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, threadIdx.x, threadIdx.y, threadIdx.z, idx, a[idx]);
}

class square {
public:
    void square_on_device(double *h_arr, const int N){
        double *d_arr = new double[N]; // initialize a_d as an array with N double pointer
        size_t size = N * sizeof(double);
        cudaMalloc((void **) &d_arr, size);
        cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

        int block_size = 3;
        int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
        cout << "block size: "<< block_size << endl;
        cout << "number of blocks: " << n_blocks << endl;

        square_on_gpu<<<n_blocks, block_size>>> (d_arr, N);
        cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
        cudaFree(d_arr);
    }
};


#endif //CUDA_SQUARE_H
