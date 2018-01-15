#ifndef BARRIERS_CUH
#define BARRIERS_CUH

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void barriers_sample(int *a_h){
    int idx = threadIdx.x;
    const int N = (sizeof(a_h)/sizeof(*a_h));
    __shared__ int array[N];
    array[idx] = threadIdx.x;
    __syncthreads();
    if (idx < N - 1) {
        int temp = array[idx+1];
        __syncthreads();
        array[idx] = temp;
        __syncthreads();
    }
}

class barriers {
public:
    void run_barriers(int *h_arr, int N){
        int *d_arr = new int[N];
        size_t size = N * sizeof(double);
        cudaMalloc((void **) &d_arr, size);
        cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

        int block_size = 3;
        int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
        cout << "block size: "<< block_size << endl;
        cout << "number of blocks: " << n_blocks << endl;

        barriers_sample<<<n_blocks, block_size>>> (d_arr);
        cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
        cudaFree(d_arr);
    }
};
#endif