#include "atomic.cuh"

int main(){
    atomic a;

    printf("%d total threads in %d blocks writing into %d array elements\n",
           NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);

    int *h_array = new int[ARRAY_SIZE];
    //a.naive_increment(h_array);
    a.atomic_increment(h_array);
}