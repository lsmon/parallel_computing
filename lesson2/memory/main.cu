// Using different memory spaces in CUDA
#include "kernels.cuh"

int main(int argc, char **argv)
{
    memory mem;
    /*
     * First, call a kernel that shows using local memory
     */
    mem.local_memory();
    /*
     * Next, call a kernel that shows using global memory
     */
    mem.global_memory();

    /*
     * Next, call a kernel that shows using shared memory
     */
    mem.shared_memory();
    return 0;
}