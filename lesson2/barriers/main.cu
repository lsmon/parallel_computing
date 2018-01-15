#include "barrier.cuh"

int main() {
    barriers b;
    const int N = 128;
    int * h_arr = new int[N];
    for (int i = 0; i < N; i++) {
        h_arr[i] = i;
    }

    b.run_barriers(h_arr,N);
    for (int i = 0; i < N; i++) {
        printf("%d\n",h_arr[i]);
    }
    return 0;
}
