#include "square.cuh"

int main() {
    square sqr;
    const int N = 10;
    double * h_arr = new double[N];
    for (int i = 0; i < N; i++) {
        h_arr[i] = i;
    }

    sqr.square_on_device(h_arr, N);
    for (int i = 0; i < N; i++) {
        printf("%f\n",h_arr[i]);
    }
}
