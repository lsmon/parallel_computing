#include "utils.h"
#include "timer.h"
#include "custom_logger.h"
#include <stdio.h>
#include <iostream>

__global__ void gaussian_blur(const unsigned char* const inputChannel,
                              unsigned char* const outputChannel, int numRows,
                              int numCols, const float* const filter,
                              const int filterWidth) {
    // Compute the thread's row and column
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= numCols || row >= numRows) {
        return;
    }

    // The following loop walks over all coefficients in the filter
    float result = 0.0f;
    for (int row_delta = -filterWidth / 2; row_delta <= filterWidth / 2; row_delta++) {
        for (int col_delta = -filterWidth / 2; col_delta <= filterWidth / 2; col_delta++) {
            // Compute the coordinates of the value this coefficient applies to.
            // Apply clamping to image boundaries.
            int value_row = min(max(row + row_delta, 0), numRows - 1);
            int value_col = min(max(col + col_delta, 0), numCols - 1);

            // Compute the partial sum this value adds to the result when scaled by
            // the appropriate coefficient.
            float channel_value = static_cast<float>(inputChannel[value_row * numCols + value_col]);
            float filter_coefficient = filter[(row_delta + filterWidth / 2) * filterWidth + (col_delta + filterWidth / 2)];
            result += channel_value * filter_coefficient;
        }
    }

    outputChannel[row * numCols + col] = result;
}

// This kernel takes in an image represented as a uchar4 and splits
// it into three images consisting of only one color channel each
__global__ void separateChannels(const uchar4* const inputImageRGBA,
                                 int numRows, int numCols,
                                 unsigned char* const redChannel,
                                 unsigned char* const greenChannel,
                                 unsigned char* const blueChannel) {
    // Compute the thread's row and column
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= numCols || row >= numRows) {
        return;
    }

    // Offset into the 1D images
    int offset = row * numCols + col;

    uchar4 rgba_pixel = inputImageRGBA[offset];
    redChannel[offset] = rgba_pixel.x;
    greenChannel[offset] = rgba_pixel.y;
    blueChannel[offset] = rgba_pixel.z;

    printf("at offset %d: %u %u %u\n", offset, redChannel[offset], greenChannel[offset], blueChannel[offset]);
    printf("at offset %d: %u\n", offset, redChannel[offset]);
}

// This kernel takes in three color channels and recombines them
// into one image.  The alpha channel is set to 255 to represent
// that this image has no transparency.
__global__ void recombineChannels(const unsigned char* const redChannel,
                                  const unsigned char* const greenChannel,
                                  const unsigned char* const blueChannel,
                                  uchar4* const outputImageRGBA, int numRows,
                                  int numCols) {
    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                         blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    // make sure we don't try and access memory outside the image
    // by having any threads mapped there return early
    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

    unsigned char red = redChannel[thread_1D_pos];
    unsigned char green = greenChannel[thread_1D_pos];
    unsigned char blue = blueChannel[thread_1D_pos];

    // Alpha should be 255 for no transparency
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);

    outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char* d_red, *d_green, *d_blue;
float* d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage,
                                const size_t numColsImage,
                                const float* const h_filter,
                                const size_t filterWidth) {
    // original
    checkCudaErrors(cudaMalloc(&d_red, sizeof(*d_red) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_green, sizeof(*d_green) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_blue, sizeof(*d_blue) * numRowsImage * numColsImage));

    size_t filtersize = sizeof(float) * filterWidth * filterWidth;
    checkCudaErrors(cudaMalloc(&d_filter, filtersize));

    checkCudaErrors(cudaMemcpy(d_filter, h_filter, filtersize, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4* const h_inputImageRGBA,
                        uchar4* const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows,
                        const size_t numCols, unsigned char* d_redBlurred,
                        unsigned char* d_greenBlurred,
                        unsigned char* d_blueBlurred, const int filterWidth) {
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(1 + (numCols / blockSize.x), 1 + (numRows / blockSize.y), 1);

    // This is for debugging
    size_t numPixels = numCols * numRows;
    std::cerr << "Image dimensions: " << numRows << "x" << numCols << " (" << numPixels << " pixels)\n";
    std::cerr << "blockSize = " << stringify(blockSize) << "\n";
    std::cerr << "gridSize = " << stringify(gridSize) << "\n";

    GpuTimer timer;
    timer.Start();

    separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

    // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
    // launching your kernel to make sure that you didn't make any mistakes.
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    timer.Stop();
    std::cerr << "separateChannels elapsed: " << timer.Elapsed() << " ms\n";

    timer.Start();
    // Blur each channel separately
    gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
    gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
    gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    timer.Stop();
    std::cerr << "gaussian_blur (x3) elapsed: " << timer.Elapsed() << " ms\n";

    timer.Start();

    recombineChannels<<<gridSize, blockSize>>> (d_redBlurred, d_greenBlurred, d_blueBlurred, d_outputImageRGBA, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    timer.Stop();
    std::cerr << "recombineChannels elapsed: " << timer.Elapsed() << " ms\n";
}

// Free all the memory that we allocated
// TODO: make sure you free any arrays that you allocated
void cleanup() {
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));
}

void gaussian_blur_shared(const uchar4* const h_inputImageRGBA,
                          uchar4* const d_inputImageRGBA,
                          uchar4* const d_outputImageRGBA, const size_t numRows,
                          const size_t numCols, unsigned char* d_redBlurred,
                          unsigned char* d_greenBlurred,
                          unsigned char* d_blueBlurred, const int filterWidth) {
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(1 + (numCols / blockSize.x), 1 + (numRows / blockSize.y), 1);

    // This is for debugging
    size_t numPixels = numCols * numRows;
    std::cerr << "Image dimensions: " << numRows << "x" << numCols << " (" << numPixels << " pixels)\n";
    std::cerr << "blockSize = " << stringify(blockSize) << "\n";
    std::cerr << "gridSize = " << stringify(gridSize) << "\n";

    GpuTimer timer;
    timer.Start();

    separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

    // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
    // launching your kernel to make sure that you didn't make any mistakes.
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    timer.Stop();
    std::cerr << "separateChannels elapsed: " << timer.Elapsed() << " ms\n";

    timer.Start();
    int shared_size = filterWidth * filterWidth * sizeof(*d_filter);
    // Blur each channel separately
    gaussian_blur<<<gridSize, blockSize, shared_size>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
    gaussian_blur<<<gridSize, blockSize, shared_size>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
    gaussian_blur<<<gridSize, blockSize, shared_size>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    timer.Stop();
    std::cerr << "gaussian_blur (x3) elapsed: " << timer.Elapsed() << " ms\n";

    timer.Start();

    recombineChannels<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred, d_outputImageRGBA, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    timer.Stop();
    std::cerr << "recombineChannels elapsed: " << timer.Elapsed() << " ms\n";
}