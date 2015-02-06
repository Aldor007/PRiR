#ifndef CUDA_KERNEL_GAUSS
#define CUDA_KERNEL_GAUSS
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/core.hpp>



__global__ void calcateGaussChannel(uchar3*  input_channels, uchar3 * output_channels, int half_width, int g_width, int g_height, float * g_gauss_filtr) {
    int y  = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if (y >= g_height || x  >=  g_width)
        return;
    int tab_index = y *g_width +x;

    float3 result = make_float3(0, 0, 0);
    for (int i = -half_width ; i<=half_width; ++i)  {
        for (int j = - half_width; j<=half_width; ++j) {
            int img_x = min(max(x + i, 0), g_width - 1);
            int img_y = min(max(y + j, 0), g_height - 1);

            int image_offset  = img_y * g_width + img_x;
            int filter_offset = (i + half_width) * 7 + (j + half_width);
            float filter_val = g_gauss_filtr[filter_offset];
            result.x += input_channels[image_offset].x * filter_val; // blue
            result.y += input_channels[image_offset].y * filter_val; // green
            result.z += input_channels[image_offset].z * filter_val; // red
        }
    }
    output_channels[tab_index] = make_uchar3(result.x, result.y, result.z);
}
// helper functions and utilities to work with CUDA
void cudaWraper (dim3 grid, dim3 block, cv::Mat &frame) {
    // w kodzie z algortym.org
    float  gauss_filterData[] = 
    {1, 1, 2, 2, 2, 1, 1,
    1, 2, 2, 4, 2, 2, 1,
    2, 2, 4, 8, 4, 2, 2,
    2, 4, 8,16, 8, 4, 2,
    2, 2, 4, 8, 4, 2, 2,
    1, 2, 2, 4, 2, 2, 1,
    1, 1, 2, 2, 2, 1, 1};
    int norm = 0;
    for (int i =0; i< 7*7; i++) {
        norm += gauss_filterData[i];
    }

    // dzieki temu nie trzeba dzielic w kernul
    for (int i =0; i< 7*7; i++) {
        gauss_filterData[i] /= (float)norm;
    }


    float * gauss_filter = NULL;

    uchar3 * channelsCuda;
    uchar3 * output;
    cudaMalloc(&gauss_filter, sizeof(float) * 47);
    cudaMemcpy(gauss_filter, gauss_filterData, sizeof(float) * 47, cudaMemcpyHostToDevice);
    size_t tmpSize = sizeof(uchar3) * 1 * frame.rows * frame.cols;
    cudaMalloc(&channelsCuda, tmpSize);
    cudaMalloc(&output, tmpSize);
    cudaMemcpy(channelsCuda, frame.ptr<unsigned char>(0), tmpSize,  cudaMemcpyHostToDevice); //kopiowanie kanalu CPU na CUDA */

    calcateGaussChannel<<<grid, block>>>(channelsCuda, output, 7/2, frame.cols, frame.rows, gauss_filter);

    cudaMemcpy(frame.ptr<unsigned char>(0), output, tmpSize, cudaMemcpyDeviceToHost);
    cudaFree(channelsCuda);
    cudaFree(output);
    cudaFree(gauss_filter);

    
}
#endif
