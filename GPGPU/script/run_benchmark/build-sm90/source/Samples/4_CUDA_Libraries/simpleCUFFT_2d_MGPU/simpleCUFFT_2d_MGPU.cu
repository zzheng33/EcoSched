/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

////////////////////////////////////////////////////////////////////////////////
//
//  simpleCUFFT_2d_MGPU.cu
//
//  This sample code demonstrate the use of CUFFT library for 2D data on multiple GPU.
//  Example showing the use of CUFFT for solving 2D-POISSON equation using FFT on multiple GPU.
//  For reference we have used the equation given in http://www.bu.edu/pasi/files/2011/07/
//  Lecture83.pdf
//
////////////////////////////////////////////////////////////////////////////////


// System includes
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUFFT Header file
#include <cufftXt.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// Complex data type
typedef float2 Complex;

// Data configuration
const int GPU_COUNT = 2;
const int BSZ_Y     = 4;
const int BSZ_X     = 4;
constexpr int kDefaultProblemSize = 64;
constexpr int kDefaultMaxIters    = 1;

void usage()
{
    printf("Usage: simpleCUFFT_2d_MGPU [--problem-size=<grid_dim>] [--max-iters=<repeat_count>] [--help]\n");
    printf("  problem-size : square grid dimension [default: %d]\n", kDefaultProblemSize);
    printf("  max-iters    : number of FFT Poisson solves [default: %d]\n", kDefaultMaxIters);
}

bool hasOption(int argc, char **argv, const char *name)
{
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, name)) {
            return true;
        }
        if (arg[0] == '-' && !strcmp(arg + 1, name)) {
            return true;
        }
        if (!strncmp(arg, "--", 2) && !strcmp(arg + 2, name)) {
            return true;
        }
    }

    return false;
}

bool parseIntOption(int argc, char **argv, const char *name, int *value)
{
    const size_t name_len = strlen(name);

    for (int i = 1; i < argc; i++) {
        const char *arg       = argv[i];
        const char *value_str = NULL;

        if ((arg[0] == '-' && !strcmp(arg + 1, name)) || (!strncmp(arg, "--", 2) && !strcmp(arg + 2, name))) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for option %s\n", arg);
                exit(EXIT_FAILURE);
            }
            value_str = argv[i + 1];
        }
        else if (!strncmp(arg, name, name_len) && arg[name_len] == '=') {
            value_str = arg + name_len + 1;
        }
        else if (arg[0] == '-' && !strncmp(arg + 1, name, name_len) && arg[name_len + 1] == '=') {
            value_str = arg + name_len + 2;
        }
        else if (!strncmp(arg, "--", 2) && !strncmp(arg + 2, name, name_len) && arg[name_len + 2] == '=') {
            value_str = arg + name_len + 3;
        }

        if (value_str != NULL) {
            char *end = NULL;
            long  parsed = strtol(value_str, &end, 10);
            if (end == value_str || *end != '\0') {
                fprintf(stderr, "Invalid integer value for option %s: %s\n", name, value_str);
                exit(EXIT_FAILURE);
            }
            *value = static_cast<int>(parsed);
            return true;
        }
    }

    return false;
}

// Forward Declaration
void solvePoissonEquation(cudaLibXtDesc *, cudaLibXtDesc *, float **, int, int);

__global__ void solvePoisson(cufftComplex *, cufftComplex *, float *, int, int, int n_gpu);

///////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("\nPoisson equation using CUFFT library on Multiple GPUs is "
           "starting...\n\n");

    int N         = kDefaultProblemSize;
    int max_iters = kDefaultMaxIters;

    if (hasOption(argc, argv, "h") || hasOption(argc, argv, "help")) {
        usage();
        return EXIT_SUCCESS;
    }

    if (parseIntOption(argc, argv, "problem-size", &N) || parseIntOption(argc, argv, "n", &N)) {
    }

    if (parseIntOption(argc, argv, "max-iters", &max_iters) || parseIntOption(argc, argv, "iterations", &max_iters)) {
    }

    if (N < 1) {
        fprintf(stderr, "problem-size must be >= 1, got %d\n", N);
        return EXIT_FAILURE;
    }

    if (N % GPU_COUNT != 0) {
        fprintf(stderr, "problem-size must be divisible by %d, got %d\n", GPU_COUNT, N);
        return EXIT_FAILURE;
    }

    if (max_iters < 1) {
        fprintf(stderr, "max-iters must be >= 1, got %d\n", max_iters);
        return EXIT_FAILURE;
    }

    printf("Configured problem size N = %d\n", N);
    printf("Configured fixed iterations = %d\n", max_iters);

    int GPU_N;
    checkCudaErrors(cudaGetDeviceCount(&GPU_N));

    if (GPU_N < GPU_COUNT) {
        printf("No. of GPU on node %d\n", GPU_N);
        printf("Two GPUs are required to run simpleCUFFT_2d_MGPU sample code\n");
        exit(EXIT_WAIVED);
    }

    int *major_minor         = (int *)malloc(sizeof(int) * GPU_N * 2);
    int  found2IdenticalGPUs = 0;
    int  nGPUs               = 2;
    int *whichGPUs;
    whichGPUs = (int *)malloc(sizeof(int) * nGPUs);

    for (int i = 0; i < GPU_N; i++) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));
        major_minor[i * 2]     = deviceProp.major;
        major_minor[i * 2 + 1] = deviceProp.minor;
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n",
               i,
               deviceProp.name,
               deviceProp.major,
               deviceProp.minor);
    }

    for (int i = 0; i < GPU_N; i++) {
        for (int j = i + 1; j < GPU_N; j++) {
            if ((major_minor[i * 2] == major_minor[j * 2]) && (major_minor[i * 2 + 1] == major_minor[j * 2 + 1])) {
                whichGPUs[0]        = i;
                whichGPUs[1]        = j;
                found2IdenticalGPUs = 1;
                break;
            }
        }
        if (found2IdenticalGPUs) {
            break;
        }
    }

    free(major_minor);
    if (!found2IdenticalGPUs) {
        printf("No Two GPUs with same architecture found\nWaiving simpleCUFFT_2d_MGPU "
               "sample\n");
        exit(EXIT_WAIVED);
    }

    float  xMAX = 1.0f, xMIN = 0.0f, yMIN = 0.0f, h = (xMAX - xMIN) / ((float)N), s = 0.1f, s2 = s * s;
    float *x, *y, *f, *u_a, r2;

    x   = (float *)malloc(sizeof(float) * N * N);
    y   = (float *)malloc(sizeof(float) * N * N);
    f   = (float *)malloc(sizeof(float) * N * N);
    u_a = (float *)malloc(sizeof(float) * N * N);

    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            x[N * j + i] = xMIN + i * h;
            y[N * j + i] = yMIN + j * h;
            r2 = (x[N * j + i] - 0.5f) * (x[N * j + i] - 0.5f) + (y[N * j + i] - 0.5f) * (y[N * j + i] - 0.5f);
            f[N * j + i]   = (r2 - 2 * s2) / (s2 * s2) * exp(-r2 / (2 * s2));
            u_a[N * j + i] = exp(-r2 / (2 * s2)); // analytical solution
        }

    float *k, *d_k[GPU_COUNT];
    k = (float *)malloc(sizeof(float) * N);
    for (int i = 0; i <= N / 2; i++) {
        k[i] = i * 2 * (float)M_PI;
    }
    for (int i = N / 2 + 1; i < N; i++) {
        k[i] = (i - N) * 2 * (float)M_PI;
    }

    // Create a complex variable on host
    Complex *h_f = (Complex *)malloc(sizeof(Complex) * N * N);

    // Initialize the memory for the signal
    for (int i = 0; i < (N * N); i++) {
        h_f[i].x = f[i];
        h_f[i].y = 0.0f;
    }

    // cufftCreate() - Create an empty plan
    cufftResult result;
    cufftHandle planComplex;
    result = cufftCreate(&planComplex);
    if (result != CUFFT_SUCCESS) {
        printf("cufftCreate failed\n");
        exit(EXIT_FAILURE);
    }

    // cufftXtSetGPUs() - Define which GPUs to use
    result = cufftXtSetGPUs(planComplex, nGPUs, whichGPUs);

    if (result == CUFFT_INVALID_DEVICE) {
        printf("This sample requires two GPUs on the same board.\n");
        printf("No such board was found. Waiving sample.\n");
        exit(EXIT_WAIVED);
    }
    else if (result != CUFFT_SUCCESS) {
        printf("cufftXtSetGPUs failed\n");
        exit(EXIT_FAILURE);
    }

    // Print the device information to run the code
    printf("\nRunning on GPUs\n");
    for (int i = 0; i < 2; i++) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, whichGPUs[i]));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n",
               whichGPUs[i],
               deviceProp.name,
               deviceProp.major,
               deviceProp.minor);
    }

    size_t *worksize;
    worksize = (size_t *)malloc(sizeof(size_t) * nGPUs);

    // cufftMakePlan2d() - Create the plan
    result = cufftMakePlan2d(planComplex, N, N, CUFFT_C2C, worksize);
    if (result != CUFFT_SUCCESS) {
        printf("*MakePlan* failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nGPUs; i++) {
        cudaSetDevice(whichGPUs[i]);
        cudaMalloc((void **)&d_k[i], sizeof(float) * N);
        cudaMemcpy(d_k[i], k, sizeof(float) * N, cudaMemcpyHostToDevice);
    }

    // Create a variable on device
    // d_f - variable on device to store the input data
    // d_d_f - variable that store the natural order of d_f data
    // d_out - device output
    cudaLibXtDesc *d_f, *d_d_f, *d_out;

    // cufftXtMalloc() - Malloc data on multiple GPUs

    result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_f, CUFFT_XT_FORMAT_INPLACE);
    if (result != CUFFT_SUCCESS) {
        printf("*XtMalloc failed\n");
        exit(EXIT_FAILURE);
    }

    result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_d_f, CUFFT_XT_FORMAT_INPLACE);
    if (result != CUFFT_SUCCESS) {
        printf("*XtMalloc failed\n");
        exit(EXIT_FAILURE);
    }

    result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_out, CUFFT_XT_FORMAT_INPLACE);
    if (result != CUFFT_SUCCESS) {
        printf("*XtMalloc failed\n");
        exit(EXIT_FAILURE);
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        // cufftXtMemcpy() - Copy the data from host to device
        result = cufftXtMemcpy(planComplex, d_f, h_f, CUFFT_COPY_HOST_TO_DEVICE);
        if (result != CUFFT_SUCCESS) {
            printf("*XtMemcpy failed\n");
            exit(EXIT_FAILURE);
        }

        // cufftXtExecDescriptorC2C() - Execute FFT on data on multiple GPUs
        result = cufftXtExecDescriptorC2C(planComplex, d_f, d_f, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS) {
            printf("*XtExecC2C  failed\n");
            exit(EXIT_FAILURE);
        }

        // cufftXtMemcpy() - Copy the data to natural order on GPUs
        result = cufftXtMemcpy(planComplex, d_d_f, d_f, CUFFT_COPY_DEVICE_TO_DEVICE);
        if (result != CUFFT_SUCCESS) {
            printf("*XtMemcpy failed\n");
            exit(EXIT_FAILURE);
        }

        solvePoissonEquation(d_d_f, d_out, d_k, N, nGPUs);

        result = cufftXtExecDescriptorC2C(planComplex, d_out, d_out, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS) {
            printf("*XtExecC2C  failed\n");
            exit(EXIT_FAILURE);
        }
    }

    printf("Completed fixed iterations = %d\n", max_iters);

    // Create a variable on host to copy the data from device
    // h_d_out - variable store the output of device
    Complex *h_d_out = (Complex *)malloc(sizeof(Complex) * N * N);

    // cufftXtMemcpy() - Copy data from multiple GPUs to host
    result = cufftXtMemcpy(planComplex, h_d_out, d_out, CUFFT_COPY_DEVICE_TO_HOST);
    if (result != CUFFT_SUCCESS) {
        printf("*XtMemcpy failed\n");
        exit(EXIT_FAILURE);
    }

    float *out      = (float *)malloc(sizeof(float) * N * N);
    float  constant = h_d_out[0].x / N * N;
    for (int i = 0; i < N * N; i++) {
        // subtract u[0] to force the arbitrary constant to be 0
        out[i] = (h_d_out[i].x / (N * N)) - constant;
    }

    // cleanup memory

    free(h_f);
    free(k);
    free(out);
    free(h_d_out);
    free(x);
    free(whichGPUs);
    free(y);
    free(f);
    free(u_a);
    free(worksize);

    // cudaXtFree() - Free GPU memory
    for (int i = 0; i < GPU_COUNT; i++) {
        cudaFree(d_k[i]);
    }
    result = cufftXtFree(d_out);
    if (result != CUFFT_SUCCESS) {
        printf("*XtFree failed\n");
        exit(EXIT_FAILURE);
    }
    result = cufftXtFree(d_f);
    if (result != CUFFT_SUCCESS) {
        printf("*XtFree failed\n");
        exit(EXIT_FAILURE);
    }
    result = cufftXtFree(d_d_f);
    if (result != CUFFT_SUCCESS) {
        printf("*XtFree failed\n");
        exit(EXIT_FAILURE);
    }

    // cufftDestroy() - Destroy FFT plan
    result = cufftDestroy(planComplex);
    if (result != CUFFT_SUCCESS) {
        printf("cufftDestroy failed: code %d\n", (int)result);
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////////
// Launch kernel on  multiple GPU
///////////////////////////////////////////////////////////////////////////////////
void solvePoissonEquation(cudaLibXtDesc *d_ft, cudaLibXtDesc *d_ft_k, float **k, int N, int nGPUs)
{
    int  device;
    int  rowsPerGpu = N / nGPUs;
    dim3 dimGrid((N + BSZ_X - 1) / BSZ_X, (rowsPerGpu + BSZ_Y - 1) / BSZ_Y);
    dim3 dimBlock(BSZ_X, BSZ_Y);

    for (int i = 0; i < nGPUs; i++) {
        device = d_ft_k->descriptor->GPUs[i];
        cudaSetDevice(device);
        solvePoisson<<<dimGrid, dimBlock>>>(
            (cufftComplex *)d_ft->descriptor->data[i], (cufftComplex *)d_ft_k->descriptor->data[i], k[i], N, i, nGPUs);
    }

    // Wait for device to finish all operation
    for (int i = 0; i < nGPUs; i++) {
        device = d_ft_k->descriptor->GPUs[i];
        cudaSetDevice(device);
        cudaDeviceSynchronize();

        // Check if kernel execution generated and error
        getLastCudaError("Kernel execution failed [ solvePoisson ]");
    }
}

////////////////////////////////////////////////////////////////////////////////
// Kernel for Solving Poisson equation on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void solvePoisson(cufftComplex *ft, cufftComplex *ft_k, float *k, int N, int gpu_id, int n_gpu)
{
    int i     = threadIdx.x + blockIdx.x * blockDim.x;
    int j     = threadIdx.y + blockIdx.y * blockDim.y;
    int index = j * N + i;
    if (i < N && j < N / n_gpu) {
        float k2 = k[i] * k[i] + k[j + gpu_id * N / n_gpu] * k[j + gpu_id * N / n_gpu];
        if (i == 0 && j == 0 && gpu_id == 0) {
            k2 = 1.0f;
        }

        ft_k[index].x = -ft[index].x * 1 / k2;
        ft_k[index].y = -ft[index].y * 1 / k2;
    }
}
