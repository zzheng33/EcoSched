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
const int MAX_GPU_COUNT      = 4;
const int BSZ_Y              = 4;
const int BSZ_X              = 4;
const int kDefaultProblemSize = 64;
const int kDefaultMaxIters    = 100;

// Forward Declaration
bool parseIntOption(int argc, char **argv, const char *name, int *value);
void usage(const char *programName);
void solvePoissonEquation(cudaLibXtDesc *, cudaLibXtDesc *, float **, int, int);

__global__ void solvePoisson(cufftComplex *, cufftComplex *, float *, int, int, int n_gpu);

bool parseIntOption(int argc, char **argv, const char *name, int *value)
{
    const size_t name_len = strlen(name);

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        const char *value_str = NULL;

        if (!strncmp(arg, "--", 2) && !strcmp(arg + 2, name)) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for option %s\n", arg);
                exit(EXIT_FAILURE);
            }
            value_str = argv[i + 1];
        }
        else if (!strncmp(arg, "--", 2) && !strncmp(arg + 2, name, name_len) && arg[name_len + 2] == '=') {
            value_str = arg + name_len + 3;
        }
        else if (arg[0] == '-' && !strcmp(arg + 1, name)) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for option %s\n", arg);
                exit(EXIT_FAILURE);
            }
            value_str = argv[i + 1];
        }
        else if (arg[0] == '-' && !strncmp(arg + 1, name, name_len) && arg[name_len + 1] == '=') {
            value_str = arg + name_len + 2;
        }
        else if (!strncmp(arg, name, name_len) && arg[name_len] == '=') {
            value_str = arg + name_len + 1;
        }

        if (value_str != NULL) {
            char *end = NULL;
            long parsed = strtol(value_str, &end, 10);
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

void usage(const char *programName)
{
    printf("Usage: %s [--problem-size=<N>] [--max-iters=<count>] [--help]\n", programName);
    printf("  --problem-size=<N>   Square grid dimension [default: %d]\n", kDefaultProblemSize);
    printf("  --max-iters=<count>  Repeat the FFT/solve/inverse sequence this many times [default: %d]\n",
           kDefaultMaxIters);
}

///////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("\nPoisson equation using CUFFT library on Multiple GPUs is "
           "starting...\n\n");

    constexpr int kMinGpusRequired = 1;

    int GPU_N;
    checkCudaErrors(cudaGetDeviceCount(&GPU_N));

    if (GPU_N < kMinGpusRequired) {
        printf("No. of GPU on node %d\n", GPU_N);
        printf("At least one GPU is required to run simpleCUFFT_2d_MGPU sample code\n");
        exit(EXIT_WAIVED);
    }

    int maxGpusToTry = (GPU_N < MAX_GPU_COUNT) ? GPU_N : MAX_GPU_COUNT;
    int visibleGpuIds[MAX_GPU_COUNT] = {0, 1, 2, 3};

    for (int i = 0; i < GPU_N; i++) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));
        if (i < MAX_GPU_COUNT) {
            visibleGpuIds[i] = i;
        }
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n",
               i,
               deviceProp.name,
               deviceProp.major,
               deviceProp.minor);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
        usage(argv[0]);
        exit(EXIT_SUCCESS);
    }

    int N = kDefaultProblemSize;
    int maxIters = kDefaultMaxIters;
    parseIntOption(argc, argv, "problem-size", &N);
    parseIntOption(argc, argv, "n", &N);
    parseIntOption(argc, argv, "max-iters", &maxIters);
    if (N < 1) {
        fprintf(stderr, "problem-size must be a positive integer\n");
        exit(EXIT_FAILURE);
    }
    if (maxIters < 1) {
        fprintf(stderr, "max-iters must be a positive integer\n");
        exit(EXIT_FAILURE);
    }

    printf("Configured problem size N = %d\n", N);
    printf("Configured fixed iterations = %d\n", maxIters);
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

    float *k;
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

    // Try the largest visible GPU count first with cuFFT XT. If none of the
    // multi-GPU configurations work, fall back to a plain single-GPU cuFFT path.
    cufftResult result = CUFFT_SUCCESS;
    cufftHandle planComplex = 0;
    size_t *worksize = NULL;
    int *whichGPUs = NULL;
    float **d_k = NULL;
    cudaLibXtDesc *d_f = NULL, *d_d_f = NULL, *d_out = NULL;
    cufftComplex *d_f_single = NULL, *d_d_f_single = NULL, *d_out_single = NULL;
    int nGPUs = 0;
    bool useXtPath = false;

    auto cleanupXtAttempt = [&](int gpuCount) {
        if (d_out != NULL) {
            cufftXtFree(d_out);
            d_out = NULL;
        }
        if (d_f != NULL) {
            cufftXtFree(d_f);
            d_f = NULL;
        }
        if (d_d_f != NULL) {
            cufftXtFree(d_d_f);
            d_d_f = NULL;
        }
        if (d_k != NULL) {
            for (int i = 0; i < gpuCount; i++) {
                if (d_k[i] != NULL) {
                    if (whichGPUs != NULL) {
                        cudaSetDevice(whichGPUs[i]);
                    }
                    cudaFree(d_k[i]);
                }
            }
            free(d_k);
            d_k = NULL;
        }
        free(worksize);
        worksize = NULL;
        if (planComplex != 0) {
            cufftDestroy(planComplex);
            planComplex = 0;
        }
        free(whichGPUs);
        whichGPUs = NULL;
    };

    for (int candidateGpus = maxGpusToTry; candidateGpus >= 2; --candidateGpus) {
        if (N % candidateGpus != 0) {
            continue;
        }

        whichGPUs = (int *)malloc(sizeof(int) * candidateGpus);
        worksize  = (size_t *)malloc(sizeof(size_t) * candidateGpus);
        d_k       = (float **)calloc(candidateGpus, sizeof(float *));

        if (whichGPUs == NULL || worksize == NULL || d_k == NULL) {
            fprintf(stderr, "Host allocation failed while setting up %d GPUs\n", candidateGpus);
            cleanupXtAttempt(candidateGpus);
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < candidateGpus; i++) {
            whichGPUs[i] = visibleGpuIds[i];
        }

        result = cufftCreate(&planComplex);
        if (result != CUFFT_SUCCESS) {
            cleanupXtAttempt(candidateGpus);
            continue;
        }

        result = cufftXtSetGPUs(planComplex, candidateGpus, whichGPUs);
        if (result != CUFFT_SUCCESS) {
            cleanupXtAttempt(candidateGpus);
            continue;
        }

        result = cufftMakePlan2d(planComplex, N, N, CUFFT_C2C, worksize);
        if (result != CUFFT_SUCCESS) {
            cleanupXtAttempt(candidateGpus);
            continue;
        }

        bool cudaSetupFailed = false;
        for (int i = 0; i < candidateGpus; i++) {
            cudaError_t cudaStatus = cudaSetDevice(whichGPUs[i]);
            if (cudaStatus == cudaSuccess) {
                cudaStatus = cudaMalloc((void **)&d_k[i], sizeof(float) * N);
            }
            if (cudaStatus == cudaSuccess) {
                cudaStatus = cudaMemcpy(d_k[i], k, sizeof(float) * N, cudaMemcpyHostToDevice);
            }
            if (cudaStatus != cudaSuccess) {
                cudaSetupFailed = true;
                break;
            }
        }
        if (cudaSetupFailed) {
            cleanupXtAttempt(candidateGpus);
            continue;
        }

        result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_f, CUFFT_XT_FORMAT_INPLACE);
        if (result != CUFFT_SUCCESS) {
            cleanupXtAttempt(candidateGpus);
            continue;
        }

        result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_d_f, CUFFT_XT_FORMAT_INPLACE);
        if (result != CUFFT_SUCCESS) {
            cleanupXtAttempt(candidateGpus);
            continue;
        }

        result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_out, CUFFT_XT_FORMAT_INPLACE);
        if (result != CUFFT_SUCCESS) {
            cleanupXtAttempt(candidateGpus);
            continue;
        }

        result = cufftXtMemcpy(planComplex, d_f, h_f, CUFFT_COPY_HOST_TO_DEVICE);
        if (result != CUFFT_SUCCESS) {
            printf("Initial *XtMemcpy failed with %d GPUs: code %d. Retrying with fewer GPUs.\n", candidateGpus, (int)result);
            cleanupXtAttempt(candidateGpus);
            continue;
        }

        nGPUs = candidateGpus;
        useXtPath = true;
        break;
    }

    if (!useXtPath) {
        nGPUs = 1;
        whichGPUs = (int *)malloc(sizeof(int));
        d_k = (float **)calloc(1, sizeof(float *));
        if (whichGPUs == NULL || d_k == NULL) {
            fprintf(stderr, "Host allocation failed while setting up single-GPU fallback\n");
            free(whichGPUs);
            free(d_k);
            exit(EXIT_FAILURE);
        }

        whichGPUs[0] = visibleGpuIds[0];
        checkCudaErrors(cudaSetDevice(whichGPUs[0]));
        checkCudaErrors(cudaMalloc((void **)&d_k[0], sizeof(float) * N));
        checkCudaErrors(cudaMemcpy(d_k[0], k, sizeof(float) * N, cudaMemcpyHostToDevice));

        result = cufftPlan2d(&planComplex, N, N, CUFFT_C2C);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "Single-GPU cufftPlan2d failed: code %d\n", (int)result);
            exit(EXIT_FAILURE);
        }

        checkCudaErrors(cudaMalloc((void **)&d_f_single, sizeof(cufftComplex) * N * N));
        checkCudaErrors(cudaMalloc((void **)&d_d_f_single, sizeof(cufftComplex) * N * N));
        checkCudaErrors(cudaMalloc((void **)&d_out_single, sizeof(cufftComplex) * N * N));
        checkCudaErrors(cudaMemcpy(d_f_single, h_f, sizeof(cufftComplex) * N * N, cudaMemcpyHostToDevice));
    }

    printf("\nRunning on GPUs\n");
    printf("Using %d GPUs\n", nGPUs);
    for (int i = 0; i < nGPUs; i++) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, whichGPUs[i]));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n",
               whichGPUs[i],
               deviceProp.name,
               deviceProp.major,
               deviceProp.minor);
    }

    if (useXtPath) {
        printf("Forward 2d FFT on multiple GPUs\n");
        printf("Solve Poisson Equation\n");
        printf("Inverse 2d FFT on multiple GPUs\n");
        for (int iter = 0; iter < maxIters; iter++) {
            result = cufftXtMemcpy(planComplex, d_f, h_f, CUFFT_COPY_HOST_TO_DEVICE);
            if (result != CUFFT_SUCCESS) {
                printf("*XtMemcpy failed\n");
                exit(EXIT_FAILURE);
            }

            result = cufftXtExecDescriptorC2C(planComplex, d_f, d_f, CUFFT_FORWARD);
            if (result != CUFFT_SUCCESS) {
                printf("*XtExecC2C failed\n");
                exit(EXIT_FAILURE);
            }

            result = cufftXtMemcpy(planComplex, d_d_f, d_f, CUFFT_COPY_DEVICE_TO_DEVICE);
            if (result != CUFFT_SUCCESS) {
                printf("*XtMemcpy failed\n");
                exit(EXIT_FAILURE);
            }

            solvePoissonEquation(d_d_f, d_out, d_k, N, nGPUs);

            result = cufftXtExecDescriptorC2C(planComplex, d_out, d_out, CUFFT_INVERSE);
            if (result != CUFFT_SUCCESS) {
                printf("*XtExecC2C failed\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    else {
        printf("Forward 2d FFT on single GPU\n");
        printf("Solve Poisson Equation\n");
        printf("Inverse 2d FFT on single GPU\n");
        for (int iter = 0; iter < maxIters; iter++) {
            checkCudaErrors(cudaMemcpy(d_f_single, h_f, sizeof(cufftComplex) * N * N, cudaMemcpyHostToDevice));

            result = cufftExecC2C(planComplex, d_f_single, d_f_single, CUFFT_FORWARD);
            if (result != CUFFT_SUCCESS) {
                fprintf(stderr, "cufftExecC2C forward failed: code %d\n", (int)result);
                exit(EXIT_FAILURE);
            }

            checkCudaErrors(cudaMemcpy(d_d_f_single, d_f_single, sizeof(cufftComplex) * N * N, cudaMemcpyDeviceToDevice));

            {
                int rowsPerGpu = N;
                dim3 dimGrid((N + BSZ_X - 1) / BSZ_X, (rowsPerGpu + BSZ_Y - 1) / BSZ_Y);
                dim3 dimBlock(BSZ_X, BSZ_Y);
                solvePoisson<<<dimGrid, dimBlock>>>(d_d_f_single, d_out_single, d_k[0], N, 0, 1);
                checkCudaErrors(cudaDeviceSynchronize());
                getLastCudaError("Kernel execution failed [ solvePoisson ]");
            }

            result = cufftExecC2C(planComplex, d_out_single, d_out_single, CUFFT_INVERSE);
            if (result != CUFFT_SUCCESS) {
                fprintf(stderr, "cufftExecC2C inverse failed: code %d\n", (int)result);
                exit(EXIT_FAILURE);
            }
        }
    }

    Complex *h_d_out = (Complex *)malloc(sizeof(Complex) * N * N);
    if (useXtPath) {
        result = cufftXtMemcpy(planComplex, h_d_out, d_out, CUFFT_COPY_DEVICE_TO_HOST);
        if (result != CUFFT_SUCCESS) {
            printf("*XtMemcpy failed\n");
            exit(EXIT_FAILURE);
        }
    }
    else {
        checkCudaErrors(cudaMemcpy(h_d_out, d_out_single, sizeof(cufftComplex) * N * N, cudaMemcpyDeviceToHost));
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
    free(y);
    free(f);
    free(u_a);
    free(worksize);

    if (d_k != NULL) {
        for (int i = 0; i < nGPUs; i++) {
            if (d_k[i] != NULL) {
                cudaSetDevice(whichGPUs[i]);
                cudaFree(d_k[i]);
            }
        }
        free(d_k);
    }

    if (useXtPath) {
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
    }
    else {
        cudaSetDevice(whichGPUs[0]);
        cudaFree(d_out_single);
        cudaFree(d_f_single);
        cudaFree(d_d_f_single);
    }

    free(whichGPUs);

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
