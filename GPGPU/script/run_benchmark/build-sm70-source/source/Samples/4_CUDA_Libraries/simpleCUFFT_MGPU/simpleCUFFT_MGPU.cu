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

/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

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

static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void                    ComplexPointwiseMulAndScale(cufftComplex *, cufftComplex *, int, float);

// Kernel for GPU
void multiplyCoefficient(cudaLibXtDesc *, cudaLibXtDesc *, int, float, int);

// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

// Padding functions
int PadData(const Complex *, Complex **, int, const Complex *, Complex **, int);

////////////////////////////////////////////////////////////////////////////////
// Data configuration
// The filter size is assumed to be a number smaller than the signal size
///////////////////////////////////////////////////////////////////////////////
constexpr int kDefaultProblemSize = 1018;
constexpr int kDefaultFilterSize   = 11;
constexpr int kDefaultMaxIters     = 1;
constexpr int kDefaultNumGPUs      = 2;

void usage()
{
    printf("Usage: simpleCUFFT_MGPU [--problem-size=<signal_size>] [--filter-size=<filter_size>] [--max-iters=<repeat_count>] [--num-gpus=<N>] [--help]\n");
    printf("  problem-size : signal size [default: %d]\n", kDefaultProblemSize);
    printf("  filter-size  : filter kernel size [default: %d]\n", kDefaultFilterSize);
    printf("  max-iters    : number of FFT convolution passes [default: %d]\n", kDefaultMaxIters);
    printf("  num-gpus     : number of GPUs to use [default: %d]\n", kDefaultNumGPUs);
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

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("\n[simpleCUFFT_MGPU] is starting...\n\n");

    int signal_size        = kDefaultProblemSize;
    int filter_kernel_size = kDefaultFilterSize;
    int max_iters          = kDefaultMaxIters;

    if (hasOption(argc, argv, "h") || hasOption(argc, argv, "help")) {
        usage();
        return EXIT_SUCCESS;
    }

    if (parseIntOption(argc, argv, "problem-size", &signal_size) || parseIntOption(argc, argv, "signal-size", &signal_size)) {
    }

    if (parseIntOption(argc, argv, "filter-size", &filter_kernel_size)) {
    }

    if (parseIntOption(argc, argv, "max-iters", &max_iters) || parseIntOption(argc, argv, "iterations", &max_iters)) {
    }

    int requested_gpus = kDefaultNumGPUs;
    parseIntOption(argc, argv, "num-gpus", &requested_gpus);

    if (signal_size < 1) {
        fprintf(stderr, "problem-size must be >= 1, got %d\n", signal_size);
        return EXIT_FAILURE;
    }

    if (filter_kernel_size < 1 || filter_kernel_size >= signal_size) {
        fprintf(stderr, "filter-size must be >= 1 and smaller than problem-size, got %d and %d\n", filter_kernel_size, signal_size);
        return EXIT_FAILURE;
    }

    if (max_iters < 1) {
        fprintf(stderr, "max-iters must be >= 1, got %d\n", max_iters);
        return EXIT_FAILURE;
    }

    printf("Configured problem size = %d\n", signal_size);
    printf("Configured filter size = %d\n", filter_kernel_size);
    printf("Configured fixed iterations = %d\n", max_iters);

    int GPU_N;
    checkCudaErrors(cudaGetDeviceCount(&GPU_N));

    int nGPUs = requested_gpus;
    if (nGPUs < 1) {
        fprintf(stderr, "num-gpus must be >= 1, got %d\n", nGPUs);
        exit(EXIT_FAILURE);
    }
    if (GPU_N < nGPUs) {
        printf("No. of GPU on node %d, but %d requested\n", GPU_N, nGPUs);
        printf("%d GPUs are required to run simpleCUFFT_MGPU sample code\n", nGPUs);
        exit(EXIT_WAIVED);
    }

    int *whichGPUs = (int *)malloc(sizeof(int) * nGPUs);
    for (int i = 0; i < nGPUs; i++) {
        whichGPUs[i] = i;
    }

    for (int i = 0; i < GPU_N; i++) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n",
               i,
               deviceProp.name,
               deviceProp.major,
               deviceProp.minor);
    }

    // Allocate host memory for the signal
    Complex *h_signal = (Complex *)malloc(sizeof(Complex) * signal_size);

    // Initialize the memory for the signal
    for (int i = 0; i < signal_size; ++i) {
        h_signal[i].x = rand() / (float)RAND_MAX;
        h_signal[i].y = 0;
    }

    // Allocate host memory for the filter
    Complex *h_filter_kernel = (Complex *)malloc(sizeof(Complex) * filter_kernel_size);

    // Initialize the memory for the filter
    for (int i = 0; i < filter_kernel_size; ++i) {
        h_filter_kernel[i].x = rand() / (float)RAND_MAX;
        h_filter_kernel[i].y = 0;
    }

    // Pad signal and filter kernel
    Complex *h_padded_signal;
    Complex *h_padded_filter_kernel;
    int      new_size =
        PadData(h_signal, &h_padded_signal, signal_size, h_filter_kernel, &h_padded_filter_kernel, filter_kernel_size);

    // cufftCreate() - Create an empty plan
    cufftResult    result                     = CUFFT_SUCCESS;
    cufftHandle    plan_input                 = 0;
    size_t        *worksize                   = NULL;
    cudaLibXtDesc *d_signal                   = NULL;
    cudaLibXtDesc *d_out_signal               = NULL;
    cudaLibXtDesc *d_filter_kernel            = NULL;
    cudaLibXtDesc *d_out_filter_kernel        = NULL;
    cufftComplex  *d_signal_single            = NULL;
    cufftComplex  *d_out_signal_single        = NULL;
    cufftComplex  *d_filter_kernel_single     = NULL;
    cufftComplex  *d_out_filter_kernel_single = NULL;

    printf("\nRunning on GPUs\n");
    for (int i = 0; i < nGPUs; i++) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, whichGPUs[i]));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n",
               whichGPUs[i],
               deviceProp.name,
               deviceProp.major,
               deviceProp.minor);
    }

    // Create host pointer pointing to padded signal
    Complex *h_convolved_signal = h_padded_signal;

    // Allocate host memory for the convolution result
    Complex *h_convolved_signal_ref = (Complex *)malloc(sizeof(Complex) * signal_size);

    if (nGPUs == 1) {
        checkCudaErrors(cudaSetDevice(whichGPUs[0]));
        checkCudaErrors(cufftPlan1d(&plan_input, new_size, CUFFT_C2C, 1));

        checkCudaErrors(cudaMalloc((void **)&d_signal_single, sizeof(cufftComplex) * new_size));
        checkCudaErrors(cudaMalloc((void **)&d_out_signal_single, sizeof(cufftComplex) * new_size));
        checkCudaErrors(cudaMalloc((void **)&d_filter_kernel_single, sizeof(cufftComplex) * new_size));
        checkCudaErrors(cudaMalloc((void **)&d_out_filter_kernel_single, sizeof(cufftComplex) * new_size));

        for (int iter = 0; iter < max_iters; ++iter) {
            checkCudaErrors(cudaMemcpy(d_signal_single,
                                       h_padded_signal,
                                       sizeof(cufftComplex) * new_size,
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_filter_kernel_single,
                                       h_padded_filter_kernel,
                                       sizeof(cufftComplex) * new_size,
                                       cudaMemcpyHostToDevice));

            checkCudaErrors(cufftExecC2C(plan_input, d_signal_single, d_signal_single, CUFFT_FORWARD));
            checkCudaErrors(cufftExecC2C(plan_input, d_filter_kernel_single, d_filter_kernel_single, CUFFT_FORWARD));

            checkCudaErrors(cudaMemcpy(d_out_signal_single,
                                       d_signal_single,
                                       sizeof(cufftComplex) * new_size,
                                       cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(d_out_filter_kernel_single,
                                       d_filter_kernel_single,
                                       sizeof(cufftComplex) * new_size,
                                       cudaMemcpyDeviceToDevice));

            ComplexPointwiseMulAndScale<<<32, 256>>>(d_out_signal_single,
                                                     d_out_filter_kernel_single,
                                                     new_size,
                                                     1.0f / new_size);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

            checkCudaErrors(cufftExecC2C(plan_input, d_out_signal_single, d_out_signal_single, CUFFT_INVERSE));
        }

        checkCudaErrors(cudaMemcpy(h_convolved_signal,
                                   d_out_signal_single,
                                   sizeof(cufftComplex) * new_size,
                                   cudaMemcpyDeviceToHost));
    }
    else {
        checkCudaErrors(cufftCreate(&plan_input));

        // cufftXtSetGPUs() - Define which GPUs to use
        result = cufftXtSetGPUs(plan_input, nGPUs, whichGPUs);

        if (result == CUFFT_INVALID_DEVICE) {
            printf("This sample requires two GPUs on the same board.\n");
            printf("No such board was found. Waiving sample.\n");
            exit(EXIT_WAIVED);
        }
        else if (result != CUFFT_SUCCESS) {
            printf("cufftXtSetGPUs failed\n");
            exit(EXIT_FAILURE);
        }

        worksize = (size_t *)malloc(sizeof(size_t) * nGPUs);

        // cufftMakePlan1d() - Create the plan
        checkCudaErrors(cufftMakePlan1d(plan_input, new_size, CUFFT_C2C, 1, worksize));

        // cufftXtMalloc() - Malloc data on multiple GPUs
        checkCudaErrors(cufftXtMalloc(plan_input, (cudaLibXtDesc **)&d_signal, CUFFT_XT_FORMAT_INPLACE));
        checkCudaErrors(cufftXtMalloc(plan_input, (cudaLibXtDesc **)&d_out_signal, CUFFT_XT_FORMAT_INPLACE));
        checkCudaErrors(cufftXtMalloc(plan_input, (cudaLibXtDesc **)&d_filter_kernel, CUFFT_XT_FORMAT_INPLACE));
        checkCudaErrors(cufftXtMalloc(plan_input, (cudaLibXtDesc **)&d_out_filter_kernel, CUFFT_XT_FORMAT_INPLACE));

        for (int iter = 0; iter < max_iters; ++iter) {
            checkCudaErrors(cufftXtMemcpy(plan_input, d_signal, h_padded_signal, CUFFT_COPY_HOST_TO_DEVICE));
            checkCudaErrors(cufftXtMemcpy(plan_input, d_filter_kernel, h_padded_filter_kernel, CUFFT_COPY_HOST_TO_DEVICE));

            checkCudaErrors(cufftXtExecDescriptorC2C(plan_input, d_signal, d_signal, CUFFT_FORWARD));
            checkCudaErrors(cufftXtExecDescriptorC2C(plan_input, d_filter_kernel, d_filter_kernel, CUFFT_FORWARD));

            checkCudaErrors(cufftXtMemcpy(plan_input, d_out_signal, d_signal, CUFFT_COPY_DEVICE_TO_DEVICE));
            checkCudaErrors(cufftXtMemcpy(plan_input, d_out_filter_kernel, d_filter_kernel, CUFFT_COPY_DEVICE_TO_DEVICE));

            if (iter == 0) {
                printf("\n\nValue of Library Descriptor\n");
                printf("Number of GPUs %d\n", d_out_signal->descriptor->nGPUs);
                printf("Device id ");
                for (int g = 0; g < d_out_signal->descriptor->nGPUs; g++) {
                    printf(" %d", d_out_signal->descriptor->GPUs[g]);
                }
                printf("\n");
                printf("Data size on GPU");
                for (int g = 0; g < d_out_signal->descriptor->nGPUs; g++) {
                    printf(" %ld", (long)(d_out_signal->descriptor->size[g] / sizeof(cufftComplex)));
                }
                printf("\n");
            }

            multiplyCoefficient(d_out_signal, d_out_filter_kernel, new_size, 1.0f / new_size, nGPUs);
            checkCudaErrors(cufftXtExecDescriptorC2C(plan_input, d_out_signal, d_out_signal, CUFFT_INVERSE));
        }

        checkCudaErrors(cufftXtMemcpy(plan_input, h_convolved_signal, d_out_signal, CUFFT_COPY_DEVICE_TO_HOST));
    }

    printf("Completed fixed iterations = %d\n", max_iters);

    // Convolve on the host
    Convolve(h_signal, signal_size, h_filter_kernel, filter_kernel_size, h_convolved_signal_ref);

    // Compare CPU and GPU result
    bool bTestResult =
        sdkCompareL2fe((float *)h_convolved_signal_ref, (float *)h_convolved_signal, 2 * signal_size, 1e-5f);
    printf("\nvalue of TestResult %d\n", bTestResult);

    // Cleanup memory
    free(whichGPUs);
    free(worksize);
    free(h_signal);
    free(h_filter_kernel);
    free(h_padded_signal);
    free(h_padded_filter_kernel);
    free(h_convolved_signal_ref);

    if (nGPUs == 1) {
        checkCudaErrors(cudaFree(d_signal_single));
        checkCudaErrors(cudaFree(d_filter_kernel_single));
        checkCudaErrors(cudaFree(d_out_signal_single));
        checkCudaErrors(cudaFree(d_out_filter_kernel_single));
    }
    else {
        // cudaXtFree() - Free GPU memory
        checkCudaErrors(cufftXtFree(d_signal));
        checkCudaErrors(cufftXtFree(d_filter_kernel));
        checkCudaErrors(cufftXtFree(d_out_signal));
        checkCudaErrors(cufftXtFree(d_out_filter_kernel));
    }

    if (plan_input != 0) {
        // cufftDestroy() - Destroy FFT plan
        checkCudaErrors(cufftDestroy(plan_input));
    }

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

///////////////////////////////////////////////////////////////////////////////////
// Function for padding original data
//////////////////////////////////////////////////////////////////////////////////
int PadData(const Complex *signal,
            Complex      **padded_signal,
            int            signal_size,
            const Complex *filter_kernel,
            Complex      **padded_filter_kernel,
            int            filter_kernel_size)
{
    int minRadius = filter_kernel_size / 2;
    int maxRadius = filter_kernel_size - minRadius;
    int new_size  = signal_size + maxRadius;

    // Pad signal
    Complex *new_data = (Complex *)malloc(sizeof(Complex) * new_size);
    memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
    memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
    *padded_signal = new_data;

    // Pad filter
    new_data = (Complex *)malloc(sizeof(Complex) * new_size);
    memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
    memset(new_data + maxRadius, 0, (new_size - filter_kernel_size) * sizeof(Complex));
    memcpy(new_data + new_size - minRadius, filter_kernel, minRadius * sizeof(Complex));
    *padded_filter_kernel = new_data;

    return new_size;
}

////////////////////////////////////////////////////////////////////////////////
// Filtering operations - Computing Convolution on the host
////////////////////////////////////////////////////////////////////////////////
void Convolve(const Complex *signal,
              int            signal_size,
              const Complex *filter_kernel,
              int            filter_kernel_size,
              Complex       *filtered_signal)
{
    int minRadius = filter_kernel_size / 2;
    int maxRadius = filter_kernel_size - minRadius;

    // Loop over output element indices
    for (int i = 0; i < signal_size; ++i) {
        filtered_signal[i].x = filtered_signal[i].y = 0;

        // Loop over convolution indices
        for (int j = -maxRadius + 1; j <= minRadius; ++j) {
            int k = i + j;

            if (k >= 0 && k < signal_size) {
                filtered_signal[i] =
                    ComplexAdd(filtered_signal[i], ComplexMul(signal[k], filter_kernel[minRadius - j]));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//  Launch Kernel on multiple GPU
////////////////////////////////////////////////////////////////////////////////
void multiplyCoefficient(cudaLibXtDesc *d_signal, cudaLibXtDesc *d_filter_kernel, int new_size, float val, int nGPUs)
{
    int device;
    // Launch the ComplexPointwiseMulAndScale<<< >>> kernel on multiple GPU
    for (int i = 0; i < nGPUs; i++) {
        device = d_signal->descriptor->GPUs[i];

        // Set device
        checkCudaErrors(cudaSetDevice(device));

        // Perform GPU computations
        ComplexPointwiseMulAndScale<<<32, 256>>>((cufftComplex *)d_signal->descriptor->data[i],
                                                 (cufftComplex *)d_filter_kernel->descriptor->data[i],
                                                 int(d_signal->descriptor->size[i] / sizeof(cufftComplex)),
                                                 val);
    }

    // Wait for device to finish all operation
    for (int i = 0; i < nGPUs; i++) {
        device = d_signal->descriptor->GPUs[i];
        checkCudaErrors(cudaSetDevice(device));
        cudaDeviceSynchronize();
        // Check if kernel execution generated and error
        getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
    }
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}
// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(cufftComplex *a, cufftComplex *b, int size, float scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads) {
        a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
    }
}
