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

/*
 * This sample implements a conjugate gradient solver on multiple GPU using
 * Unified Memory optimized prefetching and usage hints.
 *
 */

// includes, system
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>
#include <vector>

// Utilities and system includes
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <helper_cuda.h>      // helper function CUDA error checking and initialization
#include <helper_functions.h> // helper for shared functions common to CUDA Samples

namespace cg = cooperative_groups;

const char *sSDKname = "conjugateGradientMultiDeviceCG";

#define ENABLE_CPU_DEBUG_CODE 0
#define THREADS_PER_BLOCK     512

constexpr int kDefaultProblemSize = 10485760 * 2;
constexpr int kDefaultMaxIters    = 10000;

__device__ double grid_dot_result = 0.0;

void usage()
{
    printf("Usage: %s [--n=<rows>] [--max-iters=<count>] [--help]\n", sSDKname);
    printf("  --n=<rows>           Size of the tridiagonal system [default: %d]\n", kDefaultProblemSize);
    printf("  --max-iters=<count>  Run a fixed number of CG iterations and ignore convergence [default: %d]\n",
           kDefaultMaxIters);
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

    return checkCmdLineFlag(argc, (const char **)argv, name) != 0;
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

    if (checkCmdLineFlag(argc, (const char **)argv, name)) {
        *value = getCmdLineArgumentInt(argc, (const char **)argv, name);
        return true;
    }

    return false;
}

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand() / RAND_MAX + 10.0f;
    val[1] = (float)rand() / RAND_MAX;
    int start;

    for (int i = 1; i < N; i++) {
        if (i > 1) {
            I[i] = I[i - 1] + 3;
        }
        else {
            I[1] = 2;
        }

        start        = (i - 1) * 3 + 2;
        J[start]     = i - 1;
        J[start + 1] = i;

        if (i < N - 1) {
            J[start + 2] = i + 1;
        }

        val[start]     = val[start - 1];
        val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

        if (i < N - 1) {
            val[start + 2] = (float)rand() / RAND_MAX;
        }
    }

    I[N] = nz;
}

// I - contains location of the given non-zero element in the row of the matrix
// J - contains location of the given non-zero element in the column of the
// matrix val - contains values of the given non-zero elements of the matrix
// inputVecX - input vector to be multiplied
// outputVecY - resultant vector
void cpuSpMV(int *I, int *J, float *val, int nnz, int num_rows, float alpha, float *inputVecX, float *outputVecY)
{
    for (int i = 0; i < num_rows; i++) {
        int num_elems_this_row = I[i + 1] - I[i];

        float output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++) {
            output += alpha * val[I[i] + j] * inputVecX[J[I[i] + j]];
        }
        outputVecY[i] = output;
    }

    return;
}

float dotProduct(float *vecA, float *vecB, int size)
{
    float result = 0.0;

    for (int i = 0; i < size; i++) {
        result = result + (vecA[i] * vecB[i]);
    }

    return result;
}

void scaleVector(float *vec, float alpha, int size)
{
    for (int i = 0; i < size; i++) {
        vec[i] = alpha * vec[i];
    }
}

void saxpy(float *x, float *y, float a, int size)
{
    for (int i = 0; i < size; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void cpuConjugateGrad(int *I, int *J, float *val, float *x, float *Ax, float *p, float *r, int nnz, int N, int max_iter)
{
    float alpha   = 1.0;
    float alpham1 = -1.0;
    float r0      = 0.0, b, a, na;

    cpuSpMV(I, J, val, nnz, N, alpha, x, Ax);
    saxpy(Ax, r, alpham1, N);

    float r1 = dotProduct(r, r, N);

    int k = 1;

    while (k <= max_iter) {
        if (k > 1) {
            b = r1 / r0;
            scaleVector(p, b, N);

            saxpy(r, p, alpha, N);
        }
        else {
            for (int i = 0; i < N; i++)
                p[i] = r[i];
        }

        cpuSpMV(I, J, val, nnz, N, alpha, p, Ax);

        float dot = dotProduct(p, Ax, N);
        a         = r1 / dot;

        saxpy(p, x, a, N);
        na = -a;
        saxpy(Ax, r, na, N);

        r0 = r1;
        r1 = dotProduct(r, r, N);

        printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }
}

// Data filled on CPU needed for MultiGPU operations.
struct MultiDeviceData
{
    unsigned char *hostMemoryArrivedList;
    unsigned int   numDevices;
    unsigned int   deviceRank;
};

// Class used for coordination of multiple devices.
class PeerGroup
{
    const MultiDeviceData &data;
    const cg::grid_group  &grid;

    __device__ unsigned char load_arrived(unsigned char *arrived) const
    {
#if __CUDA_ARCH__ < 700
        return *(volatile unsigned char *)arrived;
#else
        unsigned int result;
        asm volatile("ld.acquire.sys.global.u8 %0, [%1];" : "=r"(result) : "l"(arrived) : "memory");
        return result;
#endif
    }

    __device__ void store_arrived(unsigned char *arrived, unsigned char val) const
    {
#if __CUDA_ARCH__ < 700
        *(volatile unsigned char *)arrived = val;
#else
        unsigned int reg_val = val;
        asm volatile("st.release.sys.global.u8 [%1], %0;" ::"r"(reg_val) "l"(arrived) : "memory");

        // Avoids compiler warnings from unused variable val.
        (void)(reg_val = reg_val);
#endif
    }

public:
    __device__ PeerGroup(const MultiDeviceData &data, const cg::grid_group &grid)
        : data(data)
        , grid(grid) {};

    __device__ unsigned int size() const { return data.numDevices * grid.size(); }

    __device__ unsigned int thread_rank() const { return data.deviceRank * grid.size() + grid.thread_rank(); }

    __device__ void sync() const
    {
        grid.sync();

        // One thread from each grid participates in the sync.
        if (grid.thread_rank() == 0) {
            if (data.deviceRank == 0) {
                // Leader grid waits for others to join and then releases them.
                // Other GPUs can arrive in any order, so the leader have to wait for
                // all others.
                for (int i = 0; i < data.numDevices - 1; i++) {
                    while (load_arrived(&data.hostMemoryArrivedList[i]) == 0)
                        ;
                }
                for (int i = 0; i < data.numDevices - 1; i++) {
                    store_arrived(&data.hostMemoryArrivedList[i], 0);
                }
                __threadfence_system();
            }
            else {
                // Other grids note their arrival and wait to be released.
                store_arrived(&data.hostMemoryArrivedList[data.deviceRank - 1], 1);
                while (load_arrived(&data.hostMemoryArrivedList[data.deviceRank - 1]) == 1)
                    ;
            }
        }

        grid.sync();
    }
};

__device__ void gpuSpMV(int             *I,
                        int             *J,
                        float           *val,
                        int              nnz,
                        int              num_rows,
                        float            alpha,
                        float           *inputVecX,
                        float           *outputVecY,
                        const PeerGroup &peer_group)
{
    for (int i = peer_group.thread_rank(); i < num_rows; i += peer_group.size()) {
        int row_elem           = I[i];
        int next_row_elem      = I[i + 1];
        int num_elems_this_row = next_row_elem - row_elem;

        float output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++) {
            output += alpha * val[row_elem + j] * inputVecX[J[row_elem + j]];
        }

        outputVecY[i] = output;
    }
}

__device__ void gpuSaxpy(float *x, float *y, float a, int size, const PeerGroup &peer_group)
{
    for (int i = peer_group.thread_rank(); i < size; i += peer_group.size()) {
        y[i] = a * x[i] + y[i];
    }
}

__device__ void
gpuDotProduct(float *vecA, float *vecB, int size, const cg::thread_block &cta, const PeerGroup &peer_group)
{
    extern __shared__ double tmp[];

    double temp_sum = 0.0;

    for (int i = peer_group.thread_rank(); i < size; i += peer_group.size()) {
        temp_sum += (double)(vecA[i] * vecB[i]);
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

    if (tile32.thread_rank() == 0) {
        tmp[tile32.meta_group_rank()] = temp_sum;
    }

    cg::sync(cta);

    if (tile32.meta_group_rank() == 0) {
        temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
        temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

        if (tile32.thread_rank() == 0) {
            atomicAdd(&grid_dot_result, temp_sum);
        }
    }
}

__device__ void gpuCopyVector(float *srcA, float *destB, int size, const PeerGroup &peer_group)
{
    for (int i = peer_group.thread_rank(); i < size; i += peer_group.size()) {
        destB[i] = srcA[i];
    }
}

__device__ void gpuScaleVectorAndSaxpy(float *x, float *y, float a, float scale, int size, const PeerGroup &peer_group)
{
    for (int i = peer_group.thread_rank(); i < size; i += peer_group.size()) {
        y[i] = a * x[i] + scale * y[i];
    }
}

extern "C" __global__ void multiGpuConjugateGradient(int            *I,
                                                     int            *J,
                                                     float          *val,
                                                     float          *x,
                                                     float          *Ax,
                                                     float          *p,
                                                     float          *r,
                                                     double         *dot_result,
                                                     int             nnz,
                                                     int             N,
                                                     int             max_iter,
                                                     MultiDeviceData multi_device_data)
{
    cg::thread_block cta  = cg::this_thread_block();
    cg::grid_group   grid = cg::this_grid();
    PeerGroup        peer_group(multi_device_data, grid);

    float alpha   = 1.0;
    float alpham1 = -1.0;
    float r0      = 0.0, r1, b, a, na;

    for (int i = peer_group.thread_rank(); i < N; i += peer_group.size()) {
        r[i] = 1.0;
        x[i] = 0.0;
    }

    cg::sync(grid);

    gpuSpMV(I, J, val, nnz, N, alpha, x, Ax, peer_group);

    cg::sync(grid);

    gpuSaxpy(Ax, r, alpham1, N, peer_group);

    cg::sync(grid);

    gpuDotProduct(r, r, N, cta, peer_group);

    cg::sync(grid);

    if (grid.thread_rank() == 0) {
        atomicAdd_system(dot_result, grid_dot_result);
        grid_dot_result = 0.0;
    }
    peer_group.sync();

    r1 = *dot_result;

    int k = 1;
    while (k <= max_iter) {
        if (k > 1) {
            b = r1 / r0;
            gpuScaleVectorAndSaxpy(r, p, alpha, b, N, peer_group);
        }
        else {
            gpuCopyVector(r, p, N, peer_group);
        }

        peer_group.sync();

        gpuSpMV(I, J, val, nnz, N, alpha, p, Ax, peer_group);

        if (peer_group.thread_rank() == 0) {
            *dot_result = 0.0;
        }
        peer_group.sync();

        gpuDotProduct(p, Ax, N, cta, peer_group);

        cg::sync(grid);

        if (grid.thread_rank() == 0) {
            atomicAdd_system(dot_result, grid_dot_result);
            grid_dot_result = 0.0;
        }
        peer_group.sync();

        a = r1 / *dot_result;

        gpuSaxpy(p, x, a, N, peer_group);

        na = -a;

        gpuSaxpy(Ax, r, na, N, peer_group);

        r0 = r1;

        peer_group.sync();

        if (peer_group.thread_rank() == 0) {
            *dot_result = 0.0;
        }

        peer_group.sync();

        gpuDotProduct(r, r, N, cta, peer_group);

        cg::sync(grid);

        if (grid.thread_rank() == 0) {
            atomicAdd_system(dot_result, grid_dot_result);
            grid_dot_result = 0.0;
        }
        peer_group.sync();

        r1 = *dot_result;
        k++;
    }
}

// Map of device version to device number
std::multimap<std::pair<int, int>, int> getIdenticalGPUs()
{
    int numGpus = 0;
    checkCudaErrors(cudaGetDeviceCount(&numGpus));

    std::multimap<std::pair<int, int>, int> identicalGpus;

    for (int i = 0; i < numGpus; i++) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));

        // Filter unsupported devices
        if (deviceProp.cooperativeLaunch && deviceProp.concurrentManagedAccess) {
            identicalGpus.emplace(std::make_pair(deviceProp.major, deviceProp.minor), i);
        }
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n",
               i,
               deviceProp.name,
               deviceProp.major,
               deviceProp.minor);
    }

    return identicalGpus;
}

int main(int argc, char **argv)
{
    constexpr size_t kMinGpusRequired = 1;
    constexpr size_t kMaxGpusSupported = 4;
    int              N = kDefaultProblemSize, nz = 0, max_iter = kDefaultMaxIters, *I = NULL, *J = NULL;
    float           *val = NULL;
    const float      tol = 1e-5f;
    float           *x;
    float            rhs = 1.0;
    float            r1;
    float           *r, *p, *Ax;

    printf("Starting [%s]...\n", sSDKname);

    if (hasOption(argc, argv, "h") || hasOption(argc, argv, "help")) {
        usage();
        return EXIT_SUCCESS;
    }

    if (parseIntOption(argc, argv, "problem-size", &N) || parseIntOption(argc, argv, "n", &N)) {
    }

    if (parseIntOption(argc, argv, "iterations", &max_iter) || parseIntOption(argc, argv, "max-iters", &max_iter)) {
    }

    if (N < 2) {
        fprintf(stderr, "Problem size N must be >= 2, got %d\n", N);
        return EXIT_FAILURE;
    }

    if (max_iter < 1) {
        fprintf(stderr, "max-iters must be >= 1, got %d\n", max_iter);
        return EXIT_FAILURE;
    }

    printf("Configured problem size N = %d\n", N);
    printf("Configured fixed iterations = %d\n", max_iter);
    auto gpusByArch = getIdenticalGPUs();

    auto it  = gpusByArch.begin();
    auto end = gpusByArch.end();

    auto bestFit = std::make_pair(it, it);
    // use std::distance to find the largest number of GPUs amongst architectures
    auto distance = [](decltype(bestFit) p) { return std::distance(p.first, p.second); };

    // Read each unique key/pair element in order
    for (; it != end; it = gpusByArch.upper_bound(it->first)) {
        // first and second are iterators bounded within the architecture group
        auto testFit = gpusByArch.equal_range(it->first);
        // Always use devices with highest architecture version or whichever has the
        // most devices available
        if (distance(bestFit) <= distance(testFit))
            bestFit = testFit;
    }

    size_t numGpusToUse = static_cast<size_t>(distance(bestFit));
    if (numGpusToUse > kMaxGpusSupported) {
        numGpusToUse = kMaxGpusSupported;
    }

    if (numGpusToUse < kMinGpusRequired) {
        printf("No %zu or more GPUs with same architecture capable of concurrentManagedAccess found. "
               "\nWaiving the sample\n",
               kMinGpusRequired);
        exit(EXIT_WAIVED);
    }

    std::vector<int> candidateDeviceIds;
    for (auto itr = bestFit.first; itr != bestFit.second; ++itr) {
        candidateDeviceIds.push_back(itr->second);
    }

    auto hasBidirectionalPeerAccess = [](int lhs, int rhs) {
        int forwardAccess = 0;
        int reverseAccess = 0;
        checkCudaErrors(cudaDeviceCanAccessPeer(&forwardAccess, lhs, rhs));
        checkCudaErrors(cudaDeviceCanAccessPeer(&reverseAccess, rhs, lhs));
        return forwardAccess && reverseAccess;
    };

    std::vector<int> p2pDeviceIds;
    auto findP2PDeviceGroup = [&](auto &&self, size_t startIndex) -> bool {
        if (p2pDeviceIds.size() == numGpusToUse) {
            return true;
        }
        if (candidateDeviceIds.size() - startIndex < numGpusToUse - p2pDeviceIds.size()) {
            return false;
        }

        for (size_t i = startIndex; i < candidateDeviceIds.size(); ++i) {
            const int candidateDeviceId = candidateDeviceIds[i];
            bool      isCompatible      = true;

            for (int selectedDeviceId : p2pDeviceIds) {
                if (!hasBidirectionalPeerAccess(candidateDeviceId, selectedDeviceId)) {
                    isCompatible = false;
                    break;
                }
            }

            if (!isCompatible) {
                continue;
            }

            p2pDeviceIds.push_back(candidateDeviceId);
            if (self(self, i + 1)) {
                return true;
            }
            p2pDeviceIds.pop_back();
        }

        return false;
    };

    const bool foundP2PDeviceGroup = findP2PDeviceGroup(findP2PDeviceGroup, 0);
    std::set<int> bestFitDeviceIds;

    if (foundP2PDeviceGroup) {
        for (int deviceId : p2pDeviceIds) {
            bestFitDeviceIds.emplace(deviceId);
        }

        printf("Selected p2p capable devices - ");
        for (int deviceId : p2pDeviceIds) {
            printf("deviceId = %d  ", deviceId);
        }
        printf("\n");

        // Perform cudaDeviceEnablePeerAccess in both directions for all
        // participating devices.
        for (auto p1_itr = bestFitDeviceIds.begin(); p1_itr != bestFitDeviceIds.end(); ++p1_itr) {
            checkCudaErrors(cudaSetDevice(*p1_itr));
            for (auto p2_itr = bestFitDeviceIds.begin(); p2_itr != bestFitDeviceIds.end(); ++p2_itr) {
                if (*p1_itr != *p2_itr) {
                    checkCudaErrors(cudaDeviceEnablePeerAccess(*p2_itr, 0));
                    checkCudaErrors(cudaSetDevice(*p1_itr));
                }
            }
        }
    }
    else {
        printf("Unable to find %zu fully peer-accessible GPUs. Falling back to non-p2p selection.\n", numGpusToUse);
        for (int deviceId : candidateDeviceIds) {
            if (bestFitDeviceIds.size() >= numGpusToUse) {
                break;
            }
            bestFitDeviceIds.emplace(deviceId);
        }
    }

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    nz = (N - 2) * 3 + 4;

    checkCudaErrors(cudaMallocManaged((void **)&I, sizeof(int) * (N + 1)));
    checkCudaErrors(cudaMallocManaged((void **)&J, sizeof(int) * nz));
    checkCudaErrors(cudaMallocManaged((void **)&val, sizeof(float) * nz));

    float *val_cpu = (float *)malloc(sizeof(float) * nz);

    genTridiag(I, J, val_cpu, N, nz);

    memcpy(val, val_cpu, sizeof(float) * nz);
    cudaMemLocation deviceLoc;
    deviceLoc.type = cudaMemLocationTypeDevice;
    deviceLoc.id   = 0; // Device location with initial device 0
    checkCudaErrors(cudaMemAdvise(I, sizeof(int) * (N + 1), cudaMemAdviseSetReadMostly, deviceLoc));
    checkCudaErrors(cudaMemAdvise(J, sizeof(int) * nz, cudaMemAdviseSetReadMostly, deviceLoc));
    checkCudaErrors(cudaMemAdvise(val, sizeof(float) * nz, cudaMemAdviseSetReadMostly, deviceLoc));

    checkCudaErrors(cudaMallocManaged((void **)&x, sizeof(float) * N));

    double *dot_result;
    checkCudaErrors(cudaMallocManaged((void **)&dot_result, sizeof(double)));

    checkCudaErrors(cudaMemset(dot_result, 0, sizeof(double)));

    // temp memory for ConjugateGradient
    checkCudaErrors(cudaMallocManaged((void **)&r, N * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&p, N * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&Ax, N * sizeof(float)));

    std::cout << "\nRunning on GPUs = " << numGpusToUse << std::endl;
    cudaStream_t nStreams[kMaxGpusSupported];

    int  sMemSize       = sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);
    int  numBlocksPerSm = INT_MAX;
    int  numThreads     = THREADS_PER_BLOCK;
    int  numSms         = INT_MAX;
    auto deviceId       = bestFitDeviceIds.begin();

    // set numSms & numBlocksPerSm to be the lowest across all selected devices
    while (deviceId != bestFitDeviceIds.end()) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaSetDevice(*deviceId));
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, *deviceId));

        int numBlocksPerSm_current = 0;
        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSm_current, multiGpuConjugateGradient, numThreads, sMemSize));

        if (numBlocksPerSm > numBlocksPerSm_current) {
            numBlocksPerSm = numBlocksPerSm_current;
        }
        if (numSms > deviceProp.multiProcessorCount) {
            numSms = deviceProp.multiProcessorCount;
        }
        deviceId++;
    }

    if (!numBlocksPerSm) {
        printf("Max active blocks per SM is returned as 0.\n Hence, Waiving the "
               "sample\n");
        exit(EXIT_WAIVED);
    }

    int device_count       = 0;
    int totalThreadsPerGPU = numSms * numBlocksPerSm * THREADS_PER_BLOCK;
    deviceId               = bestFitDeviceIds.begin();
    while (deviceId != bestFitDeviceIds.end()) {
        checkCudaErrors(cudaSetDevice(*deviceId));
        checkCudaErrors(cudaStreamCreate(&nStreams[device_count]));

        int perGPUIter = N / (totalThreadsPerGPU * numGpusToUse);
        int offset_Ax  = device_count * totalThreadsPerGPU;
        int offset_r   = device_count * totalThreadsPerGPU;
        int offset_p   = device_count * totalThreadsPerGPU;
        int offset_x   = device_count * totalThreadsPerGPU;

        // Create device location with specific device ID
        cudaMemLocation deviceLoc;
        deviceLoc.type = cudaMemLocationTypeDevice;
        deviceLoc.id   = *deviceId;
        checkCudaErrors(cudaMemPrefetchAsync(I, sizeof(int) * N, deviceLoc, 0, nStreams[device_count]));
        checkCudaErrors(cudaMemPrefetchAsync(val, sizeof(float) * nz, deviceLoc, 0, nStreams[device_count]));
        checkCudaErrors(cudaMemPrefetchAsync(J, sizeof(float) * nz, deviceLoc, 0, nStreams[device_count]));

        if (offset_Ax <= N) {
            for (int i = 0; i < perGPUIter; i++) {
                cudaMemAdvise(
                    Ax + offset_Ax, sizeof(float) * totalThreadsPerGPU, cudaMemAdviseSetPreferredLocation, deviceLoc);
                cudaMemAdvise(
                    r + offset_r, sizeof(float) * totalThreadsPerGPU, cudaMemAdviseSetPreferredLocation, deviceLoc);
                cudaMemAdvise(
                    x + offset_x, sizeof(float) * totalThreadsPerGPU, cudaMemAdviseSetPreferredLocation, deviceLoc);
                cudaMemAdvise(
                    p + offset_p, sizeof(float) * totalThreadsPerGPU, cudaMemAdviseSetPreferredLocation, deviceLoc);

                cudaMemAdvise(
                    Ax + offset_Ax, sizeof(float) * totalThreadsPerGPU, cudaMemAdviseSetAccessedBy, deviceLoc);
                cudaMemAdvise(r + offset_r, sizeof(float) * totalThreadsPerGPU, cudaMemAdviseSetAccessedBy, deviceLoc);
                cudaMemAdvise(p + offset_p, sizeof(float) * totalThreadsPerGPU, cudaMemAdviseSetAccessedBy, deviceLoc);
                cudaMemAdvise(x + offset_x, sizeof(float) * totalThreadsPerGPU, cudaMemAdviseSetAccessedBy, deviceLoc);

                offset_Ax += totalThreadsPerGPU * numGpusToUse;
                offset_r += totalThreadsPerGPU * numGpusToUse;
                offset_p += totalThreadsPerGPU * numGpusToUse;
                offset_x += totalThreadsPerGPU * numGpusToUse;

                if (offset_Ax >= N) {
                    break;
                }
            }
        }

        device_count++;
        deviceId++;
    }

#if ENABLE_CPU_DEBUG_CODE
    float *Ax_cpu = (float *)malloc(sizeof(float) * N);
    float *r_cpu  = (float *)malloc(sizeof(float) * N);
    float *p_cpu  = (float *)malloc(sizeof(float) * N);
    float *x_cpu  = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        r_cpu[i]  = 1.0;
        Ax_cpu[i] = x_cpu[i] = 0.0;
    }
#endif

    printf("Total threads per GPU = %d numBlocksPerSm  = %d\n",
           numSms * numBlocksPerSm * THREADS_PER_BLOCK,
           numBlocksPerSm);
    dim3 dimGrid(numSms * numBlocksPerSm, 1, 1), dimBlock(THREADS_PER_BLOCK, 1, 1);

    // Structure used for cross-grid synchronization.
    MultiDeviceData multi_device_data;
    checkCudaErrors(cudaHostAlloc(&multi_device_data.hostMemoryArrivedList,
                                  (numGpusToUse - 1) * sizeof(*multi_device_data.hostMemoryArrivedList),
                                  cudaHostAllocPortable));
    memset(multi_device_data.hostMemoryArrivedList,
           0,
           (numGpusToUse - 1) * sizeof(*multi_device_data.hostMemoryArrivedList));
    multi_device_data.numDevices = numGpusToUse;
    multi_device_data.deviceRank = 0;

    void *kernelArgs[] = {
        (void *)&I,
        (void *)&J,
        (void *)&val,
        (void *)&x,
        (void *)&Ax,
        (void *)&p,
        (void *)&r,
        (void *)&dot_result,
        (void *)&nz,
        (void *)&N,
        (void *)&max_iter,
        (void *)&multi_device_data,
    };

    printf("Launching kernel\n");

    deviceId     = bestFitDeviceIds.begin();
    device_count = 0;
    while (deviceId != bestFitDeviceIds.end()) {
        checkCudaErrors(cudaSetDevice(*deviceId));
        checkCudaErrors(cudaLaunchCooperativeKernel(
            (void *)multiGpuConjugateGradient, dimGrid, dimBlock, kernelArgs, sMemSize, nStreams[device_count++]));
        multi_device_data.deviceRank++;
        deviceId++;
    }

    // Use cudaMemLocationTypeHost for optimal host memory location
    cudaMemLocation hostLoc;
    hostLoc.type = cudaMemLocationTypeHost;
    checkCudaErrors(cudaMemPrefetchAsync(x, sizeof(float) * N, hostLoc, 0));
    checkCudaErrors(cudaMemPrefetchAsync(dot_result, sizeof(double), hostLoc, 0));

    deviceId     = bestFitDeviceIds.begin();
    device_count = 0;
    while (deviceId != bestFitDeviceIds.end()) {
        checkCudaErrors(cudaSetDevice(*deviceId));
        checkCudaErrors(cudaStreamSynchronize(nStreams[device_count++]));
        deviceId++;
    }

    r1 = (float)*dot_result;

    const float finalResidual = sqrt(r1);
    printf("GPU Final, residual = %e\n", finalResidual);
    printf("Completed fixed iterations = %d\n", max_iter);

#if ENABLE_CPU_DEBUG_CODE
    cpuConjugateGrad(I, J, val, x_cpu, Ax_cpu, p_cpu, r_cpu, nz, N, max_iter);
#endif

    float rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++) {
        rsum = 0.0;

        for (int j = I[i]; j < I[i + 1]; j++) {
            rsum += val_cpu[j] * x[J[j]];
        }

        diff = fabs(rsum - rhs);

        if (diff > err) {
            err = diff;
        }
    }

    checkCudaErrors(cudaFreeHost(multi_device_data.hostMemoryArrivedList));
    checkCudaErrors(cudaFree(I));
    checkCudaErrors(cudaFree(J));
    checkCudaErrors(cudaFree(val));
    checkCudaErrors(cudaFree(x));
    checkCudaErrors(cudaFree(r));
    checkCudaErrors(cudaFree(p));
    checkCudaErrors(cudaFree(Ax));
    checkCudaErrors(cudaFree(dot_result));
    free(val_cpu);

#if ENABLE_CPU_DEBUG_CODE
    free(Ax_cpu);
    free(r_cpu);
    free(p_cpu);
    free(x_cpu);
#endif

    printf("Test Summary:  Error amount = %f \n", err);
    if (finalResidual >= tol) {
        printf("Convergence check ignored for exit status; residual exceeds tolerance %e after %d iterations.\n",
               tol,
               max_iter);
    }
    fprintf(stdout, "&&&& conjugateGradientMultiDeviceCG PASSED\n");
    return EXIT_SUCCESS;
}
