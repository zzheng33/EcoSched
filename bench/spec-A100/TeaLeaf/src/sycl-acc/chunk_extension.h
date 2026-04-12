#pragma once

#include <CL/sycl.hpp>

using namespace cl;

using FieldBufferType = sycl::buffer<double, 1> *;
using StagingBufferType = sycl::buffer<double, 1> *;

struct ChunkExtension {
  sycl::queue *device_queue;
};
