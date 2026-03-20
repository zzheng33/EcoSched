#pragma once

#include <CL/sycl.hpp>

using namespace cl::sycl;

using SyclBuffer = double *;

template <typename T, typename BinaryOp> inline auto reduction_shim(T *b, T init, BinaryOp f) {
#if defined(__HIPSYCL__) || defined(__OPENSYCL__)
  return sycl::reduction(b, init, f);
#else
  return sycl::reduction(b, init, f, sycl::property::reduction::initialize_to_identity());
#endif
}
