#pragma once

#include <CL/sycl.hpp>

using namespace cl::sycl;

using SyclBuffer = buffer<double, 1>;

template <typename T, int N, typename BinaryOp> inline auto reduction_shim(buffer<T, N> &b, sycl::handler &h, T init, BinaryOp f) {
#if defined(__HIPSYCL__) || defined(__OPENSYCL__)
  return sycl::reduction(b. template get_access<access_mode::read_write>(h),  init, f);
#else
  return sycl::reduction(b, h, init, f, sycl::property::reduction::initialize_to_identity());
#endif
}