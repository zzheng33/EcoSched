#include "chunk.h"
#include "kokkos_shared.hpp"
#include "shared.h"

// Initialises the Chebyshev solver
void cheby_init(const int x, const int y, const int halo_depth, const double theta, KView &p, KView &r, KView &u, KView &u0, KView &w,
                KView &kx, KView &ky) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          const double smvp = tealeaf_SMVP(u);
          w[index] = smvp;
          r[index] = u0[index] - w[index];
          p[index] = r[index] / theta;
        }
      });
}

// Calculates U
void cheby_calc_u(const int x, const int y, const int halo_depth, KView &p, KView &u) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          u[index] += p[index];
        }
      });
}

// The main Cheby iteration step
void cheby_iterate(const int x, const int y, const int halo_depth, const double alpha, const double beta, KView &p, KView &r, KView &u,
                   KView &u0, KView &w, KView &kx, KView &ky) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          const double smvp = tealeaf_SMVP(u);
          w[index] = smvp;
          r[index] = u0[index] - w[index];
          p[index] = alpha * p[index] + beta * r[index];
        }
      });
}

// Chebyshev solver kernels
void run_cheby_init(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);

  cheby_init(chunk->x, chunk->y, settings.halo_depth, chunk->theta, *chunk->p, *chunk->r, *chunk->u, *chunk->u0, *chunk->w, *chunk->kx,
             *chunk->ky);

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_cheby_iterate(Chunk *chunk, Settings &settings, double alpha, double beta) {
  START_PROFILING(settings.kernel_profile);

  cheby_iterate(chunk->x, chunk->y, settings.halo_depth, alpha, beta, *chunk->p, *chunk->r, *chunk->u, *chunk->u0, *chunk->w, *chunk->kx,
                *chunk->ky);

  cheby_calc_u(chunk->x, chunk->y, settings.halo_depth, *chunk->p, *chunk->u);

  STOP_PROFILING(settings.kernel_profile, __func__);
}