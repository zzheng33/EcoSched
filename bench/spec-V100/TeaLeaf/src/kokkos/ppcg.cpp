#include "chunk.h"
#include "kokkos_shared.hpp"
#include "shared.h"

// Initialises Sd
void ppcg_init(const int x, const int y, const int halo_depth, const double theta, KView &sd, KView &r) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          sd[index] = r[index] / theta;
        }
      });
}

// Calculates U and R
void ppcg_calc_ur(const int x, const int y, const int halo_depth, KView &sd, KView &r, KView &u, KView &kx, KView &ky) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          const double smvp = tealeaf_SMVP(sd);
          r[index] -= smvp;
          u[index] += sd[index];
        }
      });
}

// Calculates Sd
void ppcg_calc_sd(const int x, const int y, const int halo_depth, const double theta, const double alpha, const double beta, KView &sd,
                  KView &r) {
  Kokkos::parallel_for(
      x * y, KOKKOS_LAMBDA(const int index) {
        const int kk = index % x;
        const int jj = index / x;

        if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
          sd[index] = alpha * sd[index] + beta * r[index];
        }
      });
}

// PPCG solver kernels
void run_ppcg_init(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);

  ppcg_init(chunk->x, chunk->y, settings.halo_depth, chunk->theta, *chunk->sd, *chunk->r);

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_ppcg_inner_iteration(Chunk *chunk, Settings &settings, double alpha, double beta) {
  START_PROFILING(settings.kernel_profile);

  ppcg_calc_ur(chunk->x, chunk->y, settings.halo_depth, *chunk->sd, *chunk->r, *chunk->u, *chunk->kx, *chunk->ky);

  ppcg_calc_sd(chunk->x, chunk->y, settings.halo_depth, chunk->theta, alpha, beta, *chunk->sd, *chunk->r);

  STOP_PROFILING(settings.kernel_profile, __func__);
}