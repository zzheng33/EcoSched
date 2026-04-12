#include "chunk.h"
#include "kokkos_shared.hpp"
#include "shared.h"

// Updates the local left halo region(s)
void update_left(const int x, const int y, const int depth, const int halo_depth, KView &buffer) {
  Kokkos::parallel_for(
      y * depth, KOKKOS_LAMBDA(const int index) {
        const int flip = index % depth;
        const int lines = index / depth;
        const int offset = lines * (x - depth);
        const int to_index = offset + halo_depth - depth + index;
        const int from_index = to_index + 2 * (depth - flip) - 1;
        buffer(to_index) = buffer(from_index);
      });
}

// Updates the local right halo region(s)
void update_right(const int x, const int y, const int depth, const int halo_depth, KView &buffer) {
  Kokkos::parallel_for(
      y * depth, KOKKOS_LAMBDA(const int index) {
        const int flip = index % depth;
        const int lines = index / depth;
        const int offset = x - halo_depth + lines * (x - depth);
        const int to_index = offset + index;
        const int from_index = to_index - (1 + flip * 2);
        buffer(to_index) = buffer(from_index);
      });
}

// Updates the local top halo region(s)
void update_top(const int x, const int y, const int depth, const int halo_depth, KView &buffer) {
  Kokkos::parallel_for(
      x * depth, KOKKOS_LAMBDA(const int index) {
        const int lines = index / x;
        const int offset = x * (y - halo_depth);

        const int to_index = offset + index;
        const int from_index = to_index - (1 + lines * 2) * x;
        buffer(to_index) = buffer(from_index);
      });
}

// Updates the local bottom halo region(s)
void update_bottom(const int x, const int y, const int depth, const int halo_depth, KView &buffer) {
  Kokkos::parallel_for(
      x * depth, KOKKOS_LAMBDA(const int index) {
        const int lines = index / x;
        const int offset = x * halo_depth;

        const int from_index = offset + index;
        const int to_index = from_index - (1 + lines * 2) * x;
        buffer(to_index) = buffer(from_index);
      });
}

// Updates faces in turn.
void update_face(const int x, const int y, const int halo_depth, const int *chunk_neighbours, const int depth, KView &buffer) {
  if (chunk_neighbours[CHUNK_LEFT] == EXTERNAL_FACE) {
    update_left(x, y, depth, halo_depth, buffer);
  }
  if (chunk_neighbours[CHUNK_RIGHT] == EXTERNAL_FACE) {
    update_right(x, y, depth, halo_depth, buffer);
  }
  if (chunk_neighbours[CHUNK_TOP] == EXTERNAL_FACE) {
    update_top(x, y, depth, halo_depth, buffer);
  }
  if (chunk_neighbours[CHUNK_BOTTOM] == EXTERNAL_FACE) {
    update_bottom(x, y, depth, halo_depth, buffer);
  }
}

// The kernel for updating halos locally
void local_halos(const int x, const int y, const int depth, const int halo_depth, const int *chunk_neighbours,
                 const bool *fields_to_exchange, KView &density, KView &energy0, KView &energy, KView &u, KView &p, KView &sd) {
  if (fields_to_exchange[FIELD_DENSITY]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, density);
  }
  if (fields_to_exchange[FIELD_P]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, p);
  }
  if (fields_to_exchange[FIELD_ENERGY0]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, energy0);
  }
  if (fields_to_exchange[FIELD_ENERGY1]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, energy);
  }
  if (fields_to_exchange[FIELD_U]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, u);
  }
  if (fields_to_exchange[FIELD_SD]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, sd);
  }
}

// Solver-wide kernels
void run_local_halos(Chunk *chunk, Settings &settings, int depth) {
  START_PROFILING(settings.kernel_profile);
  local_halos(chunk->x, chunk->y, depth, settings.halo_depth, chunk->neighbours, settings.fields_to_exchange, *chunk->density,
              *chunk->energy0, *chunk->energy, *chunk->u, *chunk->p, *chunk->sd);
  STOP_PROFILING(settings.kernel_profile, __func__);
}
