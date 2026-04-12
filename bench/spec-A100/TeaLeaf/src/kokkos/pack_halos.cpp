#include "chunk.h"
#include "comms.h"
#include "kokkos_shared.hpp"
#include "shared.h"

// Packs the top halo buffer(s)
void pack_top(const int x, const int y, const int depth, const int halo_depth, KView &field, KView &buffer, int buffer_offset) {
  Kokkos::parallel_for(
      x * depth, KOKKOS_LAMBDA(const int index) {
        const int offset = x * (y - halo_depth - depth);
        buffer(index + buffer_offset) = field(offset + index);
      });
}

// Packs the bottom halo buffer(s)
void pack_bottom(const int x, const int y, const int depth, const int halo_depth, KView &field, KView &buffer, int buffer_offset) {
  Kokkos::parallel_for(
      x * depth, KOKKOS_LAMBDA(const int index) {
        const int offset = x * halo_depth;
        buffer(index + buffer_offset) = field(offset + index);
      });
}

// Packs the left halo buffer(s)
void pack_left(const int x, const int y, const int depth, const int halo_depth, KView &field, KView &buffer, int buffer_offset) {
  Kokkos::parallel_for(
      y * depth, KOKKOS_LAMBDA(const int index) {
        const int lines = index / depth;
        const int offset = halo_depth + lines * (x - depth);
        buffer(index + buffer_offset) = field(offset + index);
      });
}

// Packs the right halo buffer(s)
void pack_right(const int x, const int y, const int depth, const int halo_depth, KView &field, KView &buffer, int buffer_offset) {
  Kokkos::parallel_for(
      y * depth, KOKKOS_LAMBDA(const int index) {
        const int lines = index / depth;
        const int offset = x - halo_depth - depth + lines * (x - depth);
        buffer(index + buffer_offset) = field(offset + index);
      });
}

// Unpacks the top halo buffer(s)
void unpack_top(const int x, const int y, const int depth, const int halo_depth, KView &field, KView &buffer, int buffer_offset) {
  Kokkos::parallel_for(
      x * depth, KOKKOS_LAMBDA(const int index) {
        const int offset = x * (y - halo_depth);
        field(offset + index) = buffer(index + buffer_offset);
      });
}

// Unpacks the bottom halo buffer(s)
void unpack_bottom(const int x, const int y, const int depth, const int halo_depth, KView &field, KView &buffer, int buffer_offset) {
  Kokkos::parallel_for(
      x * depth, KOKKOS_LAMBDA(const int index) {
        const int offset = x * (halo_depth - depth);
        field(offset + index) = buffer(index + buffer_offset);
      });
}

// Unpacks the left halo buffer(s)
void unpack_left(const int x, const int y, const int depth, const int halo_depth, KView &field, KView &buffer, int buffer_offset) {
  Kokkos::parallel_for(
      y * depth, KOKKOS_LAMBDA(const int index) {
        const int lines = index / depth;
        const int offset = halo_depth - depth + lines * (x - depth);
        field(offset + index) = buffer(index + buffer_offset);
      });
}

// Unpacks the right halo buffer(s)
void unpack_right(const int x, const int y, const int depth, const int halo_depth, KView &field, KView &buffer, int buffer_offset) {
  Kokkos::parallel_for(
      y * depth, KOKKOS_LAMBDA(const int index) {
        const int lines = index / depth;
        const int offset = x - halo_depth + lines * (x - depth);
        field(offset + index) = buffer(index + buffer_offset);
      });
}

void run_pack_or_unpack(Chunk *chunk, Settings &settings, int depth, int face, bool pack, KView *field, KView *buffer, int offset) {
  START_PROFILING(settings.kernel_profile);
  switch (face) {
    case CHUNK_LEFT:
      if (pack) pack_left(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset);
      else
        unpack_left(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset);
      break;
    case CHUNK_RIGHT:
      if (pack) pack_right(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset);
      else
        unpack_right(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset);
      break;
    case CHUNK_TOP:
      if (pack) pack_top(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset);
      else
        unpack_top(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset);
      break;
    case CHUNK_BOTTOM:
      if (pack) pack_bottom(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset);
      else
        unpack_bottom(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset);
      break;
    default: die(__LINE__, __FILE__, "Incorrect face provided: %d.\n", face);
  }
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_send_recv_halo(Chunk *chunk, Settings &settings,                                                       //
                        FieldBufferType src_send_buffer, FieldBufferType src_recv_buffer,                       //
                        StagingBufferType dest_staging_send_buffer, StagingBufferType dest_staging_recv_buffer, //
                        int buffer_len, int neighbour,                                                          //
                        int send_tag, int recv_tag,                                                             //
                        MPI_Request *send_request, MPI_Request *recv_request) {

  *dest_staging_send_buffer = Kokkos::create_mirror_view(*src_send_buffer);
  *dest_staging_recv_buffer = Kokkos::create_mirror_view(*src_recv_buffer);

  if (!settings.staging_buffer) Kokkos::fence();
  else
    Kokkos::deep_copy(*dest_staging_send_buffer, *src_send_buffer);

  send_recv_message(settings, //
                    settings.staging_buffer ? dest_staging_send_buffer->data() : src_send_buffer->data(),
                    settings.staging_buffer ? dest_staging_recv_buffer->data() : src_recv_buffer->data(), buffer_len, neighbour, send_tag,
                    recv_tag, send_request, recv_request);
}

void run_before_waitall_halo(Chunk *, Settings &) {}

void run_restore_recv_halo(Chunk *, Settings &settings, //
                           FieldBufferType dest_recv_buffer, StagingBufferType src_staging_recv_buffer, int buffer_len) {
  if (settings.staging_buffer) {
    Kokkos::deep_copy(*dest_recv_buffer, *src_staging_recv_buffer);
  }
}