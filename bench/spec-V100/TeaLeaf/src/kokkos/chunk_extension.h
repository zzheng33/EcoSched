#pragma once

#include "kokkos_shared.hpp"
#include <Kokkos_Core.hpp>

using FieldBufferType = Kokkos::View<double *> *;
using StagingBufferType = Kokkos::View<double *>::HostMirror *;
struct ChunkExtension {};
