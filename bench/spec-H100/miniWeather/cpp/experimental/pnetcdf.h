// Stub pnetcdf.h to allow compilation without PNetCDF library
// I/O operations will be disabled

#ifndef PNETCDF_STUB_H
#define PNETCDF_STUB_H

#include <mpi.h>

#define NC_CLOBBER 0
#define NC_UNLIMITED 0
#define NC_DOUBLE 0
#define NC_NOERR 0
#define NC_WRITE 1

// Use MPI_Offset from mpi.h (already defined)

inline int ncmpi_create(MPI_Comm comm, const char* path, int mode, MPI_Info info, int* ncid) { *ncid = 0; return 0; }
inline int ncmpi_def_dim(int ncid, const char* name, MPI_Offset len, int* dimid) { *dimid = 0; return 0; }
inline int ncmpi_def_var(int ncid, const char* name, int type, int ndims, const int* dimids, int* varid) { *varid = 0; return 0; }
inline int ncmpi_enddef(int ncid) { return 0; }
inline int ncmpi_put_vara_double_all(int ncid, int varid, const MPI_Offset* start, const MPI_Offset* count, const double* data) { return 0; }
inline int ncmpi_close(int ncid) { return 0; }
inline int ncmpi_open(MPI_Comm comm, const char* path, int mode, MPI_Info info, int* ncid) { *ncid = 0; return 0; }
inline int ncmpi_inq_varid(int ncid, const char* name, int* varid) { *varid = 0; return 0; }
inline int ncmpi_begin_indep_data(int ncid) { return 0; }
inline int ncmpi_put_vara_double(int ncid, int varid, const MPI_Offset* start, const MPI_Offset* count, const double* data) { return 0; }
inline int ncmpi_end_indep_data(int ncid) { return 0; }
inline const char* ncmpi_strerror(int err) { return "PNetCDF stub - no error"; }

#endif
