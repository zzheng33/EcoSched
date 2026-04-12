/*****************************************************************************
* 
* Copyright (c) 2018, University of Ferrara, University of Rome and INFN. All rights reserved.
* 
* 
* Authors listed in alphabetic order:
* ----------------------------------
* Luca Biferale (University of Rome "Tor Vergata" and INFN)
* Enrico Calore (University of Ferrara and INFN)
* Alessandro Gabbana (University of Ferrara and INFN)
* Mauro Sbragaglia (University of Rome "Tor Vergata" and INFN)
* Andrea Scagliarini (CNR-IAC, Roma)
* Sebastiano Fabio Schifano (University of Ferrara and INFN)(*)
* Raffaele Tripiccione (University of Ferrara and INFN)
* 
* (*) corresponding author, sebastiano.schifano __at__ unife.it
* 
* We also thanks the following students:
* -------------------------------------
* Elisa Pellegrini (University of Ferrara)
* Marco Zanella (University of Ferrara)
* 
* 
* Relevant pubblications:
* ----------------------
* 
* - Sbragaglia, M., Benzi, R., Biferale, L., Chen, H., Shan, X., Succi, S.
*   Lattice Boltzmann method with self-consistent thermo-hydrodynamic equilibria
*   (2009) Journal of Fluid Mechanics, 628, pp. 299-309. Cited 66 times.
*   https://www.scopus.com/inward/record.uri?eid=2-s2.0-67651027859&doi=10.1017%2fS002211200900665X&partnerID=40&md5=1c57f2de4c0abc5aacaf5e30e9d566d7
*   DOI: 10.1017/S002211200900665X
* 
* - Scagliarini, A., Biferale, L., Sbragaglia, M., Sugiyama, K., Toschi, F.
*   Lattice Boltzmann methods for thermal flows: Continuum limit and applications to compressible Rayleigh-Taylor systems
*   (2010) Physics of Fluids, 22 (5), art. no. 019004PHF, pp. 1-21. Cited 51 times.
*   https://www.scopus.com/inward/record.uri?eid=2-s2.0-77955618379&doi=10.1063%2f1.3392774&partnerID=40&md5=8b47e1b6c7ffd95583e2dc30d36a66f2
*   DOI: 10.1063/1.3392774
* 
* - Calore, E., Gabbana, A., Kraus, J., Pellegrini, E., Schifano, S.F., Tripiccione, R.
*   Massively parallel lattice–Boltzmann codes on large GPU clusters
*   (2016) Parallel Computing, 58, pp. 1-24.
*   https://www.scopus.com/inward/record.uri?eid=2-s2.0-84983616316&doi=10.1016%2fj.parco.2016.08.005&partnerID=40&md5=a9b40226e16f6e24bc8bc7138b789f90
*   DOI: 10.1016/j.parco.2016.08.005
* 
* - Calore, E., Gabbana, A., Kraus, J., Schifano, S.F., Tripiccione, R.
*   Performance and portability of accelerated lattice Boltzmann applications with OpenACC
*   (2016) Concurrency Computation, 28 (12), pp. 3485-3502.
*   https://www.scopus.com/inward/record.uri?eid=2-s2.0-84971215878&doi=10.1002%2fcpe.3862&partnerID=40&md5=fe286342abec002fa287da5967b9d2c2
*   DOI: 10.1002/cpe.3862
* 
* 
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
* 
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
* 
*   * Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
* 
*   * Neither the name of the above Institutions nor the names of their
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
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
* 
*******************************************************************************/

#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "common.h"

///////////////////////////////////////////////////////////////////////////////
// extern routines

///////////////////////////////////////////////////////////////////////////////
// extern variables

extern int mpi_rank;          // Rank of MPI node.

extern int n_mpi;             // total number of MPI ranks == n_mpi_x * n_mpi_y == mpi_size

extern int n_mpi_x;           // number of MPI ranks along X direction
extern int n_mpi_y;           // number of MPI ranks alonf Y direction

extern int mpi_rank_x;        // MPI X coordinate.
extern int mpi_rank_y;        // MPI Y coordinate.

extern int mpi_rank;          // Rank of MPI node.
extern int mpi_size;          // Size of MPI world communicator.

extern int mpi_peer_left, mpi_peer_right, mpi_peer_top, mpi_peer_bot; // Rank of neighboring MPI ranks

extern char * mpi_order;      // MPI ranks order string

extern int GSIZEX, GSIZEY;    // global lattice sizex and sizex
extern int LSIZEX, LSIZEY;    // local lattice sizex and sizex
extern int NX, NY;	          // x-length and y-lenght of lattice in memory.

extern long int SEED;                // seed for random generator

extern uvrt_t *uvrt;
extern par_t  *param;

///////////////////////////////////////////////////////////////////////////////
// Cuda constants.

__constant__ data_t param_ww[NPOP];
__constant__ data_t param_cx[NPOP];
__constant__ data_t param_cy[NPOP];
__constant__ data_t param_H0[NPOP];
__constant__ data_t param_H1x[NPOP];
__constant__ data_t param_H1y[NPOP];
__constant__ data_t param_H2xx[NPOP];
__constant__ data_t param_H2xy[NPOP];
__constant__ data_t param_H2yy[NPOP];
__constant__ data_t param_H3xxx[NPOP];
__constant__ data_t param_H3xxy[NPOP];
__constant__ data_t param_H3xyy[NPOP];
__constant__ data_t param_H3yyy[NPOP];
__constant__ data_t param_H4xxxx[NPOP];
__constant__ data_t param_H4xxxy[NPOP];
__constant__ data_t param_H4xxyy[NPOP];
__constant__ data_t param_H4xyyy[NPOP];
__constant__ data_t param_H4yyyy[NPOP];


__constant__ int GSIZEY_d;
__constant__ int LSIZEX_d;
__constant__ int LSIZEY_d;
__constant__ int NX_d;
__constant__ int NY_d;


///////////////////////////////////////////////////////////////////////////////

#include "utils.h"

#define PROPAGATE_COLLIDE_BULK_THRBLK 256
#define PROPAGATE_COLLIDE_LR_THRBLK 256
#define PROPAGATE_COLLIDE_TB_THRBLK 32
#define BC_BLOCK_DIMY 32
#define PACK_TB_THBLK 32
#define PACK_LR_THBLK 32

#include "propagate_kernel.cuh"
#include "bc_kernel.cuh"
#include "collide_kernel.cuh"
#include "propagate_collide_kernel.cuh"

///////////////////////////////////////////////////////////////////////////////
// Check CUDA error.

void CUDACheckError(const char* action){
  cudaError_t error;

  error = cudaGetLastError();
  if(error != cudaSuccess){
    fprintf(
      stderr, "ERROR: Error while '%s': %s\n",
      action,
      cudaGetErrorString(error)
    );
    exit(-1);
  }
}

///////////////////////////////////////////////////////////////////////////////
// A CUDA function is called and error is checked.

#ifdef USE_DEBUG
#define CUDA_CALL( call )       \
  call;                         \
  CUDACheckError( #call );
#else
#define CUDA_CALL( call )       \
  call;
#endif

///////////////////////////////////////////////////////////////////////////////

cudaStream_t stream[5];

///////////////////////////////////////////////////////////////////////////////

#include "pbc.cuh"

///////////////////////////////////////////////////////////////////////////////

// contants used by propagate_collide, propagate and collide
#define BULK_OFF  ((HY+hy+1)        + NY*(HX+hx))        // Offset to address the bulk.
#define LC_OFF    ((HY+hy+1)        + NY*HX)             // Offset to address left border.
#define RC_OFF    ((HY+hy+1)        + NY*(HX+LSIZEX-hx)) // Offset to address right border.
#define T_OFF     ((HY+LSIZEY-hy-1) + NY*HX)             // Offset to address top border.
#define B_OFF     (HY               + NY*HX)             // Offset to address bottom border.

// contant used by bc
#define BC_OFF    (HY               + NY*HX)             // Offset to address lattice for bc.

///////////////////////////////////////////////////////////////////////////////
// Function used to launch kernels.

extern "C" void lbm(int niter, pop_t *f_aos_h, par_t *param) {
  int ii, yoff;
  data_t *f2_soa_h;
  data_t *f1_soa_d, *f2_soa_d, *f_tmp_d;
  data_t mass, i_mass, g_mass;

//  double mpimin_dt_tot, mpimax_dt_tot;
//  double mpimax_dt_swap_nc, mpimin_dt_swap_nc, double mpimax_dt_swap_c, mpimin_dt_swap_c;

#ifndef SPEC
  double mpimax_dt_tot, mpimax_dt_swap_nc, mpimax_dt_swap_c;
  struct timeval ti[5], tf[5];
  double dT[5];
  //double latticesize; //, bw, p;
  double mlups;
#endif
#ifdef SPEC
  struct timeval ti_batch, tf_batch;
  double batch_time;
#endif
  cudaEvent_t unpackLEndEvent, unpackREndEvent;

  // Dimension x of grid for computing propagate on bulk and left and right borders.
  // If LSIZEY-6 is not divisible by PROPAGATE_COLLIDE_BULK_THRBLK, is added 1 to the dimension X of grid.
  int gridBulkDimX=(((LSIZEY-2*(hy+1))%PROPAGATE_COLLIDE_BULK_THRBLK)==0)?((LSIZEY-2*(hy+1))/PROPAGATE_COLLIDE_BULK_THRBLK):(((LSIZEY-2*(hy+1))/PROPAGATE_COLLIDE_BULK_THRBLK)+1);

  // Dimension of grid and blocks for computing propagate on the bulk.
  dim3 gridBulk  (gridBulkDimX, (LSIZEX - (2*hx)), 1);
  dim3 blockBulk (PROPAGATE_COLLIDE_BULK_THRBLK, 1, 1);

  // Dimension x of grid for computing propagate on bulk and left and right borders.
  // If LSIZEY-6 is not divisible by PROPAGATE_COLLIDE_LR_THRBLK, is added 1 to the dimension X of grid.
  int gridLRDimX=(((LSIZEY-2*(hy+1))%PROPAGATE_COLLIDE_LR_THRBLK)==0)?((LSIZEY-2*(hy+1))/PROPAGATE_COLLIDE_LR_THRBLK):(((LSIZEY-2*(hy+1))/PROPAGATE_COLLIDE_LR_THRBLK)+1);
  
  // Dimension of grid and blocks for computing propagate on left and right borders.
  dim3 gridLR    (gridLRDimX, hx, 1);
  dim3 blockLR   (PROPAGATE_COLLIDE_LR_THRBLK, 1, 1);

  // Dimension x of grid for computing contiguous swap on left and right borders.
  // If NY is not divisible by PACK_LR_THBLK, is added 1 to the dimension X of grid.
  int gridSwapDimX=(((LSIZEY+2*hy)%PACK_LR_THBLK)==0)?((LSIZEY+2*hy)/PACK_LR_THBLK):(((LSIZEY+2*hy)/PACK_LR_THBLK)+1);
  
  // Dimension of grid and blocks for computing propagate on left and right borders.
  dim3 gridSwapLR    (gridSwapDimX, hx, 1);
  dim3 blockSwapLR   (PACK_LR_THBLK, 1, 1);

  // Dimension y of grid for computing propagate on top and bottom borders.
  // If the dimension x of bulk is not divisible by PROPAGATE_COLLIDE_TB_THRBLK, is added 1 to the dimension y of grid.
  int gridTB_DimY = (LSIZEX%PROPAGATE_COLLIDE_TB_THRBLK == 0) ? (LSIZEX/PROPAGATE_COLLIDE_TB_THRBLK) : (LSIZEX/PROPAGATE_COLLIDE_TB_THRBLK+1);

  // Dimension of grid and blocks for computing propagate on top and bottom borders.
  dim3 gridTB ( 1,  gridTB_DimY, 1 );
  dim3 blockTB ( hy+1, PROPAGATE_COLLIDE_TB_THRBLK, 1 );

  // Dimension y of grid for computing bc on top and bottom borders.
  // If the dimension x of bulk is not divisible by BC_BLOCK_DIMY, is added 1 to the dimension y of grid.
  int bcGridDimY=(((LSIZEX)%BC_BLOCK_DIMY)==0)?((LSIZEX)/BC_BLOCK_DIMY):((LSIZEX)/BC_BLOCK_DIMY)+1;

  // Dimension of grid and blocks for computing bc on top and bottom borders.
  dim3 bcGridTB ( 1, bcGridDimY, 1 );
  dim3 bcBlockTB( hy, BC_BLOCK_DIMY, 1);

  // Params are copied into GPU constant memory.
  CUDA_CALL( cudaMemcpyToSymbol ( param_ww    , param->ww    , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_cx    , param->cx    , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_cy    , param->cy    , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H0    , param->H0    , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H1x   , param->H1x   , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H1y   , param->H1y   , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H2xx  , param->H2xx  , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H2xy  , param->H2xy  , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H2yy  , param->H2yy  , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H3xxx , param->H3xxx , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H3xxy , param->H3xxy , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H3xyy , param->H3xyy , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H3yyy , param->H3yyy , NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H4xxxx, param->H4xxxx, NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H4xxxy, param->H4xxxy, NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H4xxyy, param->H4xxyy, NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H4xyyy, param->H4xyyy, NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( param_H4yyyy, param->H4yyyy, NPOP*sizeof(data_t)  , 0, cudaMemcpyHostToDevice ) );

  //fprintf(stderr, "[MPI%04d]: NX: %ld, NY: %ld, LSIZEX: %ld, LSIZEY: %ld \n", mpi_rank, NX, NY, LSIZEX, LSIZEY);

  CUDA_CALL( cudaMemcpyToSymbol ( GSIZEY_d, &GSIZEY, sizeof(int), 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( LSIZEX_d, &LSIZEX, sizeof(int), 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( LSIZEY_d, &LSIZEY, sizeof(int), 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( NX_d,     &NX,     sizeof(int), 0, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemcpyToSymbol ( NY_d,     &NY,     sizeof(int), 0, cudaMemcpyHostToDevice ) );
  
  // GPU cache is configured.
  CUDA_CALL( cudaFuncSetCacheConfig(propagateCollideBulk, cudaFuncCachePreferL1) );
  CUDA_CALL( cudaFuncSetCacheConfig(propagateCollideL, cudaFuncCachePreferL1) );
  CUDA_CALL( cudaFuncSetCacheConfig(propagateCollideR, cudaFuncCachePreferL1) );
  CUDA_CALL( cudaFuncSetCacheConfig(propagateT, cudaFuncCachePreferL1) );
  CUDA_CALL( cudaFuncSetCacheConfig(propagateB, cudaFuncCachePreferL1) );
  CUDA_CALL( cudaFuncSetCacheConfig(collideT, cudaFuncCachePreferL1) );
  CUDA_CALL( cudaFuncSetCacheConfig(collideB, cudaFuncCachePreferL1) );
  CUDA_CALL( cudaFuncSetCacheConfig(bcT, cudaFuncCachePreferL1) );
  CUDA_CALL( cudaFuncSetCacheConfig(bcB, cudaFuncCachePreferL1) );

  
  // Lattices are allocated on GPU.
  CUDA_CALL( cudaMalloc((void **) &f1_soa_d, NX*NY*NPOP*sizeof(data_t)) );
  CUDA_CALL( cudaMalloc((void **) &f2_soa_d, NX*NY*NPOP*sizeof(data_t)) );
  
  // Lattice is converted and copied to the GPU.
  CUDA_CALL( cudaHostAlloc((void **) &f2_soa_h, NX*NY*NPOP*sizeof(data_t), cudaHostAllocDefault) );
  
  AoStoSoA(f_aos_h, f2_soa_h);
  
  CUDA_CALL( cudaMemcpy(f1_soa_d, f2_soa_h, NX*NY*NPOP*sizeof(data_t), cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(f2_soa_d, f2_soa_h, NX*NY*NPOP*sizeof(data_t), cudaMemcpyHostToDevice) );
  
  // Streams are created.
  CUDA_CALL( cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking) );
  CUDA_CALL( cudaStreamCreateWithFlags(&stream[1], cudaStreamNonBlocking) );
  CUDA_CALL( cudaStreamCreateWithFlags(&stream[2], cudaStreamNonBlocking) );
  CUDA_CALL( cudaStreamCreateWithFlags(&stream[3], cudaStreamNonBlocking) );
  CUDA_CALL( cudaStreamCreateWithFlags(&stream[4], cudaStreamNonBlocking) );

  // create events
  CUDA_CALL( cudaEventCreate (&unpackLEndEvent) );
  CUDA_CALL( cudaEventCreate (&unpackREndEvent) );

  // Init PBC parameters
  initPbc();

#ifndef SPEC
  // Delta-times are initialized.
  dT[0] = 0.0;
  dT[1] = 0.0;
  dT[2] = 0.0;
  dT[3] = 0.0;
  dT[4] = 0.0;
#endif 

  yoff = mpi_rank_y * (GSIZEY / n_mpi_y);  

  #ifdef CHECK_MASS
  mass = dumpMass(f_aos_h);
  MPI_Reduce(&mass, &i_mass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  #endif
  
  MPI_Barrier(MPI_COMM_WORLD);

#ifndef SPEC
  gettimeofday(&ti[0], NULL);
#endif      

  // NITER iterations are performed.
  for(ii = 1; ii <= niter; ii++){

    ///////////////////////////////////////////////////////////////////////////

#ifndef SPEC
    gettimeofday(&ti[1], NULL);
#endif

#ifdef SPEC
      if (mpi_rank == 0) {
        if (ii == 1 || (ii - 1) % 500 == 0) {
          gettimeofday(&ti_batch, NULL);
          printf("Starting batch at iteration: %d\n", ii);
        }
      }
#endif

    // exchanging non contiguous borders (top and bottom): exchange of left- and right 
    // borders CANNOT start before this operation is fully completed.
    if(n_mpi_y > 1)
      pbc_nc( f2_soa_d );

#ifndef SPEC
    gettimeofday(&tf[1], NULL);
#endif
    ///////////////////////////////////////////////////////////////////////////

    // launching pack of right border using stream one
    CUDA_CALL( (pack_right <<< gridSwapLR, blockSwapLR, 0, stream[1] >>> (f2_soa_d + SRC_RGT_OFF, sndRgtBuf_d)) );

    // launching pack of left border using stream two
    CUDA_CALL( (pack_left <<< gridSwapLR, blockSwapLR, 0, stream[2] >>> (f2_soa_d + SRC_LFT_OFF, sndLftBuf_d)) );

    ///////////////////////////////////////////////////////////////////////////

    // launching propagate and collide over bulk using stream zero
    CUDA_CALL( (propagateCollideBulk <<< gridBulk, blockBulk, 0, stream[0] >>> (f1_soa_d + BULK_OFF, f2_soa_d + BULK_OFF, yoff)) );

    ///////////////////////////////////////////////////////////////////////////

    if (n_mpi_x == 1) {
      // Wait end of RIGHT Pack
      CUDA_CALL( cudaStreamSynchronize( stream[1] ) );

      // Wait end of LEFT Pack
      CUDA_CALL( cudaStreamSynchronize( stream[2] ) );

      //-- in this case rcvLftBuf_d = sndRgtBuf_d and rcvRgtBuf_d = sndLftBuf_d;
      CUDA_CALL( (unpack_left  <<< gridSwapLR, blockSwapLR, 0, stream[1] >>> (sndRgtBuf_d, f2_soa_d + DST_LFT_OFF )) );
      CUDA_CALL( (unpack_right <<< gridSwapLR, blockSwapLR, 0, stream[2] >>> (sndLftBuf_d, f2_soa_d + DST_RGT_OFF )) );
    } else {
#ifdef SPEC_ACCEL_AWARE_MPI
      // Wait end of RIGHT Pack
      CUDA_CALL( cudaStreamSynchronize( stream[1] ) );
      // Exchange of LEFT contiguous halo
      MPI_Irecv(rcvLftBuf_d, 26*(LSIZEY+2*hy), MPI_DOUBLE, mpi_peer_left, 0, MPI_COMM_WORLD, halo_swapL_recvreq);
      MPI_Isend(sndRgtBuf_d, 26*(LSIZEY+2*hy), MPI_DOUBLE, mpi_peer_right, 0, MPI_COMM_WORLD, halo_swapR_sendreq);

      // Wait end of LEFT Pack
      CUDA_CALL( cudaStreamSynchronize( stream[2] ) );
      // Exchange of RIGHT contiguous halo
      MPI_Irecv(rcvRgtBuf_d, 26*(LSIZEY+2*hy), MPI_DOUBLE, mpi_peer_right, 1, MPI_COMM_WORLD, halo_swapR_recvreq);
      MPI_Isend(sndLftBuf_d, 26*(LSIZEY+2*hy), MPI_DOUBLE, mpi_peer_left, 1, MPI_COMM_WORLD, halo_swapL_sendreq);

      // Unpack LEFT and right halos
      MPI_Waitall(1, halo_swapL_recvreq, MPI_STATUS_IGNORE);
      MPI_Waitall(1, halo_swapR_sendreq, MPI_STATUS_IGNORE);
      CUDA_CALL( (unpack_left  <<< gridSwapLR, blockSwapLR, 0, stream[1] >>> (rcvLftBuf_d, f2_soa_d + DST_LFT_OFF )) );

      MPI_Waitall(1, halo_swapR_recvreq, MPI_STATUS_IGNORE);
      MPI_Waitall(1, halo_swapL_sendreq, MPI_STATUS_IGNORE);
      CUDA_CALL( (unpack_right <<< gridSwapLR, blockSwapLR, 0, stream[2] >>> (rcvRgtBuf_d, f2_soa_d + DST_RGT_OFF )) );
#else //-- USE_MPI_REGULAR
      CUDA_CALL( cudaMemcpyAsync(sndRgtBuf_h, sndRgtBuf_d, (15+8+3)*(LSIZEY+2*hy)*sizeof(data_t), cudaMemcpyDeviceToHost, stream[1] ) );
      CUDA_CALL( cudaStreamSynchronize(stream[1]) );
      // Exchange of LEFT contiguous halo
      MPI_Irecv(rcvLftBuf_h, 26*(LSIZEY+2*hy), MPI_DOUBLE, mpi_peer_left, 0, MPI_COMM_WORLD, halo_swapL_recvreq);
      MPI_Isend(sndRgtBuf_h, 26*(LSIZEY+2*hy), MPI_DOUBLE, mpi_peer_right, 0, MPI_COMM_WORLD, halo_swapR_sendreq);

      CUDA_CALL( cudaMemcpyAsync(sndLftBuf_h, sndLftBuf_d, (15+8+3)*(LSIZEY+2*hy)*sizeof(data_t), cudaMemcpyDeviceToHost, stream[2] ) );
      CUDA_CALL( cudaStreamSynchronize(stream[2]) );
      // Exchange of LEFT contiguous halo
      MPI_Irecv(rcvRgtBuf_h, 26*(LSIZEY+2*hy), MPI_DOUBLE, mpi_peer_right, 1, MPI_COMM_WORLD, halo_swapR_recvreq);
      MPI_Isend(sndLftBuf_h, 26*(LSIZEY+2*hy), MPI_DOUBLE, mpi_peer_left, 1, MPI_COMM_WORLD, halo_swapL_sendreq);

      MPI_Waitall(1, halo_swapL_recvreq, MPI_STATUS_IGNORE);
      MPI_Waitall(1, halo_swapR_sendreq, MPI_STATUS_IGNORE);
      CUDA_CALL( cudaMemcpyAsync(rcvLftBuf_d, rcvLftBuf_h, (15+8+3)*(LSIZEY+2*hy)*sizeof(data_t), cudaMemcpyHostToDevice, stream[1]) );

      CUDA_CALL( (unpack_left  <<< gridSwapLR, blockSwapLR, 0, stream[1] >>> (rcvLftBuf_d, f2_soa_d + DST_LFT_OFF )) );

      MPI_Waitall(1, halo_swapR_recvreq, MPI_STATUS_IGNORE);
      MPI_Waitall(1, halo_swapL_sendreq, MPI_STATUS_IGNORE);
      CUDA_CALL( cudaMemcpyAsync(rcvRgtBuf_d, rcvRgtBuf_h, (15+8+3)*(LSIZEY+2*hy)*sizeof(data_t), cudaMemcpyHostToDevice, stream[2]) );

      CUDA_CALL( (unpack_right <<< gridSwapLR, blockSwapLR, 0, stream[2] >>> (rcvRgtBuf_d, f2_soa_d + DST_RGT_OFF )) );
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // Wait END of left and righ unpack kernels: processing of TOP and BOTTOM 
    // borders require that unpack of left and right halos has been fully completed 
    CUDA_CALL( cudaEventRecord (unpackLEndEvent, stream[1]) );
    CUDA_CALL( cudaEventRecord (unpackREndEvent, stream[2]) );
    ///////////////////////////////////////////////////////////////////////////

    // launching propagate and collide on left border using stream one
    CUDA_CALL( (propagateCollideL <<< gridLR, blockLR, 0, stream[1] >>> (f1_soa_d + LC_OFF, f2_soa_d + LC_OFF, yoff)) );

    // launching propagate and collide on right border using stream two
    CUDA_CALL( (propagateCollideR <<< gridLR, blockLR, 0, stream[2] >>> (f1_soa_d + RC_OFF, f2_soa_d + RC_OFF, yoff)) );

    ///////////////////////////////////////////////////////////////////////////

    // wait end of unpack left and right kernels before starting stream three
    CUDA_CALL( cudaStreamWaitEvent ( stream[3], unpackLEndEvent, 0 ) );
    CUDA_CALL( cudaStreamWaitEvent ( stream[3], unpackREndEvent, 0 ) );

    if (mpi_rank_y == (n_mpi_y-1)) {
      // launching propagate on top border using stream three
      CUDA_CALL( (propagateT <<< gridTB, blockTB, 0, stream[3] >>> (f1_soa_d + T_OFF, f2_soa_d + T_OFF)) );
      // launching bc on top border using stream three
      CUDA_CALL( (bcT <<< bcGridTB, bcBlockTB, 0, stream[3] >>> (f1_soa_d + BC_OFF, f2_soa_d + BC_OFF)) );
      // launching collide on the 3+1 topmost rows using stream three
      CUDA_CALL( (collideT <<< gridTB, blockTB, 0, stream[3] >>> (f1_soa_d + T_OFF  , f1_soa_d + T_OFF, yoff)) );
    } else {
      // launching propagate and collide on top border using stream three
      CUDA_CALL( (propagateCollideT <<< gridTB, blockTB, 0, stream[3] >>> (f1_soa_d + T_OFF, f2_soa_d + T_OFF, yoff)) );
    }

    // wait end of unpack left and right kernels before starting stream four
    CUDA_CALL( cudaStreamWaitEvent ( stream[4], unpackLEndEvent, 0 ) );
    CUDA_CALL( cudaStreamWaitEvent ( stream[4], unpackREndEvent, 0 ) );

    if (mpi_rank_y == 0) {
      // launching propagate on bottom border using stream four
      CUDA_CALL( (propagateB <<< gridTB, blockTB, 0, stream[4] >>> (f1_soa_d + B_OFF, f2_soa_d + B_OFF)) );
      // launching bc on bottom border using stream four
      CUDA_CALL( (bcB <<< bcGridTB, bcBlockTB, 0, stream[4] >>> (f1_soa_d + BC_OFF, f2_soa_d + BC_OFF)) );
      // launching collide on the 3+1 most bottom rows using stream four
      CUDA_CALL( (collideB <<< gridTB, blockTB, 0, stream[4] >>> (f1_soa_d + B_OFF  , f1_soa_d + B_OFF, yoff)) );
    } else {
      // launching propagate and collide on bottom border using stream four
      CUDA_CALL( (propagateCollideB <<< gridTB, blockTB, 0, stream[4] >>> (f1_soa_d + B_OFF, f2_soa_d + B_OFF, yoff)) );
    }

   //////////////////////////////////////////////////////////
    // wait end of all streams

    CUDA_CALL( cudaStreamSynchronize(stream[0]) );
    CUDA_CALL( cudaStreamSynchronize(stream[1]) );
    CUDA_CALL( cudaStreamSynchronize(stream[2]) );
    CUDA_CALL( cudaStreamSynchronize(stream[3]) );
    CUDA_CALL( cudaStreamSynchronize(stream[4]) );

    //////////////////////////////////////////////////////////
    // swap pointers
    f_tmp_d  = f1_soa_d;
    f1_soa_d = f2_soa_d;
    f2_soa_d = f_tmp_d;

    ///////////////////////////////////////////////////////////////////////////
#ifndef SPEC
    dT[1] += (double)(tf[1].tv_sec  - ti[1].tv_sec )*1.e6 +
             (double)(tf[1].tv_usec - ti[1].tv_usec);

    dT[2] += (double)(tf[2].tv_sec  - ti[2].tv_sec )*1.e6 +
             (double)(tf[2].tv_usec - ti[2].tv_usec);
#endif

#ifdef SPEC
    if (mpi_rank == 0 && ii % 500 == 0) {
      gettimeofday(&tf_batch, NULL);
      batch_time = (double)(tf_batch.tv_sec - ti_batch.tv_sec) +
                   (double)(tf_batch.tv_usec - ti_batch.tv_usec) * 1.e-6;
      printf("Completed iteration: %d, Time for last 500 iterations: %.2f seconds (%.4f sec/iter)\n",
             ii, batch_time, batch_time/500.0);
      fflush(stdout);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // this barrier helps to keep process synchronized
    MPI_Barrier(MPI_COMM_WORLD);
      
  }

  //////////////////////////////////////////////////////////////////////////////
  // End of LBM iterations.
 
#ifndef SPEC 
  gettimeofday(&tf[0], NULL);

  dT[0] = (double)(tf[0].tv_sec  - ti[0].tv_sec )*1.e6 +
          (double)(tf[0].tv_usec - ti[0].tv_usec);  
#endif  

  // mass check
  #ifdef CHECK_MASS
  // Lattice is copied back to the host and converted to AoS
  CUDA_CALL( cudaMemcpy(f2_soa_h, f2_soa_d, NX*NY*NPOP*sizeof(data_t), cudaMemcpyDeviceToHost) );
  SoAtoAoS(f2_soa_h, f_aos_h);
  mass = dumpMass(f_aos_h);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Reduce(&mass, &g_mass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if ( mpi_rank == 0 ) {
    if( fabs((i_mass - g_mass) / i_mass) < 1.e-10 ){
#ifdef SPEC
      fprintf(stdout, "Mass check: PASSED.\n");
#else
      fprintf(stdout, "Mass check: PASSED %.10e == %.10e (Er: %e).\n", i_mass, g_mass, fabs((i_mass - g_mass))/i_mass);
#endif
    } else{
      fprintf(stdout, "Mass check: FAILED %.10e <> %.10e (Er: %e).\n", i_mass, g_mass, fabs((i_mass - g_mass))/i_mass);
    }
    fflush(stdout);
  }
  #endif

#ifndef SPEC
  MPI_Barrier(MPI_COMM_WORLD);

  // collect execution times from different MPI processes 
  //MPI_Reduce(&dT[0], &mpimin_dt_tot, 1,     MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  //MPI_Reduce(&dT[1], &mpimin_dt_swap_nc, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  //MPI_Reduce(&dT[2], &mpimin_dt_swap_c, 1,  MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&dT[0], &mpimax_dt_tot, 1,     MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&dT[1], &mpimax_dt_swap_nc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&dT[2], &mpimax_dt_swap_c, 1,  MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  /////////////////////////////////////////////////////////////////////////// 
  // print Statistics.
  
  //latticesize = ((double)GSIZEX*(double)GSIZEY*37.0*(double)sizeof(data_t)); 
  //bw = (2.0*latticesize*(double)niter) / (dT[1]*1.e3);
  //p  = ((double)FLOPs*(double)GSIZEX*(double)GSIZEY*(double)(niter)) / (dT[3]*1.e3); 
  mlups = (double)(LSIZEX*LSIZEY)/(dT[0]/(double)niter); 
	
  // Each node prints its own satistics
  for(ii = 0; ii < mpi_size; ii++){
    if(ii == mpi_rank){
      fprintf(stdout, "[MPI%04d]: Wct: %.2f s, time/iter: %.2f us, Tswap_nc: %.2f s (%.2f us/iter), MLUP/s: %4.2f / GPU \n", 
        ii, dT[0]*1.e-6, dT[0]/(double)niter, dT[1]*1.e-6, dT[1]/(double)niter, mlups );
    }
    fflush(stdout);
    
    MPI_Barrier(MPI_COMM_WORLD);  
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  // MPI0 prints global info
  if(mpi_rank == 0){
    fprintf(stdout, "[GLOBAL-INFO] [%dx%d] NITER: %d NMPI: %d [%02dx%02d %s] Wct: %.02f s (%.02f ms/iter), Tswap_nc: %.02f s (%.02f us/iter), P: %.2f GFLOPs, MLUP/s: %.2f (FLOPs/site: %d)\n",
      GSIZEX, GSIZEY,
      niter, mpi_size,
      n_mpi_x, n_mpi_y, mpi_order,
      mpimax_dt_tot*1.e-6,     (mpimax_dt_tot*1.0e-3)     / (double)niter,
      mpimax_dt_swap_nc*1.e-6, (mpimax_dt_swap_nc) / (double)niter,
      ((double)FLOPs*(double)GSIZEX*(double)GSIZEY*(double)(niter))/(mpimax_dt_tot*1e3),
      (double)(GSIZEX*GSIZEY)/((mpimax_dt_tot)/(double)niter),
      FLOPs
    );
  }
  
  fflush(stdout);
#endif

  MPI_Barrier(MPI_COMM_WORLD);
 
  // Cuda streams are destroyed.
  CUDA_CALL( cudaStreamDestroy(stream[0]) );
  CUDA_CALL( cudaStreamDestroy(stream[1]) );
  CUDA_CALL( cudaStreamDestroy(stream[2]) );
  CUDA_CALL( cudaStreamDestroy(stream[3]) );
  CUDA_CALL( cudaStreamDestroy(stream[4]) );

  // Cuda events are destroyed.
  CUDA_CALL( cudaEventDestroy(unpackLEndEvent) );
  CUDA_CALL( cudaEventDestroy(unpackREndEvent) );

  // Frees memory.
  CUDA_CALL( cudaFree(f1_soa_d) );
  CUDA_CALL( cudaFree(f2_soa_d) );
  CUDA_CALL( cudaFreeHost(f2_soa_h) );
  
  freePbc();

  CUDA_CALL(cudaDeviceSynchronize()); // just to be sure that all ops on GPUs are finished
}
