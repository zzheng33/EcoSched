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

#ifndef _PBC_CUH
#define _PBC_CUH

// Define device costant for halo exchange
#define SRC_BOT_OFF (HX*NY + HY)
#define DST_TOP_OFF (HX*NY + HY+LSIZEY)
#define SRC_TOP_OFF (HX*NY + HY+LSIZEY-hy)
#define DST_BOT_OFF (HX*NY + HY-hy)
#define SRC_RGT_OFF ((HX+LSIZEX-hx)*NY + HY-hy)
#define DST_LFT_OFF ((HX-hx)*NY + HY-hy)
#define SRC_LFT_OFF (HX*NY + HY-hy)
#define DST_RGT_OFF ((HX+LSIZEX)*NY + HY-hy)

int pack_blockX;
int pack_gridY;

dim3 dimBlockPck;
dim3 dimGridPck;

data_t *sndBotBuf_d, *rcvTopBuf_d, *sndTopBuf_d, *rcvBotBuf_d;
data_t *sndLftBuf_d, *rcvLftBuf_d, *sndRgtBuf_d, *rcvRgtBuf_d;

cudaStream_t pbcStreams[2];
MPI_Request halo_swapT_recvreq[1], halo_swapB_recvreq[1];
MPI_Request halo_swapB_sendreq[1], halo_swapT_sendreq[1];

#ifndef SPEC_ACCEL_AWARE_MPI
data_t *sndRgtBuf_h, *sndLftBuf_h, *rcvRgtBuf_h, *rcvLftBuf_h;
data_t *sndBotBuf_h, *sndTopBuf_h, *rcvBotBuf_h, *rcvTopBuf_h;
#endif

MPI_Request halo_swapL_recvreq[1], halo_swapR_recvreq[1];
MPI_Request halo_swapL_sendreq[1], halo_swapR_sendreq[1];

///////////////////////////////////////////////////////////////////////////////
//// Init function for pbc

void initPbc(){

  // initialize grid and block size for (un)pack_* kernels
  pack_blockX = 3;
  pack_gridY  = ((LSIZEX%PACK_TB_THBLK)==0) ? (LSIZEX/PACK_TB_THBLK) : (LSIZEX/PACK_TB_THBLK+1);

  dimBlockPck.x = pack_blockX;
  dimBlockPck.y = PACK_TB_THBLK;
  dimBlockPck.z = 1;

  dimGridPck.x = 1;
  dimGridPck.y = pack_gridY;
  dimGridPck.z = 1;

  // allocate streams and memory for (un)pack_* kernels
  CUDA_CALL( cudaStreamCreateWithFlags(&pbcStreams[0], cudaStreamNonBlocking) );
  CUDA_CALL( cudaStreamCreateWithFlags(&pbcStreams[1], cudaStreamNonBlocking) );

  CUDA_CALL( cudaMalloc((void **)&sndBotBuf_d, LSIZEX*(3+8+15)*sizeof(data_t)) );
  CUDA_CALL( cudaMalloc((void **)&rcvTopBuf_d, LSIZEX*(3+8+15)*sizeof(data_t)) );
  CUDA_CALL( cudaMalloc((void **)&sndTopBuf_d, LSIZEX*(3+8+15)*sizeof(data_t)) );
  CUDA_CALL( cudaMalloc((void **)&rcvBotBuf_d, LSIZEX*(3+8+15)*sizeof(data_t)) );

  CUDA_CALL( cudaMalloc((void **)&sndLftBuf_d, (LSIZEY+2*hy)*(3+8+15)*sizeof(data_t)) );
  CUDA_CALL( cudaMalloc((void **)&rcvLftBuf_d, (LSIZEY+2*hy)*(3+8+15)*sizeof(data_t)) );
  CUDA_CALL( cudaMalloc((void **)&sndRgtBuf_d, (LSIZEY+2*hy)*(3+8+15)*sizeof(data_t)) );
  CUDA_CALL( cudaMalloc((void **)&rcvRgtBuf_d, (LSIZEY+2*hy)*(3+8+15)*sizeof(data_t)) );

  #ifndef SPEC_ACCEL_AWARE_MPI //-- USE_MPI_REGULAR
  CUDA_CALL( cudaHostAlloc((void **) &sndRgtBuf_h, (LSIZEY+2*hy)*(3+8+15)*sizeof(data_t), cudaHostAllocDefault) );
  CUDA_CALL( cudaHostAlloc((void **) &sndLftBuf_h, (LSIZEY+2*hy)*(3+8+15)*sizeof(data_t), cudaHostAllocDefault) );
  CUDA_CALL( cudaHostAlloc((void **) &rcvRgtBuf_h, (LSIZEY+2*hy)*(3+8+15)*sizeof(data_t), cudaHostAllocDefault) );
  CUDA_CALL( cudaHostAlloc((void **) &rcvLftBuf_h, (LSIZEY+2*hy)*(3+8+15)*sizeof(data_t), cudaHostAllocDefault) );

  CUDA_CALL( cudaHostAlloc((void **) &sndBotBuf_h, LSIZEX*(3+8+15)*sizeof(data_t), cudaHostAllocDefault) );
  CUDA_CALL( cudaHostAlloc((void **) &sndTopBuf_h, LSIZEX*(3+8+15)*sizeof(data_t), cudaHostAllocDefault) );
  CUDA_CALL( cudaHostAlloc((void **) &rcvBotBuf_h, LSIZEX*(3+8+15)*sizeof(data_t), cudaHostAllocDefault) );
  CUDA_CALL( cudaHostAlloc((void **) &rcvTopBuf_h, LSIZEX*(3+8+15)*sizeof(data_t), cudaHostAllocDefault) );
  #endif

}

///////////////////////////////////////////////////////////////////////////////
//// Free function for pbc

void freePbc(){
  
  CUDA_CALL( cudaStreamDestroy( pbcStreams[0] ) );
  CUDA_CALL( cudaStreamDestroy( pbcStreams[1] ) );

  CUDA_CALL( cudaFree(sndBotBuf_d) );
  CUDA_CALL( cudaFree(rcvTopBuf_d) );
  CUDA_CALL( cudaFree(sndTopBuf_d) );
  CUDA_CALL( cudaFree(rcvBotBuf_d) );

  CUDA_CALL( cudaFree(sndLftBuf_d) );
  CUDA_CALL( cudaFree(rcvLftBuf_d) );
  CUDA_CALL( cudaFree(sndRgtBuf_d) );
  CUDA_CALL( cudaFree(rcvRgtBuf_d) );
 
  #ifndef SPEC_ACCEL_AWARE_MPI //-- USE_MPI_REGULAR
  CUDA_CALL( cudaFree(sndRgtBuf_h) );
  CUDA_CALL( cudaFree(sndLftBuf_h) );
  CUDA_CALL( cudaFree(rcvRgtBuf_h) );
  CUDA_CALL( cudaFree(rcvLftBuf_h) );
  
  CUDA_CALL( cudaFree(sndBotBuf_h) );
  CUDA_CALL( cudaFree(sndTopBuf_h) );
  CUDA_CALL( cudaFree(rcvBotBuf_h) );
  CUDA_CALL( cudaFree(rcvTopBuf_h) );
  #endif
 
}

///////////////////////////////////////////////////////////////////////////////
//// Pack non-contiguous buffer in a contiguos buffer
//// Top boder

__global__ void pack_bot(data_t *f, data_t *sndBuf){

  int idx1_c, idx2_c, idx3_c, idx_nc;

  // Index for contiguous buffer to store 3 rows
  idx3_c = ( blockIdx.y  * 3 * blockDim.y ) +
           ( threadIdx.y * 3              ) +
           ( threadIdx.x                  );

  // Index for contiguous buffer to store 2 rows
  idx2_c = ( blockIdx.y  * 2 * blockDim.y ) +
           ( threadIdx.y * 2              ) +
           ( threadIdx.x                  );

  // Index for contiguous buffer to store 1 row
  idx1_c = ( blockIdx.y  * blockDim.y ) +
           ( threadIdx.y              ) +
           ( threadIdx.x              );

  // Index for non-contiguous buffer
  idx_nc = ( blockIdx.y  * blockDim.y * NY_d ) +  // Offset-y block
           ( threadIdx.y * NY_d              ) +  // Offset columns
           ( threadIdx.x                     );   // Index of thread

  if( (threadIdx.x < blockDim.x) && (idx_nc < (LSIZEX_d*NY_d)) ){

    if(threadIdx.x == 0){ sndBuf[ idx1_c               ] = f[ idx_nc + 0 *NX_d*NY_d ]; }
    if(threadIdx.x != 2){ sndBuf[ idx2_c + 1 *LSIZEX_d ] = f[ idx_nc + 3 *NX_d*NY_d ]; }
    if(threadIdx.x == 0){ sndBuf[ idx1_c + 3 *LSIZEX_d ] = f[ idx_nc + 4 *NX_d*NY_d ]; }
    
                          sndBuf[ idx3_c + 4 *LSIZEX_d ] = f[ idx_nc + 8 *NX_d*NY_d ];
    
    if(threadIdx.x != 2){ sndBuf[ idx2_c + 7 *LSIZEX_d ] = f[ idx_nc + 9 *NX_d*NY_d ]; }
    if(threadIdx.x == 0){ sndBuf[ idx1_c + 9 *LSIZEX_d ] = f[ idx_nc + 10*NX_d*NY_d ]; }
    
                          sndBuf[ idx3_c + 10*LSIZEX_d ] = f[ idx_nc + 15*NX_d*NY_d ];
    
    if(threadIdx.x != 2){ sndBuf[ idx2_c + 13*LSIZEX_d ] = f[ idx_nc + 16*NX_d*NY_d ]; }
    if(threadIdx.x == 0){ sndBuf[ idx1_c + 15*LSIZEX_d ] = f[ idx_nc + 17*NX_d*NY_d ]; }
    
                          sndBuf[ idx3_c + 16*LSIZEX_d ] = f[ idx_nc + 22*NX_d*NY_d ];
    
    if(threadIdx.x != 2){ sndBuf[ idx2_c + 19*LSIZEX_d ] = f[ idx_nc + 23*NX_d*NY_d ]; }
    if(threadIdx.x == 0){ sndBuf[ idx1_c + 21*LSIZEX_d ] = f[ idx_nc + 24*NX_d*NY_d ]; }
    if(threadIdx.x != 2){ sndBuf[ idx2_c + 22*LSIZEX_d ] = f[ idx_nc + 29*NX_d*NY_d ]; }
    if(threadIdx.x == 0){ sndBuf[ idx1_c + 24*LSIZEX_d ] = f[ idx_nc + 30*NX_d*NY_d ];
                          sndBuf[ idx1_c + 25*LSIZEX_d ] = f[ idx_nc + 34*NX_d*NY_d ]; }

  }
}

//////////////////////////////////////////////////////////////////////////////////////////
//// Unpack a contiguous buffer in a non-contiguous buffer
//// Top border

__global__ void unpack_top(data_t *rcvBuf, data_t *f){

  int idx1_c, idx2_c, idx3_c, idx_nc;

  // Index for contiguous buffer to store 3 rows
  idx3_c = ( blockIdx.y  * 3 * blockDim.y ) +
           ( threadIdx.y * 3              ) +
           ( threadIdx.x                  );

  // Index for contiguous buffer to store 2 rows
  idx2_c = ( blockIdx.y  * 2 * blockDim.y ) +
           ( threadIdx.y * 2              ) +
           ( threadIdx.x                  );

  // Index for contiguous buffer to store 1 row
  idx1_c = ( blockIdx.y  * blockDim.y ) +
           ( threadIdx.y              ) +
           ( threadIdx.x              );

  // Index for non-contiguous buffer
  idx_nc = ( blockIdx.y  * blockDim.y * NY_d ) +  // Offset-y block
           ( threadIdx.y * NY_d              ) +  // Offset columns
           ( threadIdx.x                     );   // Index of thread

  if( (threadIdx.x < blockDim.x) && (idx_nc < (LSIZEX_d*NY_d)) ){

    if(threadIdx.x == 0){ f[ idx_nc + 0 *NX_d*NY_d ] = rcvBuf[ idx1_c               ]; }
    if(threadIdx.x != 2){ f[ idx_nc + 3 *NX_d*NY_d ] = rcvBuf[ idx2_c + 1 *LSIZEX_d ]; }
    if(threadIdx.x == 0){ f[ idx_nc + 4 *NX_d*NY_d ] = rcvBuf[ idx1_c + 3 *LSIZEX_d ]; }

                          f[ idx_nc + 8 *NX_d*NY_d ] = rcvBuf[ idx3_c + 4 *LSIZEX_d ];

    if(threadIdx.x != 2){ f[ idx_nc + 9 *NX_d*NY_d ] = rcvBuf[ idx2_c + 7 *LSIZEX_d ]; }
    if(threadIdx.x == 0){ f[ idx_nc + 10*NX_d*NY_d ] = rcvBuf[ idx1_c + 9 *LSIZEX_d ]; }

                          f[ idx_nc + 15*NX_d*NY_d ] = rcvBuf[ idx3_c + 10*LSIZEX_d ];

    if(threadIdx.x != 2){ f[ idx_nc + 16*NX_d*NY_d ] = rcvBuf[ idx2_c + 13*LSIZEX_d ]; }
    if(threadIdx.x == 0){ f[ idx_nc + 17*NX_d*NY_d ] = rcvBuf[ idx1_c + 15*LSIZEX_d ]; }

                          f[ idx_nc + 22*NX_d*NY_d ] = rcvBuf[ idx3_c + 16*LSIZEX_d ];

    if(threadIdx.x != 2){ f[ idx_nc + 23*NX_d*NY_d ] = rcvBuf[ idx2_c + 19*LSIZEX_d ]; }
    if(threadIdx.x == 0){ f[ idx_nc + 24*NX_d*NY_d ] = rcvBuf[ idx1_c + 21*LSIZEX_d ]; }
    if(threadIdx.x != 2){ f[ idx_nc + 29*NX_d*NY_d ] = rcvBuf[ idx2_c + 22*LSIZEX_d ]; }
    if(threadIdx.x == 0){ f[ idx_nc + 30*NX_d*NY_d ] = rcvBuf[ idx1_c + 24*LSIZEX_d ];
                          f[ idx_nc + 34*NX_d*NY_d ] = rcvBuf[ idx1_c + 25*LSIZEX_d ]; }
  }
}

///////////////////////////////////////////////////////////////////////////////
//// Pack non-contiguous buffer in a contiguos buffer
//// Bottom boder
 
__global__ void pack_top(data_t *f, data_t *sndBuf){

  int idx1_c, idx2_c, idx3_c, idx_nc;

  // Index for contiguous buffer to store 3 rows
  idx3_c = ( blockIdx.y  * 3 * blockDim.y ) +
           ( threadIdx.y * 3              ) +
           ( threadIdx.x                  );

  // Index for contiguous buffer to store 2 rows
  idx2_c = ( blockIdx.y  * 2 * blockDim.y ) +
           ( threadIdx.y * 2              ) +
           ( threadIdx.x - 1              );

  // Index for contiguous buffer to store 1 row
  idx1_c = ( blockIdx.y  * blockDim.y ) +
           ( threadIdx.y              ) +
           ( threadIdx.x - 2          );

  // Index for non-contiguous buffer
  idx_nc = ( blockIdx.y  * blockDim.y * NY_d ) +  // Offset-y block
           ( threadIdx.y * NY_d              ) +  // Offset columns
           ( threadIdx.x                     );   // Index of thread

  if( (threadIdx.x < blockDim.x) && (idx_nc < (LSIZEX_d*NY_d)) ){

    if(threadIdx.x == 2){ sndBuf[ idx1_c               ] = f[ idx_nc + 2 *NX_d*NY_d ];
                          sndBuf[ idx1_c + 1 *LSIZEX_d ] = f[ idx_nc + 6 *NX_d*NY_d ]; }
    if(threadIdx.x != 0){ sndBuf[ idx2_c + 2 *LSIZEX_d ] = f[ idx_nc + 7 *NX_d*NY_d ]; }
    if(threadIdx.x == 2){ sndBuf[ idx1_c + 4 *LSIZEX_d ] = f[ idx_nc + 12*NX_d*NY_d ]; }
    if(threadIdx.x != 0){ sndBuf[ idx2_c + 5 *LSIZEX_d ] = f[ idx_nc + 13*NX_d*NY_d ]; }

                          sndBuf[ idx3_c + 7 *LSIZEX_d ] = f[ idx_nc + 14*NX_d*NY_d ];

    if(threadIdx.x == 2){ sndBuf[ idx1_c + 10*LSIZEX_d ] = f[ idx_nc + 19*NX_d*NY_d ]; }
    if(threadIdx.x != 0){ sndBuf[ idx2_c + 11*LSIZEX_d ] = f[ idx_nc + 20*NX_d*NY_d ]; }

                          sndBuf[ idx3_c + 13*LSIZEX_d ] = f[ idx_nc + 21*NX_d*NY_d ];

    if(threadIdx.x == 2){ sndBuf[ idx1_c + 16*LSIZEX_d ] = f[ idx_nc + 26*NX_d*NY_d ]; }
    if(threadIdx.x != 0){ sndBuf[ idx2_c + 17*LSIZEX_d ] = f[ idx_nc + 27*NX_d*NY_d ]; }

                          sndBuf[ idx3_c + 19*LSIZEX_d ] = f[ idx_nc + 28*NX_d*NY_d ];

    if(threadIdx.x == 2){ sndBuf[ idx1_c + 22*LSIZEX_d ] = f[ idx_nc + 32*NX_d*NY_d ]; }
    if(threadIdx.x != 0){ sndBuf[ idx2_c + 23*LSIZEX_d ] = f[ idx_nc + 33*NX_d*NY_d ]; }
    if(threadIdx.x == 2){ sndBuf[ idx1_c + 25*LSIZEX_d ] = f[ idx_nc + 36*NX_d*NY_d ]; }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
//// Unpack a contiguous buffer in a non-contiguous buffer
//// Bottom border

__global__ void unpack_bot(data_t *rcvBuf, data_t *f){

  int idx1_c, idx2_c, idx3_c, idx_nc;

  // Index for contiguous buffer to store 3 rows
  idx3_c = ( blockIdx.y  * 3 * blockDim.y ) +
           ( threadIdx.y * 3              ) +
           ( threadIdx.x                  );

  // Index for contiguous buffer to store 2 rows
  idx2_c = ( blockIdx.y  * 2 * blockDim.y ) +
           ( threadIdx.y * 2              ) +
           ( threadIdx.x - 1              );

  // Index for contiguous buffer to store 1 row
  idx1_c = ( blockIdx.y  * blockDim.y ) +
           ( threadIdx.y              ) +
           ( threadIdx.x - 2          );

  // Index for non-contiguous buffer
  idx_nc = ( blockIdx.y  * blockDim.y * NY_d ) +  // Offset-y block
           ( threadIdx.y * NY_d              ) +  // Offset columns
           ( threadIdx.x                     );   // Index of thread

  if( (threadIdx.x < blockDim.x) && (idx_nc < (LSIZEX_d*NY_d)) ){

    if(threadIdx.x == 2){ f[ idx_nc + 2 *NX_d*NY_d ] = rcvBuf[ idx1_c               ];
                          f[ idx_nc + 6 *NX_d*NY_d ] = rcvBuf[ idx1_c + 1 *LSIZEX_d ]; }
    if(threadIdx.x != 0){ f[ idx_nc + 7 *NX_d*NY_d ] = rcvBuf[ idx2_c + 2 *LSIZEX_d ]; }
    if(threadIdx.x == 2){ f[ idx_nc + 12*NX_d*NY_d ] = rcvBuf[ idx1_c + 4 *LSIZEX_d ]; }
    if(threadIdx.x != 0){ f[ idx_nc + 13*NX_d*NY_d ] = rcvBuf[ idx2_c + 5 *LSIZEX_d ]; }

                          f[ idx_nc + 14*NX_d*NY_d ] = rcvBuf[ idx3_c + 7 *LSIZEX_d ];

    if(threadIdx.x == 2){ f[ idx_nc + 19*NX_d*NY_d ] = rcvBuf[ idx1_c + 10*LSIZEX_d ]; }
    if(threadIdx.x != 0){ f[ idx_nc + 20*NX_d*NY_d ] = rcvBuf[ idx2_c + 11*LSIZEX_d ]; }

                          f[ idx_nc + 21*NX_d*NY_d ] = rcvBuf[ idx3_c + 13*LSIZEX_d ];

    if(threadIdx.x == 2){ f[ idx_nc + 26*NX_d*NY_d ] = rcvBuf[ idx1_c + 16*LSIZEX_d ]; }
    if(threadIdx.x != 0){ f[ idx_nc + 27*NX_d*NY_d ] = rcvBuf[ idx2_c + 17*LSIZEX_d ]; }

                          f[ idx_nc + 28*NX_d*NY_d ] = rcvBuf[ idx3_c + 19*LSIZEX_d ];

    if(threadIdx.x == 2){ f[ idx_nc + 32*NX_d*NY_d ] = rcvBuf[ idx1_c + 22*LSIZEX_d ]; }
    if(threadIdx.x != 0){ f[ idx_nc + 33*NX_d*NY_d ] = rcvBuf[ idx2_c + 23*LSIZEX_d ]; }
    if(threadIdx.x == 2){ f[ idx_nc + 36*NX_d*NY_d ] = rcvBuf[ idx1_c + 25*LSIZEX_d ]; }
  }
}

void pbc_nc(data_t *f_d) {
  #ifdef SPEC_ACCEL_AWARE_MPI
  if(mpi_rank_y != 0        ) {
    // Pack of BOTTOM non contiguous border
    CUDA_CALL( (pack_bot <<< dimGridPck, dimBlockPck, 0, pbcStreams[0] >>> (f_d + SRC_BOT_OFF, sndBotBuf_d)) );
  }
  if(mpi_rank_y != (n_mpi_y-1)) {
    // Pack of TOP non contiguous border
    CUDA_CALL( (pack_top <<< dimGridPck, dimBlockPck, 0, pbcStreams[1] >>> (f_d + SRC_TOP_OFF, sndTopBuf_d)) );
  }

  // Receive of TOP and BOTTOM non-contiguous halo
  if(mpi_rank_y != (n_mpi_y-1)) {
    MPI_Irecv(rcvTopBuf_d, (15+8+3)*LSIZEX, MPI_DOUBLE, mpi_peer_top, 0, MPI_COMM_WORLD, halo_swapT_recvreq);
  }
  if(mpi_rank_y != 0        ) {
    MPI_Irecv(rcvBotBuf_d, (15+8+3)*LSIZEX, MPI_DOUBLE, mpi_peer_bot, 1, MPI_COMM_WORLD, halo_swapB_recvreq);
  }

  // Wait end of BOTTOM Pack and send BOTTOM border
  if(mpi_rank_y != 0        ) {
    CUDA_CALL( cudaStreamSynchronize( pbcStreams[0] ) );
    MPI_Isend(sndBotBuf_d, (15+8+3)*LSIZEX, MPI_DOUBLE, mpi_peer_bot, 0, MPI_COMM_WORLD, halo_swapB_sendreq);
  }
  // Wait the end of TOP Pack and send TOP border
  if(mpi_rank_y != (n_mpi_y-1)) {
    CUDA_CALL( cudaStreamSynchronize( pbcStreams[1] ) );
    MPI_Isend(sndTopBuf_d, (15+8+3)*LSIZEX, MPI_DOUBLE, mpi_peer_top, 1, MPI_COMM_WORLD, halo_swapT_sendreq);
  }

  if(mpi_rank_y != (n_mpi_y-1)) {
    MPI_Waitall(1, halo_swapT_recvreq, MPI_STATUS_IGNORE); // Wait end of receive
    MPI_Waitall(1, halo_swapT_sendreq, MPI_STATUS_IGNORE); // Wait end of send
//cudaDeviceSynchronize();
    // Unpack of TOP halo
    CUDA_CALL( (unpack_top <<< dimGridPck, dimBlockPck, 0, pbcStreams[0] >>> (rcvTopBuf_d, f_d + DST_TOP_OFF )) );
  }
  if(mpi_rank_y != 0        ) {
    MPI_Waitall(1, halo_swapB_recvreq, MPI_STATUS_IGNORE); // Wait end of receive
    MPI_Waitall(1, halo_swapB_sendreq, MPI_STATUS_IGNORE); // Wait end of send
//cudaDeviceSynchronize();
    // Unpack of BOTTOM halo
    CUDA_CALL( (unpack_bot <<< dimGridPck, dimBlockPck, 0, pbcStreams[1] >>> (rcvBotBuf_d, f_d + DST_BOT_OFF)) );
  }

  // Wait the end of TOP and BOTTOM unpack: this is required to have fully completed before staring
  // exchanging of left and right borders
  if(mpi_rank_y != (n_mpi_y-1)) {
    CUDA_CALL( cudaStreamSynchronize( pbcStreams[0] ) );
  }
  if(mpi_rank_y != 0        ) {
    CUDA_CALL( cudaStreamSynchronize( pbcStreams[1] ) );
  }
#else //-- USE_MPI_REGULAR
  if(mpi_rank_y != 0        ) {
    // Pack of all BOTTOM non contiguous halos
    CUDA_CALL( (pack_bot <<< dimGridPck, dimBlockPck, 0, pbcStreams[0] >>> (f_d + SRC_BOT_OFF, sndBotBuf_d)) );
  }
  if(mpi_rank_y != (n_mpi_y-1)) {
    // Pack of all TOP non contiguous halos
    CUDA_CALL( (pack_top <<< dimGridPck, dimBlockPck, 0, pbcStreams[1] >>> (f_d + SRC_TOP_OFF, sndTopBuf_d)) );
  }

  // Top halos are received
  if(mpi_rank_y != (n_mpi_y-1)) {
    MPI_Irecv(rcvTopBuf_h, (15+8+3)*LSIZEX, MPI_DOUBLE, mpi_peer_top, 0, MPI_COMM_WORLD, halo_swapT_recvreq);
  }
  // Bottom halos are received
  if(mpi_rank_y != 0        ) {
    MPI_Irecv(rcvBotBuf_h, (15+8+3)*LSIZEX, MPI_DOUBLE, mpi_peer_bot, 1, MPI_COMM_WORLD, halo_swapB_recvreq);
  }

  // Wait the end of BOTTOM Pack, copy buffer from device to host and send BOTTOM border
  if(mpi_rank_y != 0        ) {
    CUDA_CALL( cudaStreamSynchronize( pbcStreams[0] ) );
    CUDA_CALL( cudaMemcpyAsync(sndBotBuf_h, sndBotBuf_d, (15+8+3)*LSIZEX*sizeof(data_t), cudaMemcpyDeviceToHost, pbcStreams[0] ) );
    CUDA_CALL( cudaStreamSynchronize(pbcStreams[0]) );
    MPI_Isend(sndBotBuf_h, (15+8+3)*LSIZEX, MPI_DOUBLE, mpi_peer_bot, 0, MPI_COMM_WORLD, halo_swapB_sendreq);
  }
  // Wait the end of TOP Pack, copy buffer from device to host and send TOP border
  if(mpi_rank_y != (n_mpi_y-1)) {
    CUDA_CALL( cudaStreamSynchronize( pbcStreams[1] ) );
    CUDA_CALL( cudaMemcpyAsync(sndTopBuf_h, sndTopBuf_d, (15+8+3)*LSIZEX*sizeof(data_t), cudaMemcpyDeviceToHost, pbcStreams[1] ) );
    CUDA_CALL( cudaStreamSynchronize(pbcStreams[1]) );
    MPI_Isend(sndTopBuf_h, (15+8+3)*LSIZEX, MPI_DOUBLE, mpi_peer_top, 1, MPI_COMM_WORLD, halo_swapT_sendreq);
  }

  if(mpi_rank_y != (n_mpi_y-1)) {
    MPI_Waitall(1, halo_swapT_recvreq, MPI_STATUS_IGNORE); // Wait end of receive
    MPI_Waitall(1, halo_swapT_sendreq, MPI_STATUS_IGNORE); // Wait end of send
    // Copy buffer from host to device
    CUDA_CALL( cudaMemcpyAsync(rcvTopBuf_d, rcvTopBuf_h, (15+8+3)*LSIZEX*sizeof(data_t), cudaMemcpyHostToDevice, pbcStreams[0]) );
    CUDA_CALL( cudaStreamSynchronize(pbcStreams[0]) );
    // Unpack of all TOP halos
    CUDA_CALL( (unpack_top <<< dimGridPck, dimBlockPck, 0, pbcStreams[0] >>> (rcvTopBuf_d, f_d + DST_TOP_OFF )) );
  }
  if(mpi_rank_y != 0        ) {
    MPI_Waitall(1, halo_swapB_recvreq, MPI_STATUS_IGNORE); // Wait end of receive
    MPI_Waitall(1, halo_swapB_sendreq, MPI_STATUS_IGNORE); // Wait end of send
    // Copy buffer from host to device
    CUDA_CALL( cudaMemcpyAsync(rcvBotBuf_d, rcvBotBuf_h, (15+8+3)*LSIZEX*sizeof(data_t), cudaMemcpyHostToDevice, pbcStreams[1]) );
    CUDA_CALL( cudaStreamSynchronize(pbcStreams[1]) );
    // Unpack of all BOTTOM halos
    CUDA_CALL( (unpack_bot <<< dimGridPck, dimBlockPck, 0, pbcStreams[1] >>> (rcvBotBuf_d, f_d + DST_BOT_OFF)) );
  }

  // Wait the end of TOP and BOTTOM unpack: this is required to have fully completed before staring
  // exchanging of left and right borders
  if(mpi_rank_y != (n_mpi_y-1)) {
    CUDA_CALL( cudaStreamSynchronize( pbcStreams[0] ) );
  }
  if(mpi_rank_y != 0        ) {
    CUDA_CALL( cudaStreamSynchronize( pbcStreams[1] ) );
  }
  #endif
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void pack_right(data_t *f, data_t *sndBuf){

  int idx_c, idx_nc;

  // Index for contiguous buffer
  idx_c = ( blockIdx.y * (LSIZEY_d+2*hy) ) +
          ( blockDim.x * blockIdx.x    ) +
          ( threadIdx.x                );

  // Index for non-contiguous buffer
  idx_nc = ( blockIdx.y * NY_d         ) +
           ( blockDim.x * blockIdx.x ) +
           ( threadIdx.x             );


  if( (threadIdx.x < blockDim.x) && (idx_c < ((blockIdx.y+1)*(LSIZEY_d+2*hy))) ){

    sndBuf[ idx_c                    ] = f[ idx_nc + 0 *NX_d*NY_d ];
    sndBuf[ idx_c + 3 *(LSIZEY_d+2*hy) ] = f[ idx_nc + 1 *NX_d*NY_d ];
    sndBuf[ idx_c + 6 *(LSIZEY_d+2*hy) ] = f[ idx_nc + 2 *NX_d*NY_d ];

    if(blockIdx.y!=0){
      sndBuf[ idx_c + 8 *(LSIZEY_d+2*hy) ] = f[ idx_nc + 3 *NX_d*NY_d ];
      sndBuf[ idx_c + 10*(LSIZEY_d+2*hy) ] = f[ idx_nc + 4 *NX_d*NY_d ];
      sndBuf[ idx_c + 12*(LSIZEY_d+2*hy) ] = f[ idx_nc + 5 *NX_d*NY_d ];
      sndBuf[ idx_c + 14*(LSIZEY_d+2*hy) ] = f[ idx_nc + 6 *NX_d*NY_d ];
      sndBuf[ idx_c + 16*(LSIZEY_d+2*hy) ] = f[ idx_nc + 7 *NX_d*NY_d ];
    }

    if(blockIdx.y==2){
      sndBuf[ idx_c + 17*(LSIZEY_d+2*hy) ] = f[ idx_nc + 8 *NX_d*NY_d ];
      sndBuf[ idx_c + 18*(LSIZEY_d+2*hy) ] = f[ idx_nc + 9 *NX_d*NY_d ];
      sndBuf[ idx_c + 19*(LSIZEY_d+2*hy) ] = f[ idx_nc + 10*NX_d*NY_d ];
      sndBuf[ idx_c + 20*(LSIZEY_d+2*hy) ] = f[ idx_nc + 11*NX_d*NY_d ];
      sndBuf[ idx_c + 21*(LSIZEY_d+2*hy) ] = f[ idx_nc + 12*NX_d*NY_d ];
      sndBuf[ idx_c + 22*(LSIZEY_d+2*hy) ] = f[ idx_nc + 13*NX_d*NY_d ];
      sndBuf[ idx_c + 23*(LSIZEY_d+2*hy) ] = f[ idx_nc + 14*NX_d*NY_d ];
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void unpack_left(data_t *rcvBuf, data_t *f){

  int idx_c, idx_nc;

  // Index for contiguous buffer
  idx_c = ( blockIdx.y * (LSIZEY_d+2*hy) ) +
          ( blockDim.x * blockIdx.x    ) +
          ( threadIdx.x                );

  // Index for non-contiguous buffer
  idx_nc = ( blockIdx.y * NY_d         ) +
           ( blockDim.x * blockIdx.x ) +
           ( threadIdx.x             );


  if( (threadIdx.x < blockDim.x) && (idx_c < ((blockIdx.y+1)*(LSIZEY_d+2*hy))) ){

    f[ idx_nc + 0 *NX_d*NY_d ] = rcvBuf[ idx_c                    ];
    f[ idx_nc + 1 *NX_d*NY_d ] = rcvBuf[ idx_c + 3 *(LSIZEY_d+2*hy) ];
    f[ idx_nc + 2 *NX_d*NY_d ] = rcvBuf[ idx_c + 6 *(LSIZEY_d+2*hy) ];

    if(blockIdx.y!=0){
      f[ idx_nc + 3 *NX_d*NY_d ] = rcvBuf[ idx_c + 8 *(LSIZEY_d+2*hy) ];
      f[ idx_nc + 4 *NX_d*NY_d ] = rcvBuf[ idx_c + 10*(LSIZEY_d+2*hy) ];
      f[ idx_nc + 5 *NX_d*NY_d ] = rcvBuf[ idx_c + 12*(LSIZEY_d+2*hy) ];
      f[ idx_nc + 6 *NX_d*NY_d ] = rcvBuf[ idx_c + 14*(LSIZEY_d+2*hy) ];
      f[ idx_nc + 7 *NX_d*NY_d ] = rcvBuf[ idx_c + 16*(LSIZEY_d+2*hy) ];
    }

    if(blockIdx.y==2){
      f[ idx_nc + 8 *NX_d*NY_d ] = rcvBuf[ idx_c + 17*(LSIZEY_d+2*hy) ];
      f[ idx_nc + 9 *NX_d*NY_d ] = rcvBuf[ idx_c + 18*(LSIZEY_d+2*hy) ];
      f[ idx_nc + 10*NX_d*NY_d ] = rcvBuf[ idx_c + 19*(LSIZEY_d+2*hy) ];
      f[ idx_nc + 11*NX_d*NY_d ] = rcvBuf[ idx_c + 20*(LSIZEY_d+2*hy) ];
      f[ idx_nc + 12*NX_d*NY_d ] = rcvBuf[ idx_c + 21*(LSIZEY_d+2*hy) ];
      f[ idx_nc + 13*NX_d*NY_d ] = rcvBuf[ idx_c + 22*(LSIZEY_d+2*hy) ];
      f[ idx_nc + 14*NX_d*NY_d ] = rcvBuf[ idx_c + 23*(LSIZEY_d+2*hy) ];
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void pack_left(data_t *f, data_t *sndBuf){

  int idx_c, idx_nc;

  // Index for contiguous buffer
  idx_c = ( blockIdx.y * (LSIZEY_d+2*hy) ) +
          ( blockDim.x * blockIdx.x    ) +
          ( threadIdx.x                );

  // Index for non-contiguous buffer
  idx_nc = ( blockIdx.y * NY_d         ) +
           ( blockDim.x * blockIdx.x ) +
           ( threadIdx.x             );


  if( (threadIdx.x < blockDim.x) && (idx_c < ((blockIdx.y+1)*(LSIZEY_d+2*hy))) ){

    if(blockIdx.y==0){
      sndBuf[ idx_c                    ] = f[ idx_nc + 22*NX_d*NY_d ];
      sndBuf[ idx_c + 1 *(LSIZEY_d+2*hy) ] = f[ idx_nc + 23*NX_d*NY_d ];
      sndBuf[ idx_c + 2 *(LSIZEY_d+2*hy) ] = f[ idx_nc + 24*NX_d*NY_d ];
      sndBuf[ idx_c + 3 *(LSIZEY_d+2*hy) ] = f[ idx_nc + 25*NX_d*NY_d ];
      sndBuf[ idx_c + 4 *(LSIZEY_d+2*hy) ] = f[ idx_nc + 26*NX_d*NY_d ];
      sndBuf[ idx_c + 5 *(LSIZEY_d+2*hy) ] = f[ idx_nc + 27*NX_d*NY_d ];
      sndBuf[ idx_c + 6 *(LSIZEY_d+2*hy) ] = f[ idx_nc + 28*NX_d*NY_d ];
    }

    if(blockIdx.y!=2){
      sndBuf[ idx_c + 7 *(LSIZEY_d+2*hy) ] = f[ idx_nc + 29*NX_d*NY_d ];
      sndBuf[ idx_c + 9 *(LSIZEY_d+2*hy) ] = f[ idx_nc + 30*NX_d*NY_d ];
      sndBuf[ idx_c + 11*(LSIZEY_d+2*hy) ] = f[ idx_nc + 31*NX_d*NY_d ];
      sndBuf[ idx_c + 13*(LSIZEY_d+2*hy) ] = f[ idx_nc + 32*NX_d*NY_d ];
      sndBuf[ idx_c + 15*(LSIZEY_d+2*hy) ] = f[ idx_nc + 33*NX_d*NY_d ];
    }

    sndBuf[ idx_c + 17*(LSIZEY_d+2*hy) ] = f[ idx_nc + 34*NX_d*NY_d ];
    sndBuf[ idx_c + 20*(LSIZEY_d+2*hy) ] = f[ idx_nc + 35*NX_d*NY_d ];
    sndBuf[ idx_c + 23*(LSIZEY_d+2*hy) ] = f[ idx_nc + 36*NX_d*NY_d ];
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void unpack_right(data_t *rcvBuf, data_t *f){

  int idx_c, idx_nc;

  // Index for contiguous buffer
  idx_c = ( blockIdx.y * (LSIZEY_d+2*hy) ) +
          ( blockDim.x * blockIdx.x    ) +
          ( threadIdx.x                );

  // Index for non-contiguous buffer
  idx_nc = ( blockIdx.y * NY_d         ) +
           ( blockDim.x * blockIdx.x ) +
           ( threadIdx.x             );


  if( (threadIdx.x < blockDim.x) && (idx_c < ((blockIdx.y+1)*(LSIZEY_d+2*hy))) ){

    if(blockIdx.y==0){
      f[ idx_nc + 22*NX_d*NY_d ] = rcvBuf[ idx_c                    ];
      f[ idx_nc + 23*NX_d*NY_d ] = rcvBuf[ idx_c + 1 *(LSIZEY_d+2*hy) ];
      f[ idx_nc + 24*NX_d*NY_d ] = rcvBuf[ idx_c + 2 *(LSIZEY_d+2*hy) ];
      f[ idx_nc + 25*NX_d*NY_d ] = rcvBuf[ idx_c + 3 *(LSIZEY_d+2*hy) ];
      f[ idx_nc + 26*NX_d*NY_d ] = rcvBuf[ idx_c + 4 *(LSIZEY_d+2*hy) ];
      f[ idx_nc + 27*NX_d*NY_d ] = rcvBuf[ idx_c + 5 *(LSIZEY_d+2*hy) ];
      f[ idx_nc + 28*NX_d*NY_d ] = rcvBuf[ idx_c + 6 *(LSIZEY_d+2*hy) ];
    }

    if(blockIdx.y!=2){
      f[ idx_nc + 29*NX_d*NY_d ] = rcvBuf[ idx_c + 7 *(LSIZEY_d+2*hy) ];
      f[ idx_nc + 30*NX_d*NY_d ] = rcvBuf[ idx_c + 9 *(LSIZEY_d+2*hy) ];
      f[ idx_nc + 31*NX_d*NY_d ] = rcvBuf[ idx_c + 11*(LSIZEY_d+2*hy) ];
      f[ idx_nc + 32*NX_d*NY_d ] = rcvBuf[ idx_c + 13*(LSIZEY_d+2*hy) ];
      f[ idx_nc + 33*NX_d*NY_d ] = rcvBuf[ idx_c + 15*(LSIZEY_d+2*hy) ];
    }

    f[ idx_nc + 34*NX_d*NY_d ] = rcvBuf[ idx_c + 17*(LSIZEY_d+2*hy) ];
    f[ idx_nc + 35*NX_d*NY_d ] = rcvBuf[ idx_c + 20*(LSIZEY_d+2*hy) ];
    f[ idx_nc + 36*NX_d*NY_d ] = rcvBuf[ idx_c + 23*(LSIZEY_d+2*hy) ];
  }
}

#endif /* _PBC_CUH */
