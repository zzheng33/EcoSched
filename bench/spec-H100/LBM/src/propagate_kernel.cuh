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

#ifndef _PROPAGATE_KERNEL_CUH
#define _PROPAGATE_KERNEL_CUH

///////////////////////////////////////////////////////////////////////////////
// Device kernel function for propagate.

__device__ void propagate (
      data_t * __restrict__ const p_nxt,
      const data_t * __restrict__ const p_prv,
      long idx_cur
    ) {
  
  
  	/////////////////////////////////////////////////////////////////////////////
  	// Populations are moved from previous lattice to next one.
  
  	p_nxt[( 0 * NX_d*NY_d) + idx_cur] = p_prv[ ( 0 * NX_d*NY_d) + idx_cur - 3*NY_d + 1 ];
  	p_nxt[( 1 * NX_d*NY_d) + idx_cur] = p_prv[ ( 1 * NX_d*NY_d) + idx_cur - 3*NY_d     ];
  	p_nxt[( 2 * NX_d*NY_d) + idx_cur] = p_prv[ ( 2 * NX_d*NY_d) + idx_cur - 3*NY_d - 1 ];
  	p_nxt[( 3 * NX_d*NY_d) + idx_cur] = p_prv[ ( 3 * NX_d*NY_d) + idx_cur - 2*NY_d + 2 ];
  	p_nxt[( 4 * NX_d*NY_d) + idx_cur] = p_prv[ ( 4 * NX_d*NY_d) + idx_cur - 2*NY_d + 1 ];
  	p_nxt[( 5 * NX_d*NY_d) + idx_cur] = p_prv[ ( 5 * NX_d*NY_d) + idx_cur - 2*NY_d     ];
  	p_nxt[( 6 * NX_d*NY_d) + idx_cur] = p_prv[ ( 6 * NX_d*NY_d) + idx_cur - 2*NY_d - 1 ];
  	p_nxt[( 7 * NX_d*NY_d) + idx_cur] = p_prv[ ( 7 * NX_d*NY_d) + idx_cur - 2*NY_d - 2 ];
  	p_nxt[( 8 * NX_d*NY_d) + idx_cur] = p_prv[ ( 8 * NX_d*NY_d) + idx_cur -   NY_d + 3 ];
  	p_nxt[( 9 * NX_d*NY_d) + idx_cur] = p_prv[ ( 9 * NX_d*NY_d) + idx_cur -   NY_d + 2 ];
  	p_nxt[(10 * NX_d*NY_d) + idx_cur] = p_prv[ (10 * NX_d*NY_d) + idx_cur -   NY_d + 1 ];
  	p_nxt[(11 * NX_d*NY_d) + idx_cur] = p_prv[ (11 * NX_d*NY_d) + idx_cur -   NY_d     ];
  	p_nxt[(12 * NX_d*NY_d) + idx_cur] = p_prv[ (12 * NX_d*NY_d) + idx_cur -   NY_d - 1 ];
  	p_nxt[(13 * NX_d*NY_d) + idx_cur] = p_prv[ (13 * NX_d*NY_d) + idx_cur -   NY_d - 2 ];
  	p_nxt[(14 * NX_d*NY_d) + idx_cur] = p_prv[ (14 * NX_d*NY_d) + idx_cur -   NY_d - 3 ];
  	p_nxt[(15 * NX_d*NY_d) + idx_cur] = p_prv[ (15 * NX_d*NY_d) + idx_cur        + 3 ];
  	p_nxt[(16 * NX_d*NY_d) + idx_cur] = p_prv[ (16 * NX_d*NY_d) + idx_cur        + 2 ];
  	p_nxt[(17 * NX_d*NY_d) + idx_cur] = p_prv[ (17 * NX_d*NY_d) + idx_cur        + 1 ];
  	p_nxt[(18 * NX_d*NY_d) + idx_cur] = p_prv[ (18 * NX_d*NY_d) + idx_cur	         ];
  	p_nxt[(19 * NX_d*NY_d) + idx_cur] = p_prv[ (19 * NX_d*NY_d) + idx_cur        - 1 ];
  	p_nxt[(20 * NX_d*NY_d) + idx_cur] = p_prv[ (20 * NX_d*NY_d) + idx_cur        - 2 ];
  	p_nxt[(21 * NX_d*NY_d) + idx_cur] = p_prv[ (21 * NX_d*NY_d) + idx_cur        - 3 ];
  	p_nxt[(22 * NX_d*NY_d) + idx_cur] = p_prv[ (22 * NX_d*NY_d) + idx_cur +   NY_d + 3 ];
  	p_nxt[(23 * NX_d*NY_d) + idx_cur] = p_prv[ (23 * NX_d*NY_d) + idx_cur +   NY_d + 2 ];
  	p_nxt[(24 * NX_d*NY_d) + idx_cur] = p_prv[ (24 * NX_d*NY_d) + idx_cur +   NY_d + 1 ];
  	p_nxt[(25 * NX_d*NY_d) + idx_cur] = p_prv[ (25 * NX_d*NY_d) + idx_cur +   NY_d     ];
  	p_nxt[(26 * NX_d*NY_d) + idx_cur] = p_prv[ (26 * NX_d*NY_d) + idx_cur +   NY_d - 1 ];
  	p_nxt[(27 * NX_d*NY_d) + idx_cur] = p_prv[ (27 * NX_d*NY_d) + idx_cur +   NY_d - 2 ];
  	p_nxt[(28 * NX_d*NY_d) + idx_cur] = p_prv[ (28 * NX_d*NY_d) + idx_cur +   NY_d - 3 ];
  	p_nxt[(29 * NX_d*NY_d) + idx_cur] = p_prv[ (29 * NX_d*NY_d) + idx_cur + 2*NY_d + 2 ];
  	p_nxt[(30 * NX_d*NY_d) + idx_cur] = p_prv[ (30 * NX_d*NY_d) + idx_cur + 2*NY_d + 1 ];
  	p_nxt[(31 * NX_d*NY_d) + idx_cur] = p_prv[ (31 * NX_d*NY_d) + idx_cur + 2*NY_d     ];
  	p_nxt[(32 * NX_d*NY_d) + idx_cur] = p_prv[ (32 * NX_d*NY_d) + idx_cur + 2*NY_d - 1 ];
  	p_nxt[(33 * NX_d*NY_d) + idx_cur] = p_prv[ (33 * NX_d*NY_d) + idx_cur + 2*NY_d - 2 ];
  	p_nxt[(34 * NX_d*NY_d) + idx_cur] = p_prv[ (34 * NX_d*NY_d) + idx_cur + 3*NY_d + 1 ];
  	p_nxt[(35 * NX_d*NY_d) + idx_cur] = p_prv[ (35 * NX_d*NY_d) + idx_cur + 3*NY_d     ];
  	p_nxt[(36 * NX_d*NY_d) + idx_cur] = p_prv[ (36 * NX_d*NY_d) + idx_cur + 3*NY_d - 1 ];
  	/////////////////////////////////////////////////////////////////////////////

}

__global__ void propagateT (
      data_t * __restrict__ const p_nxt,
      const data_t * __restrict__ const p_prv
    ) {
      long idx_cur;
     idx_cur =  ( blockIdx.y  * blockDim.y * NY_d ) +   // Offset columns.
                ( blockIdx.x  * blockDim.x      ) +   // Offset Y block.
                ( threadIdx.y * NY_d              ) +   // Index of Y thread.
                ( threadIdx.x                   );    // Index of X thread.

      if( (threadIdx.x < blockDim.x) && ((blockIdx.y*blockDim.y+threadIdx.y) < LSIZEX_d) )
        propagate(p_nxt, p_prv, idx_cur );
    }

__global__ void propagateB (
      data_t * __restrict__ const p_nxt,
      const data_t * __restrict__ const p_prv
    ) {
   
      long idx_cur;
      idx_cur =  ( blockIdx.y  * blockDim.y * NY_d ) +   // Offset columns.
                 ( blockIdx.x  * blockDim.x      ) +   // Offset Y block.
                 ( threadIdx.y * NY_d              ) +   // Index of Y thread.
                 ( threadIdx.x                   );    // Index of X thread.

       if( (threadIdx.x < blockDim.x) && ((blockIdx.y*blockDim.y+threadIdx.y) < LSIZEX_d) )
         propagate(p_nxt, p_prv, idx_cur );
    }

#endif

