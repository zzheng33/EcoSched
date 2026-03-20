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

#ifndef _BC_CUDA_H
#define _BC_CUDA_H

////////////////////////////////////////////////////////////////////////////////

#define _COMPUTE_U_V_RHO_TEMP                   \
  _COMPUTE_U_V_RHO( 0)                          \
  _COMPUTE_U_V_RHO( 1)                          \
  _COMPUTE_U_V_RHO( 2)                          \
  _COMPUTE_U_V_RHO( 3)                          \
  _COMPUTE_U_V_RHO( 4)                          \
  _COMPUTE_U_V_RHO( 5)                          \
  _COMPUTE_U_V_RHO( 6)                          \
  _COMPUTE_U_V_RHO( 7)                          \
  _COMPUTE_U_V_RHO( 8)                          \
  _COMPUTE_U_V_RHO( 9)                          \
  _COMPUTE_U_V_RHO(10)                          \
  _COMPUTE_U_V_RHO(11)                          \
  _COMPUTE_U_V_RHO(12)                          \
  _COMPUTE_U_V_RHO(13)                          \
  _COMPUTE_U_V_RHO(14)                          \
  _COMPUTE_U_V_RHO(15)                          \
  _COMPUTE_U_V_RHO(16)                          \
  _COMPUTE_U_V_RHO(17)                          \
  _COMPUTE_U_V_RHO(18)                          \
  _COMPUTE_U_V_RHO(19)                          \
  _COMPUTE_U_V_RHO(20)                          \
  _COMPUTE_U_V_RHO(21)                          \
  _COMPUTE_U_V_RHO(22)                          \
  _COMPUTE_U_V_RHO(23)                          \
  _COMPUTE_U_V_RHO(24)                          \
  _COMPUTE_U_V_RHO(25)                          \
  _COMPUTE_U_V_RHO(26)                          \
  _COMPUTE_U_V_RHO(27)                          \
  _COMPUTE_U_V_RHO(28)                          \
  _COMPUTE_U_V_RHO(29)                          \
  _COMPUTE_U_V_RHO(30)                          \
  _COMPUTE_U_V_RHO(31)                          \
  _COMPUTE_U_V_RHO(32)                          \
  _COMPUTE_U_V_RHO(33)                          \
  _COMPUTE_U_V_RHO(34)                          \
  _COMPUTE_U_V_RHO(35)                          \
  _COMPUTE_U_V_RHO(36)                          \
                                                \
  forcey = GRAVITY * DELTAT * rho;              \
                                                \
  rhoi  = 1.0 / rho;                            \
  u     = u * rhoi;                             \
  v     = v * rhoi;                             \
                                                \
  _COMPUTE_SCALAR_TEMP_BC( 0)                   \
  _COMPUTE_SCALAR_TEMP_BC( 1)                   \
  _COMPUTE_SCALAR_TEMP_BC( 2)                   \
  _COMPUTE_SCALAR_TEMP_BC( 3)                   \
  _COMPUTE_SCALAR_TEMP_BC( 4)                   \
  _COMPUTE_SCALAR_TEMP_BC( 5)                   \
  _COMPUTE_SCALAR_TEMP_BC( 6)                   \
  _COMPUTE_SCALAR_TEMP_BC( 7)                   \
  _COMPUTE_SCALAR_TEMP_BC( 8)                   \
  _COMPUTE_SCALAR_TEMP_BC( 9)                   \
  _COMPUTE_SCALAR_TEMP_BC(10)                   \
  _COMPUTE_SCALAR_TEMP_BC(11)                   \
  _COMPUTE_SCALAR_TEMP_BC(12)                   \
  _COMPUTE_SCALAR_TEMP_BC(13)                   \
  _COMPUTE_SCALAR_TEMP_BC(14)                   \
  _COMPUTE_SCALAR_TEMP_BC(15)                   \
  _COMPUTE_SCALAR_TEMP_BC(16)                   \
  _COMPUTE_SCALAR_TEMP_BC(17)                   \
  _COMPUTE_SCALAR_TEMP_BC(18)                   \
  _COMPUTE_SCALAR_TEMP_BC(19)                   \
  _COMPUTE_SCALAR_TEMP_BC(20)                   \
  _COMPUTE_SCALAR_TEMP_BC(21)                   \
  _COMPUTE_SCALAR_TEMP_BC(22)                   \
  _COMPUTE_SCALAR_TEMP_BC(23)                   \
  _COMPUTE_SCALAR_TEMP_BC(24)                   \
  _COMPUTE_SCALAR_TEMP_BC(25)                   \
  _COMPUTE_SCALAR_TEMP_BC(26)                   \
  _COMPUTE_SCALAR_TEMP_BC(27)                   \
  _COMPUTE_SCALAR_TEMP_BC(28)                   \
  _COMPUTE_SCALAR_TEMP_BC(29)                   \
  _COMPUTE_SCALAR_TEMP_BC(30)                   \
  _COMPUTE_SCALAR_TEMP_BC(31)                   \
  _COMPUTE_SCALAR_TEMP_BC(32)                   \
  _COMPUTE_SCALAR_TEMP_BC(33)                   \
  _COMPUTE_SCALAR_TEMP_BC(34)                   \
  _COMPUTE_SCALAR_TEMP_BC(35)                   \
  _COMPUTE_SCALAR_TEMP_BC(36)                   \
                                                \
  temp = temp * rhoi;

////////////////////////////////////////////////////////////////////////////////

#define _COMPUTE_U_V_RHO(_k)                    \
  rho = rho + localPop[ _k ];                   \
  u  = u   + param_cx[ _k ] * localPop[ _k ];   \
  v  = v   + param_cy[ _k ] * localPop[ _k ];

////////////////////////////////////////////////////////////////////////////////
// It is the same of _COMPUTE_SCALAR_TEMP.
#define _COMPUTE_SCALAR_TEMP_BC(_k)                           \
  scalar =  (param_cx[ _k ] - u) * (param_cx[ _k ] - u) +     \
            (param_cy[ _k ] - v) * (param_cy[ _k ] - v);      \
  temp = temp + 0.5*scalar*localPop[ _k ];   

////////////////////////////////////////////////////////////////////////////////

#define _polint(xa,ya)                              \
                                                    \
  int i,m,ns=1;                                     \
  float den,dif,dift,ho,hp,w;                       \
  float y,dy,x;                                     \
  float c[ENNE+2] ;                                 \
  float d[ENNE+2] ;                                 \
                                                    \
  x=xa[0];                                          \
                                                    \
  dif=fabs(x-xa[1]);                                \
                                                    \
  for (i=1;i<=ENNE;i++) {                           \
    if ( (dift=fabs(x-xa[i])) < dif) {              \
      ns=i;                                         \
      dif=dift;                                     \
    }                                               \
    c[i]=ya[i];                                     \
    d[i]=ya[i];                                     \
  }                                                 \
                                                    \
  y=ya[ns--];                                       \
  for (m=1;m<ENNE;m++) {                            \
    for (i=1;i<=ENNE-m;i++) {                       \
      ho=xa[i]-x;                                   \
      hp=xa[i+m]-x;                                 \
      w=c[i+1]-d[i];                                \
      den=ho-hp;                                    \
      den=w/den;                                    \
      d[i]=hp*den;                                  \
      c[i]=ho*den;                                  \
    }                                               \
    y += (dy=(2*ns < (ENNE-m) ? c[ns+1] : d[ns--]));\
  }                                                 \
                                                    \
  ya[0]=y;

////////////////////////////////////////////////////////////////////////////////
// Global function BC.
__global__ void bcB (
      data_t *p_nxt,
      data_t *p_prv
    ) {
  
  long idx_cur;       // Index of current site.
  
  data_t localPop[NPOP];  // Array where local populations are stored 
                          // during operations.
    
  //////////////////////////////////////////////////////////////////////////////

  data_t a1,a2,a3;
  data_t b1,b2;
  data_t c1,c2,c3;
  data_t d1,d2,d3;
  
  data_t utilde1,vtilde1;
  data_t utilde2,vtilde2;
  data_t utilde3;
  
  data_t term1,term2;
  
  data_t temptilde1,temptilde2,temptilde3;
  data_t localTempfic;
  
  data_t px,py,E,S,N,Ox,Oy,OE;
  data_t ptildex,ptildey,Etilde,massapost;
  
#ifndef ADIABATIC
  data_t xa[ENNE+2], ya[ENNE+2];
#endif
  
  data_t rhoi, forcey, scalar;
  
  long idx3;
  
  data_t phi[NPOP];
  data_t fPre[NPOP], fPost[NPOP];
  
  data_t  rho, temp, u, v;
    
  ////////////////////////////////////////////////////////////////////////////
  // Set index of current site.      
  idx_cur = blockIdx.y*blockDim.y*NY_d + threadIdx.y*NY_d + threadIdx.x;    

  // This condition is set to block final threads when LSIZEX_d is not divisible by BC_BLOCK_DIMY
  if (idx_cur < LSIZEX_d*NY_d) {

    ////////////////////////////////////////////////////////////////////////////
    // Load populations of prv lattice in localPop array.
    
    fPre[ 0] = p_prv[ ( 0 * NX_d*NY_d) + idx_cur ];
    fPre[ 1] = p_prv[ ( 1 * NX_d*NY_d) + idx_cur ];
    fPre[ 2] = p_prv[ ( 2 * NX_d*NY_d) + idx_cur ];
    fPre[ 3] = p_prv[ ( 3 * NX_d*NY_d) + idx_cur ];
    fPre[ 4] = p_prv[ ( 4 * NX_d*NY_d) + idx_cur ];
    fPre[ 5] = p_prv[ ( 5 * NX_d*NY_d) + idx_cur ];
    fPre[ 6] = p_prv[ ( 6 * NX_d*NY_d) + idx_cur ];
    fPre[ 7] = p_prv[ ( 7 * NX_d*NY_d) + idx_cur ];
    fPre[ 8] = p_prv[ ( 8 * NX_d*NY_d) + idx_cur ];
    fPre[ 9] = p_prv[ ( 9 * NX_d*NY_d) + idx_cur ];
    fPre[10] = p_prv[ (10 * NX_d*NY_d) + idx_cur ];
    fPre[11] = p_prv[ (11 * NX_d*NY_d) + idx_cur ];
    fPre[12] = p_prv[ (12 * NX_d*NY_d) + idx_cur ];
    fPre[13] = p_prv[ (13 * NX_d*NY_d) + idx_cur ];
    fPre[14] = p_prv[ (14 * NX_d*NY_d) + idx_cur ];
    fPre[15] = p_prv[ (15 * NX_d*NY_d) + idx_cur ];
    fPre[16] = p_prv[ (16 * NX_d*NY_d) + idx_cur ];
    fPre[17] = p_prv[ (17 * NX_d*NY_d) + idx_cur ];
    fPre[18] = p_prv[ (18 * NX_d*NY_d) + idx_cur ];
    fPre[19] = p_prv[ (19 * NX_d*NY_d) + idx_cur ];
    fPre[20] = p_prv[ (20 * NX_d*NY_d) + idx_cur ];
    fPre[21] = p_prv[ (21 * NX_d*NY_d) + idx_cur ];
    fPre[22] = p_prv[ (22 * NX_d*NY_d) + idx_cur ];
    fPre[23] = p_prv[ (23 * NX_d*NY_d) + idx_cur ];
    fPre[24] = p_prv[ (24 * NX_d*NY_d) + idx_cur ];
    fPre[25] = p_prv[ (25 * NX_d*NY_d) + idx_cur ];
    fPre[26] = p_prv[ (26 * NX_d*NY_d) + idx_cur ];
    fPre[27] = p_prv[ (27 * NX_d*NY_d) + idx_cur ];
    fPre[28] = p_prv[ (28 * NX_d*NY_d) + idx_cur ];
    fPre[29] = p_prv[ (29 * NX_d*NY_d) + idx_cur ];
    fPre[30] = p_prv[ (30 * NX_d*NY_d) + idx_cur ];
    fPre[31] = p_prv[ (31 * NX_d*NY_d) + idx_cur ];
    fPre[32] = p_prv[ (32 * NX_d*NY_d) + idx_cur ];
    fPre[33] = p_prv[ (33 * NX_d*NY_d) + idx_cur ];
    fPre[34] = p_prv[ (34 * NX_d*NY_d) + idx_cur ];
    fPre[35] = p_prv[ (35 * NX_d*NY_d) + idx_cur ];
    fPre[36] = p_prv[ (36 * NX_d*NY_d) + idx_cur ];
      
    ////////////////////////////////////////////////////////////////////////////
    // Load populations of succ lattice in localPop array.
    
    fPost[ 0] = p_nxt[ ( 0 * NX_d*NY_d) + idx_cur ];
    fPost[ 1] = p_nxt[ ( 1 * NX_d*NY_d) + idx_cur ];
    fPost[ 2] = p_nxt[ ( 2 * NX_d*NY_d) + idx_cur ];
    fPost[ 3] = p_nxt[ ( 3 * NX_d*NY_d) + idx_cur ];
    fPost[ 4] = p_nxt[ ( 4 * NX_d*NY_d) + idx_cur ];
    fPost[ 5] = p_nxt[ ( 5 * NX_d*NY_d) + idx_cur ];
    fPost[ 6] = p_nxt[ ( 6 * NX_d*NY_d) + idx_cur ];
    fPost[ 7] = p_nxt[ ( 7 * NX_d*NY_d) + idx_cur ];
    fPost[ 8] = p_nxt[ ( 8 * NX_d*NY_d) + idx_cur ];
    fPost[ 9] = p_nxt[ ( 9 * NX_d*NY_d) + idx_cur ];
    fPost[10] = p_nxt[ (10 * NX_d*NY_d) + idx_cur ];
    fPost[11] = p_nxt[ (11 * NX_d*NY_d) + idx_cur ];
    fPost[12] = p_nxt[ (12 * NX_d*NY_d) + idx_cur ];
    fPost[13] = p_nxt[ (13 * NX_d*NY_d) + idx_cur ];
    fPost[14] = p_nxt[ (14 * NX_d*NY_d) + idx_cur ];
    fPost[15] = p_nxt[ (15 * NX_d*NY_d) + idx_cur ];
    fPost[16] = p_nxt[ (16 * NX_d*NY_d) + idx_cur ];
    fPost[17] = p_nxt[ (17 * NX_d*NY_d) + idx_cur ];
    fPost[18] = p_nxt[ (18 * NX_d*NY_d) + idx_cur ];
    fPost[19] = p_nxt[ (19 * NX_d*NY_d) + idx_cur ];
    fPost[20] = p_nxt[ (20 * NX_d*NY_d) + idx_cur ];
    fPost[21] = p_nxt[ (21 * NX_d*NY_d) + idx_cur ];
    fPost[22] = p_nxt[ (22 * NX_d*NY_d) + idx_cur ];
    fPost[23] = p_nxt[ (23 * NX_d*NY_d) + idx_cur ];
    fPost[24] = p_nxt[ (24 * NX_d*NY_d) + idx_cur ];
    fPost[25] = p_nxt[ (25 * NX_d*NY_d) + idx_cur ];
    fPost[26] = p_nxt[ (26 * NX_d*NY_d) + idx_cur ];
    fPost[27] = p_nxt[ (27 * NX_d*NY_d) + idx_cur ];
    fPost[28] = p_nxt[ (28 * NX_d*NY_d) + idx_cur ];
    fPost[29] = p_nxt[ (29 * NX_d*NY_d) + idx_cur ];
    fPost[30] = p_nxt[ (30 * NX_d*NY_d) + idx_cur ];
    fPost[31] = p_nxt[ (31 * NX_d*NY_d) + idx_cur ];
    fPost[32] = p_nxt[ (32 * NX_d*NY_d) + idx_cur ];
    fPost[33] = p_nxt[ (33 * NX_d*NY_d) + idx_cur ];
    fPost[34] = p_nxt[ (34 * NX_d*NY_d) + idx_cur ];
    fPost[35] = p_nxt[ (35 * NX_d*NY_d) + idx_cur ];
    fPost[36] = p_nxt[ (36 * NX_d*NY_d) + idx_cur ];
      
    ////////////////////////////////////////////////////////////////////////////
    
    // Lower bord.
      
    idx3 = idx_cur - threadIdx.x + 3; // index of population in y=3
      
    //////////////////////////////////////////////////////////////////////////
    // Load populations in Y = 3 in localPop array.
      
    //---- FROM NOW ON everything is done on the local structure .....
    /////////PROCEDURE OF EXTRAPOLATION-INTERPOLATION////////////
    //temperature//
    //evaluating "tempfic[idx3]" in order to easily:
    // 1) allow ADIABATIC initialization;
    // 2) evaluating ya[2]
    //NOTE: tempfic[idx3] in theoriginal code is here 
    //replaced with "localTempfic"
    //
    //-- the line above to be replaced by the following set of equations
    //-- localPop is read from fPost[idx3]
    //-- THE SAME TO BE DONE FOR THE UPPER LAYER...
          
    localPop[ 0] = p_nxt[ ( 0 * NX_d*NY_d) + idx3 ];
    localPop[ 1] = p_nxt[ ( 1 * NX_d*NY_d) + idx3 ];
    localPop[ 2] = p_nxt[ ( 2 * NX_d*NY_d) + idx3 ];
    localPop[ 3] = p_nxt[ ( 3 * NX_d*NY_d) + idx3 ];
    localPop[ 4] = p_nxt[ ( 4 * NX_d*NY_d) + idx3 ];
    localPop[ 5] = p_nxt[ ( 5 * NX_d*NY_d) + idx3 ];
    localPop[ 6] = p_nxt[ ( 6 * NX_d*NY_d) + idx3 ];
    localPop[ 7] = p_nxt[ ( 7 * NX_d*NY_d) + idx3 ];
    localPop[ 8] = p_nxt[ ( 8 * NX_d*NY_d) + idx3 ];
    localPop[ 9] = p_nxt[ ( 9 * NX_d*NY_d) + idx3 ];
    localPop[10] = p_nxt[ (10 * NX_d*NY_d) + idx3 ];
    localPop[11] = p_nxt[ (11 * NX_d*NY_d) + idx3 ];
    localPop[12] = p_nxt[ (12 * NX_d*NY_d) + idx3 ];
    localPop[13] = p_nxt[ (13 * NX_d*NY_d) + idx3 ];
    localPop[14] = p_nxt[ (14 * NX_d*NY_d) + idx3 ];
    localPop[15] = p_nxt[ (15 * NX_d*NY_d) + idx3 ];
    localPop[16] = p_nxt[ (16 * NX_d*NY_d) + idx3 ];
    localPop[17] = p_nxt[ (17 * NX_d*NY_d) + idx3 ];
    localPop[18] = p_nxt[ (18 * NX_d*NY_d) + idx3 ];
    localPop[19] = p_nxt[ (19 * NX_d*NY_d) + idx3 ];
    localPop[20] = p_nxt[ (20 * NX_d*NY_d) + idx3 ];
    localPop[21] = p_nxt[ (21 * NX_d*NY_d) + idx3 ];
    localPop[22] = p_nxt[ (22 * NX_d*NY_d) + idx3 ];
    localPop[23] = p_nxt[ (23 * NX_d*NY_d) + idx3 ];
    localPop[24] = p_nxt[ (24 * NX_d*NY_d) + idx3 ];
    localPop[25] = p_nxt[ (25 * NX_d*NY_d) + idx3 ];
    localPop[26] = p_nxt[ (26 * NX_d*NY_d) + idx3 ];
    localPop[27] = p_nxt[ (27 * NX_d*NY_d) + idx3 ];
    localPop[28] = p_nxt[ (28 * NX_d*NY_d) + idx3 ];
    localPop[29] = p_nxt[ (29 * NX_d*NY_d) + idx3 ];
    localPop[30] = p_nxt[ (30 * NX_d*NY_d) + idx3 ];
    localPop[31] = p_nxt[ (31 * NX_d*NY_d) + idx3 ];
    localPop[32] = p_nxt[ (32 * NX_d*NY_d) + idx3 ];
    localPop[33] = p_nxt[ (33 * NX_d*NY_d) + idx3 ];
    localPop[34] = p_nxt[ (34 * NX_d*NY_d) + idx3 ];
    localPop[35] = p_nxt[ (35 * NX_d*NY_d) + idx3 ];
    localPop[36] = p_nxt[ (36 * NX_d*NY_d) + idx3 ];
    
    //////////////////////////////////////////////////////////////////////////
    // Compute local u, v, rho e temp.
      
    rho   = 0.0; 
    u     = 0.0; 
    v     = 0.0;
    temp  = 0.0;
     
    //Mmacro that compute u, v, rho, temp.
    _COMPUTE_U_V_RHO_TEMP
      
    localTempfic = temp + 
                   (forcex * forcex + forcey * forcey) * 
                   rhoi * rhoi / 8.0;
      
    // ---- end of replacement -----------------------------------------------
    //////////////////////////////////////////////////////////////////////////
      
#ifdef ADIABATIC
      
    temptilde1 = localTempfic ;
    temptilde2 = localTempfic ;
    temptilde3 = localTempfic ;
      
#else
      
    temptilde1 = TEMPWALLDOWN;
    temptilde2 = TEMPWALLDOWN;
    temptilde3 = TEMPWALLDOWN;
      
    xa[1] = 3.0;   
    xa[2] = 4.0;   
      
    ya[1] = temptilde3;  
    // In the original code here we found:
    //   ya[2]=tempfic[idx3];
    // the avaluation of tempfic is made above, so
    // we replace it with the following:
    ya[2] = localTempfic;
      
    // fprintf(stdout,"x: %03d %f\n", i, ya[2]);
     
    xa[0] = 2.0;//I want it here
    _polint(xa,ya); MANCA
    temptilde2 = ya[0];
      
    xa[0] = 1.0;//I want it here
    _polint(xa,ya); MANCA
    temptilde1 = ya[0];
      
#endif
      
    // X velocity.
    utilde1 = 0.0;
    utilde2 = 0.0;
    utilde3 = 0.0;
    /// Y velocity.
    vtilde1 = 0.0;
    vtilde2 = 0.0;
      
    if (threadIdx.x == 0) {

      ////////////////////LOWER LAYER 1 //////////////////////////////////////
        
      N = (fPre[15] + fPre[22] + fPre[ 8]) + (fPre[29] +
           fPre[23] + fPre[16] + fPre[ 9]  +  fPre[ 3])+
          (fPre[34] + fPre[30] + fPre[24]  +  fPre[17] +
           fPre[10] + fPre[ 4] + fPre[ 0]);
        
      massapost = N +
                  fPost[35] + fPost[31] + fPost[25] + fPost[11] +
                  fPost[ 5] + fPost[ 1] + fPost[34] + fPost[30] +
                  fPost[24] + fPost[17] + fPost[10] + fPost[ 4] +
                  fPost[ 0] + fPost[29] + fPost[23] + fPost[16] +
                  fPost[ 9] + fPost[ 3] + fPost[22] + fPost[15] +
                  fPost[ 8] + fPost[18];

      ptildex = massapost*(utilde1) - 0.5*FORCEPOIS*DELTAT;
      ptildey = massapost*(vtilde1) - 0.5*GRAVITY*massapost*DELTAT;

      Etilde  =  (temptilde1)*massapost +
                 0.5*(ptildex*ptildex+ptildey*ptildey)/massapost;


      Ox = param_cx[35] * fPost[35] + param_cx[31] * fPost[31] +
           param_cx[25] * fPost[25] + param_cx[11] * fPost[11] +
           param_cx[ 5] * fPost[ 5] + param_cx[ 1] * fPost[ 1] +
           param_cx[34] * fPost[34] + param_cx[30] * fPost[30] +
           param_cx[24] * fPost[24] + param_cx[17] * fPost[17] +
           param_cx[10] * fPost[10] + param_cx[ 4] * fPost[ 4] +
           param_cx[ 0] * fPost[ 0] + param_cx[29] * fPost[29] +
           param_cx[23] * fPost[23] + param_cx[16] * fPost[16] +
           param_cx[ 9] * fPost[ 9] + param_cx[ 3] * fPost[ 3] +
           param_cx[22] * fPost[22] + param_cx[15] * fPost[15] +
           param_cx[ 8] * fPost[ 8];

      Oy = param_cy[35] * fPost[35] + param_cy[31] * fPost[31] +
           param_cy[25] * fPost[25] + param_cy[11] * fPost[11] +
           param_cy[ 5] * fPost[ 5] + param_cy[ 1] * fPost[ 1] +
           param_cy[34] * fPost[34] + param_cy[30] * fPost[30] +
           param_cy[24] * fPost[24] + param_cy[17] * fPost[17] +
           param_cy[10] * fPost[10] + param_cy[ 4] * fPost[ 4] +
           param_cy[ 0] * fPost[ 0] + param_cy[29] * fPost[29] +
           param_cy[23] * fPost[23] + param_cy[16] * fPost[16] +
           param_cy[ 9] * fPost[ 9] + param_cy[ 3] * fPost[ 3] +
           param_cy[22] * fPost[22] + param_cy[15] * fPost[15] +
           param_cy[ 8] * fPost[ 8];
      
      OE = 0.5 * (param_cx[35]*param_cx[35]+param_cy[35]*param_cy[35]) * fPost[35] +
           0.5 * (param_cx[31]*param_cx[31]+param_cy[31]*param_cy[31]) * fPost[31] +
           0.5 * (param_cx[25]*param_cx[25]+param_cy[25]*param_cy[25]) * fPost[25] +
           0.5 * (param_cx[11]*param_cx[11]+param_cy[11]*param_cy[11]) * fPost[11] +
           0.5 * (param_cx[ 5]*param_cx[ 5]+param_cy[ 5]*param_cy[ 5]) * fPost[ 5] +
           0.5 * (param_cx[ 1]*param_cx[ 1]+param_cy[ 1]*param_cy[ 1]) * fPost[ 1] +
           0.5 * (param_cx[34]*param_cx[34]+param_cy[34]*param_cy[34]) * fPost[34] +
           0.5 * (param_cx[30]*param_cx[30]+param_cy[30]*param_cy[30]) * fPost[30] +
           0.5 * (param_cx[24]*param_cx[24]+param_cy[24]*param_cy[24]) * fPost[24] +
           0.5 * (param_cx[17]*param_cx[17]+param_cy[17]*param_cy[17]) * fPost[17] +
           0.5 * (param_cx[10]*param_cx[10]+param_cy[10]*param_cy[10]) * fPost[10] +
           0.5 * (param_cx[ 4]*param_cx[ 4]+param_cy[ 4]*param_cy[ 4]) * fPost[ 4] +
           0.5 * (param_cx[ 0]*param_cx[ 0]+param_cy[ 0]*param_cy[ 0]) * fPost[ 0] +
           0.5 * (param_cx[29]*param_cx[29]+param_cy[29]*param_cy[29]) * fPost[29] +
           0.5 * (param_cx[23]*param_cx[23]+param_cy[23]*param_cy[23]) * fPost[23] +
           0.5 * (param_cx[16]*param_cx[16]+param_cy[16]*param_cy[16]) * fPost[16] +
           0.5 * (param_cx[ 9]*param_cx[ 9]+param_cy[ 9]*param_cy[ 9]) * fPost[ 9] +
           0.5 * (param_cx[ 3]*param_cx[ 3]+param_cy[ 3]*param_cy[ 3]) * fPost[ 3] +
           0.5 * (param_cx[22]*param_cx[22]+param_cy[22]*param_cy[22]) * fPost[22] +
           0.5 * (param_cx[15]*param_cx[15]+param_cy[15]*param_cy[15]) * fPost[15] +
           0.5 * (param_cx[ 8]*param_cx[ 8]+param_cy[ 8]*param_cy[ 8]) * fPost[ 8];
    
      a1 = 26.*(ptildex-Ox)*UNIT;
      b1 =-40.*N*UNIT*UNIT;
      c1 = 47.*(ptildex-Ox)*UNIT*UNIT;
      d1 = 15.*(ptildex-Ox);

      a2 = 26.*(ptildey-Oy)*UNIT-54.*N*UNIT*UNIT;
      c2 = 47.*(ptildey-Oy)*UNIT*UNIT-91.*N*UNIT*UNIT*UNIT;
      d2 = 15.*(ptildey-Oy)-26.*N*UNIT;

      a3 = 26.*(Etilde-OE)*UNIT-91.*N*UNIT*UNIT*UNIT;
      c3 = 47.*(Etilde-OE)*UNIT*UNIT-N*UNIT*UNIT*UNIT*UNIT*367./2.;
      d3 = 15.*(Etilde-OE)-47.*N*UNIT*UNIT;

      py = (-c3*d2+c2*d3)/(-a3*c2+a2*c3);
      px = (a2*c3*d1-a2*c1*d3-a3*c2*d1-c3*a1*d2+c1*a3*d2+a1*c2*d3)/(b1*(a3*c2-a2*c3));

      E = (-a3*d2+a2*d3)/(a3*c2-a2*c3);
  
      phi[19] = 1.+param_cx[19]*px+param_cy[19]*py+0.5*(param_cx[19]*param_cx[19]+param_cy[19]*param_cy[19])*E;
      phi[20] = 1.+param_cx[20]*px+param_cy[20]*py+0.5*(param_cx[20]*param_cx[20]+param_cy[20]*param_cy[20])*E;
      phi[21] = 1.+param_cx[21]*px+param_cy[21]*py+0.5*(param_cx[21]*param_cx[21]+param_cy[21]*param_cy[21])*E;
      phi[12] = 1.+param_cx[12]*px+param_cy[12]*py+0.5*(param_cx[12]*param_cx[12]+param_cy[12]*param_cy[12])*E;
      phi[13] = 1.+param_cx[13]*px+param_cy[13]*py+0.5*(param_cx[13]*param_cx[13]+param_cy[13]*param_cy[13])*E;
      phi[14] = 1.+param_cx[14]*px+param_cy[14]*py+0.5*(param_cx[14]*param_cx[14]+param_cy[14]*param_cy[14])*E;
      phi[26] = 1.+param_cx[26]*px+param_cy[26]*py+0.5*(param_cx[26]*param_cx[26]+param_cy[26]*param_cy[26])*E;
      phi[27] = 1.+param_cx[27]*px+param_cy[27]*py+0.5*(param_cx[27]*param_cx[27]+param_cy[27]*param_cy[27])*E;
      phi[28] = 1.+param_cx[28]*px+param_cy[28]*py+0.5*(param_cx[28]*param_cx[28]+param_cy[28]*param_cy[28])*E;
      phi[ 7] = 1.+param_cx[ 7]*px+param_cy[ 7]*py+0.5*(param_cx[ 7]*param_cx[ 7]+param_cy[ 7]*param_cy[ 7])*E;
      phi[ 6] = 1.+param_cx[ 6]*px+param_cy[ 6]*py+0.5*(param_cx[ 6]*param_cx[ 6]+param_cy[ 6]*param_cy[ 6])*E;
      phi[33] = 1.+param_cx[33]*px+param_cy[33]*py+0.5*(param_cx[33]*param_cx[33]+param_cy[33]*param_cy[33])*E;
      phi[32] = 1.+param_cx[32]*px+param_cy[32]*py+0.5*(param_cx[32]*param_cx[32]+param_cy[32]*param_cy[32])*E;
      phi[ 2] = 1.+param_cx[ 2]*px+param_cy[ 2]*py+0.5*(param_cx[ 2]*param_cx[ 2]+param_cy[ 2]*param_cy[ 2])*E;
      phi[36] = 1.+param_cx[36]*px+param_cy[36]*py+0.5*(param_cx[36]*param_cx[36]+param_cy[36]*param_cy[36])*E;    

      S = phi[19] + phi[20] + phi[21] + phi[12] + phi[13] +
          phi[14] + phi[26] + phi[27] + phi[28] + phi[ 7] +
          phi[ 6] + phi[33] + phi[32] + phi[ 2] + phi[36];

      fPost[19] = phi[19]*N/S;
      fPost[20] = phi[20]*N/S;
      fPost[21] = phi[21]*N/S;

      fPost[12] = phi[12]*N/S;
      fPost[13] = phi[13]*N/S;
      fPost[14] = phi[14]*N/S;

      fPost[26] = phi[26]*N/S;
      fPost[27] = phi[27]*N/S;
      fPost[28] = phi[28]*N/S;

      fPost[ 7] = phi[ 7]*N/S;
      fPost[ 6] = phi[ 6]*N/S;

      fPost[33] = phi[33]*N/S;
      fPost[32] = phi[32]*N/S;

      fPost[ 2] = phi[ 2]*N/S;
      fPost[36] = phi[36]*N/S;

      //////////FINAL LOWER LAYER1////////////////////
    
    } // end if ( threadIdx.x==0 )
        
    else
      
      if (threadIdx.x == 1) {
  
        //////////LOWER LAYER 2/////////////////////////////////

        N = (fPre[15] + fPre[22] + fPre[ 8]) +
            (fPre[29] + fPre[23] + fPre[16] + fPre[ 9] + fPre[ 3]);

        massapost = N + 
                    fPost[36] + fPost[32] + fPost[26] +       
                    fPost[19] + fPost[12] + fPost[ 6] +
                    fPost[ 2] + fPost[35] + fPost[31] +
                    fPost[25] + fPost[11] + fPost[ 5] +
                    fPost[ 1] + fPost[34] + fPost[30] +
                    fPost[24] + fPost[17] + fPost[10] +
                    fPost[ 4] + fPost[ 0] + fPost[29] +
                    fPost[23] + fPost[16] + fPost[ 9] +
                    fPost[ 3] + fPost[22] + fPost[15] +
                    fPost[ 8] + fPost[18];

        ptildex = massapost*(utilde2) - 0.5*FORCEPOIS*DELTAT;
        ptildey = massapost*(vtilde2) - 0.5*GRAVITY*massapost*DELTAT;

        Etilde = (temptilde2)*massapost+
                 0.5*(ptildex*ptildex+ptildey*ptildey)/massapost;

        Ox = param_cx[36]*fPost[36] + param_cx[32]*fPost[32] +
             param_cx[26]*fPost[26] + param_cx[19]*fPost[19] +
             param_cx[12]*fPost[12] + param_cx[ 6]*fPost[ 6] +
             param_cx[ 2]*fPost[ 2] + param_cx[35]*fPost[35] +
             param_cx[31]*fPost[31] + param_cx[25]*fPost[25] +
             param_cx[11]*fPost[11] + param_cx[ 5]*fPost[ 5] +
             param_cx[ 1]*fPost[ 1] + param_cx[34]*fPost[34] +
             param_cx[30]*fPost[30] + param_cx[24]*fPost[24] +
             param_cx[17]*fPost[17] + param_cx[10]*fPost[10] +
             param_cx[ 4]*fPost[ 4] + param_cx[ 0]*fPost[ 0] +
             param_cx[29]*fPost[29] + param_cx[23]*fPost[23] +
             param_cx[16]*fPost[16] + param_cx[ 9]*fPost[ 9] +
             param_cx[ 3]*fPost[ 3] + param_cx[22]*fPost[22] +
             param_cx[15]*fPost[15] + param_cx[ 8]*fPost[ 8];
  
        Oy = param_cy[36]*fPost[36] + param_cy[32]*fPost[32] +
             param_cy[26]*fPost[26] + param_cy[19]*fPost[19] +
             param_cy[12]*fPost[12] + param_cy[ 6]*fPost[ 6] +
             param_cy[ 2]*fPost[ 2] + param_cy[35]*fPost[35] +
             param_cy[31]*fPost[31] + param_cy[25]*fPost[25] +
             param_cy[11]*fPost[11] + param_cy[ 5]*fPost[ 5] +
             param_cy[ 1]*fPost[ 1] + param_cy[34]*fPost[34] +
             param_cy[30]*fPost[30] + param_cy[24]*fPost[24] +
             param_cy[17]*fPost[17] + param_cy[10]*fPost[10] +
             param_cy[ 4]*fPost[ 4] + param_cy[ 0]*fPost[ 0] +
             param_cy[29]*fPost[29] + param_cy[23]*fPost[23] +
             param_cy[16]*fPost[16] + param_cy[ 9]*fPost[ 9] +
             param_cy[ 3]*fPost[ 3] + param_cy[22]*fPost[22] +
             param_cy[15]*fPost[15] + param_cy[ 8]*fPost[ 8];

        OE = 0.5*(param_cx[36]*param_cx[36] + param_cy[36]*param_cy[36])*fPost[36] +
             0.5*(param_cx[32]*param_cx[32] + param_cy[32]*param_cy[32])*fPost[32] +
             0.5*(param_cx[26]*param_cx[26] + param_cy[26]*param_cy[26])*fPost[26] +
             0.5*(param_cx[19]*param_cx[19] + param_cy[19]*param_cy[19])*fPost[19] +
             0.5*(param_cx[12]*param_cx[12] + param_cy[12]*param_cy[12])*fPost[12] +
             0.5*(param_cx[ 6]*param_cx[ 6] + param_cy[ 6]*param_cy[ 6])*fPost[ 6] +
             0.5*(param_cx[ 2]*param_cx[ 2] + param_cy[ 2]*param_cy[ 2])*fPost[ 2] +
             0.5*(param_cx[35]*param_cx[35] + param_cy[35]*param_cy[35])*fPost[35] +
             0.5*(param_cx[31]*param_cx[31] + param_cy[31]*param_cy[31])*fPost[31] +
             0.5*(param_cx[25]*param_cx[25] + param_cy[25]*param_cy[25])*fPost[25] +
             0.5*(param_cx[11]*param_cx[11] + param_cy[11]*param_cy[11])*fPost[11] +
             0.5*(param_cx[ 5]*param_cx[ 5] + param_cy[ 5]*param_cy[ 5])*fPost[ 5] +
             0.5*(param_cx[ 1]*param_cx[ 1] + param_cy[ 1]*param_cy[ 1])*fPost[ 1] +
             0.5*(param_cx[34]*param_cx[34] + param_cy[34]*param_cy[34])*fPost[34] +
             0.5*(param_cx[30]*param_cx[30] + param_cy[30]*param_cy[30])*fPost[30] +
             0.5*(param_cx[24]*param_cx[24] + param_cy[24]*param_cy[24])*fPost[24] +
             0.5*(param_cx[17]*param_cx[17] + param_cy[17]*param_cy[17])*fPost[17] + 
             0.5*(param_cx[10]*param_cx[10] + param_cy[10]*param_cy[10])*fPost[10] +
             0.5*(param_cx[ 4]*param_cx[ 4] + param_cy[ 4]*param_cy[ 4])*fPost[ 4] +
             0.5*(param_cx[ 0]*param_cx[ 0] + param_cy[ 0]*param_cy[ 0])*fPost[ 0] +
             0.5*(param_cx[29]*param_cx[29] + param_cy[29]*param_cy[29])*fPost[29] +
             0.5*(param_cx[23]*param_cx[23] + param_cy[23]*param_cy[23])*fPost[23] +
             0.5*(param_cx[16]*param_cx[16] + param_cy[16]*param_cy[16])*fPost[16] +
             0.5*(param_cx[ 9]*param_cx[ 9] + param_cy[ 9]*param_cy[ 9])*fPost[ 9] +
             0.5*(param_cx[ 3]*param_cx[ 3] + param_cy[ 3]*param_cy[ 3])*fPost[ 3] +
             0.5*(param_cx[22]*param_cx[22] + param_cy[22]*param_cy[22])*fPost[22] +
             0.5*(param_cx[15]*param_cx[15] + param_cy[15]*param_cy[15])*fPost[15] +
             0.5*(param_cx[ 8]*param_cx[ 8] + param_cy[ 8]*param_cy[ 8])*fPost[ 8] ;

        a1 = 19.*(ptildex-Ox)*UNIT;
        b1 = -12.*N*UNIT*UNIT;
        c1 = (59/2.)*(ptildex-Ox)*UNIT*UNIT;
        d1 = 8.*(ptildex-Ox);

        a2 = 19.*(ptildey-Oy)*UNIT-47.*N*UNIT*UNIT;
        c2 = (59/2.)*(ptildey-Oy)*UNIT*UNIT-(147./2.)*N*UNIT*UNIT*UNIT;
        d2 = 8.*(ptildey-Oy)-19.*N*UNIT;

        a3 = 19.*(Etilde-OE)*UNIT-(147./2.)*N*UNIT*UNIT*UNIT;
        c3 = (59./2.)*(Etilde-OE)*UNIT*UNIT-N*UNIT*UNIT*UNIT*UNIT*475./4.;
        d3 = 8.*(Etilde-OE)-(59./2.)*N*UNIT*UNIT;

        py = (-c3*d2+c2*d3)/(-a3*c2+a2*c3);

        px = (a2*c3*d1-a2*c1*d3-a3*c2*d1-c3*a1*d2+c1*a3*d2+a1*c2*d3)/(b1*(a3*c2-a2*c3));

        E  = (-a3*d2+a2*d3)/(a3*c2-a2*c3);

        phi[20] = 1.+param_cx[20]*px+param_cy[20]*py+0.5*(param_cx[20]*param_cx[20]+param_cy[20]*param_cy[20])*E;
        phi[21] = 1.+param_cx[21]*px+param_cy[21]*py+0.5*(param_cx[21]*param_cx[21]+param_cy[21]*param_cy[21])*E;
        phi[13] = 1.+param_cx[13]*px+param_cy[13]*py+0.5*(param_cx[13]*param_cx[13]+param_cy[13]*param_cy[13])*E;
        phi[14] = 1.+param_cx[14]*px+param_cy[14]*py+0.5*(param_cx[14]*param_cx[14]+param_cy[14]*param_cy[14])*E;
        phi[27] = 1.+param_cx[27]*px+param_cy[27]*py+0.5*(param_cx[27]*param_cx[27]+param_cy[27]*param_cy[27])*E;
        phi[28] = 1.+param_cx[28]*px+param_cy[28]*py+0.5*(param_cx[28]*param_cx[28]+param_cy[28]*param_cy[28])*E;
        phi[ 7] = 1.+param_cx[ 7]*px+param_cy[ 7]*py+0.5*(param_cx[ 7]*param_cx[ 7]+param_cy[ 7]*param_cy[ 7])*E;
        phi[33] = 1.+param_cx[33]*px+param_cy[33]*py+0.5*(param_cx[33]*param_cx[33]+param_cy[33]*param_cy[33])*E;

        S = phi[20] + phi[21] + phi[13] + phi[14] + 
            phi[27] + phi[28] + phi[ 7] + phi[33];
 
        fPost[20] = phi[20]*N/S;  
        fPost[21] = phi[21]*N/S;  
          
        fPost[13] = phi[13]*N/S;
        fPost[14] = phi[14]*N/S;  
          
        fPost[27] = phi[27]*N/S;  
        fPost[28] = phi[28]*N/S;

        fPost[ 7] = phi[ 7]*N/S;  

        fPost[33] = phi[33]*N/S;  

        ////////////FINE LOWER LAYER 2////////////////////////
    
      } // end if ( threadIdx.x==1 )
      
      else { //if (threadIdx.x == 2)
  
        //////////LOWER LAYER 3/////////////////////////////////

        //N=(f2[idx2].p[15]+f2[idx2].p[22]+f2[idx2].p[ 8]); new

        term1 = param_cy[33]*fPost[33] + param_cy[27]*fPost[27] +
                param_cy[20]*fPost[20] + param_cy[13]*fPost[13] +
                param_cy[ 7]*fPost[ 7] + param_cy[36]*fPost[36] +
                param_cy[32]*fPost[32] + param_cy[26]*fPost[26] +
                param_cy[19]*fPost[19] + param_cy[12]*fPost[12] +
                param_cy[ 6]*fPost[ 6] + param_cy[ 2]*fPost[ 2] +
                param_cy[35]*fPost[35] + param_cy[31]*fPost[31] +
                param_cy[25]*fPost[25] + param_cy[11]*fPost[11] +
                param_cy[ 5]*fPost[ 5] + param_cy[ 1]*fPost[ 1] +
                param_cy[34]*fPost[34] + param_cy[30]*fPost[30] +
                param_cy[24]*fPost[24] + param_cy[17]*fPost[17] +
                param_cy[10]*fPost[10] + param_cy[ 4]*fPost[ 4] +
                param_cy[ 0]*fPost[ 0] + param_cy[29]*fPost[29] +
                param_cy[23]*fPost[23] + param_cy[16]*fPost[16] +
                param_cy[ 9]*fPost[ 9] + param_cy[ 3]*fPost[ 3] +
                param_cy[22]*fPost[22] + param_cy[15]*fPost[15] +
                param_cy[ 8]*fPost[ 8];

        term2 = fPost[33] + fPost[27] +
                fPost[20] + fPost[13] +
                fPost[ 7] + fPost[36] +
                fPost[32] + fPost[26] +
                fPost[19] + fPost[12] +
                fPost[ 6] + fPost[ 2] +
                fPost[35] + fPost[31] +
                fPost[25] + fPost[11] +
                fPost[ 5] + fPost[ 1] +
                fPost[34] + fPost[30] +
                fPost[24] + fPost[17] +
                fPost[10] + fPost[ 4] +
                fPost[ 0] + fPost[29] +
                fPost[23] + fPost[16] +
                fPost[ 9] + fPost[ 3] +
                fPost[22] + fPost[15] +
                fPost[ 8] + fPost[18] +
                (fPre[15] +  fPre[22] + fPre[ 8]);
  
        N = -term1/(3.*UNIT)-0.5*GRAVITY*DELTAT*(term2)/(3.*UNIT);

        fPost[18] = fPost[18] - N + (fPre[15]+fPre[22]+fPre[ 8]);  
    
    
        massapost = N + 
                    fPost[33] + fPost[27] +
                    fPost[20] + fPost[13] +
                    fPost[ 7] + fPost[36] +
                    fPost[32] + fPost[26] +
                    fPost[19] + fPost[12] +
                    fPost[ 6] + fPost[ 2] +
                    fPost[35] + fPost[31] +
                    fPost[25] + fPost[11] +
                    fPost[ 5] + fPost[ 1] +
                    fPost[34] + fPost[30] +
                    fPost[24] + fPost[17] +
                    fPost[10] + fPost[ 4] +
                    fPost[ 0] + fPost[29] +
                    fPost[23] + fPost[16] +
                    fPost[ 9] + fPost[ 3] +
                    fPost[22] + fPost[15] +
                    fPost[ 8] + fPost[18];
                  
        ptildex = massapost*(utilde3)-0.5*FORCEPOIS*DELTAT;

        ptildey = 3. * N * UNIT + 
                  param_cy[33]*fPost[33] + param_cy[27]*fPost[27] +
                  param_cy[20]*fPost[20] + param_cy[13]*fPost[13] +
                  param_cy[ 7]*fPost[ 7] + param_cy[36]*fPost[36] +
                  param_cy[32]*fPost[32] + param_cy[26]*fPost[26] +
                  param_cy[19]*fPost[19] + param_cy[12]*fPost[12] +
                  param_cy[ 6]*fPost[ 6] + param_cy[ 2]*fPost[ 2] +
                  param_cy[35]*fPost[35] + param_cy[31]*fPost[31] +
                  param_cy[25]*fPost[25] + param_cy[11]*fPost[11] +
                  param_cy[ 5]*fPost[ 5] + param_cy[ 1]*fPost[ 1] +
                  param_cy[34]*fPost[34] + param_cy[30]*fPost[30] +
                  param_cy[24]*fPost[24] + param_cy[17]*fPost[17] +
                  param_cy[10]*fPost[10] + param_cy[ 4]*fPost[ 4] +
                  param_cy[ 0]*fPost[ 0] + param_cy[29]*fPost[29] +
                  param_cy[23]*fPost[23] + param_cy[16]*fPost[16] +
                  param_cy[ 9]*fPost[ 9] + param_cy[ 3]*fPost[ 3] +
                  param_cy[22]*fPost[22] + param_cy[15]*fPost[15] +
                  param_cy[ 8]*fPost[ 8];

        Etilde = (temptilde3)*massapost +
                 0.5*(ptildex*ptildex + ptildey*ptildey)/massapost;
  
        Ox = param_cx[33]*fPost[33] + param_cx[27]*fPost[27] +
             param_cx[20]*fPost[20] + param_cx[13]*fPost[13] +
             param_cx[ 7]*fPost[ 7] + param_cx[36]*fPost[36] +
             param_cx[32]*fPost[32] + param_cx[26]*fPost[26] +
             param_cx[19]*fPost[19] + param_cx[12]*fPost[12] +
             param_cx[ 6]*fPost[ 6] + param_cx[ 2]*fPost[ 2] +
             param_cx[35]*fPost[35] + param_cx[31]*fPost[31] +
             param_cx[25]*fPost[25] + param_cx[11]*fPost[11] +
             param_cx[ 5]*fPost[ 5] + param_cx[ 1]*fPost[ 1] +
             param_cx[34]*fPost[34] + param_cx[30]*fPost[30] +
             param_cx[24]*fPost[24] + param_cx[17]*fPost[17] +
             param_cx[10]*fPost[10] + param_cx[ 4]*fPost[ 4] +
             param_cx[ 0]*fPost[ 0] + param_cx[29]*fPost[29] +
             param_cx[23]*fPost[23] + param_cx[16]*fPost[16] +
             param_cx[ 9]*fPost[ 9] + param_cx[ 3]*fPost[ 3] +
             param_cx[22]*fPost[22] + param_cx[15]*fPost[15] +
             param_cx[ 8]*fPost[ 8];

        OE = 0.5*(param_cx[33]*param_cx[33]+param_cy[33]*param_cy[33])*fPost[33] +
             0.5*(param_cx[27]*param_cx[27]+param_cy[27]*param_cy[27])*fPost[27] +
             0.5*(param_cx[20]*param_cx[20]+param_cy[20]*param_cy[20])*fPost[20] +
             0.5*(param_cx[13]*param_cx[13]+param_cy[13]*param_cy[13])*fPost[13] +
             0.5*(param_cx[ 7]*param_cx[ 7]+param_cy[ 7]*param_cy[ 7])*fPost[ 7] +
             0.5*(param_cx[36]*param_cx[36]+param_cy[36]*param_cy[36])*fPost[36] +
             0.5*(param_cx[32]*param_cx[32]+param_cy[32]*param_cy[32])*fPost[32] +
             0.5*(param_cx[26]*param_cx[26]+param_cy[26]*param_cy[26])*fPost[26] +
             0.5*(param_cx[19]*param_cx[19]+param_cy[19]*param_cy[19])*fPost[19] +
             0.5*(param_cx[12]*param_cx[12]+param_cy[12]*param_cy[12])*fPost[12] +
             0.5*(param_cx[ 6]*param_cx[ 6]+param_cy[ 6]*param_cy[ 6])*fPost[ 6] +
             0.5*(param_cx[ 2]*param_cx[ 2]+param_cy[ 2]*param_cy[ 2])*fPost[ 2] +
             0.5*(param_cx[35]*param_cx[35]+param_cy[35]*param_cy[35])*fPost[35] +
             0.5*(param_cx[31]*param_cx[31]+param_cy[31]*param_cy[31])*fPost[31] +
             0.5*(param_cx[25]*param_cx[25]+param_cy[25]*param_cy[25])*fPost[25] +
             0.5*(param_cx[11]*param_cx[11]+param_cy[11]*param_cy[11])*fPost[11] +
             0.5*(param_cx[ 5]*param_cx[ 5]+param_cy[ 5]*param_cy[ 5])*fPost[ 5] +
             0.5*(param_cx[ 1]*param_cx[ 1]+param_cy[ 1]*param_cy[ 1])*fPost[ 1] +
             0.5*(param_cx[34]*param_cx[34]+param_cy[34]*param_cy[34])*fPost[34] +
             0.5*(param_cx[30]*param_cx[30]+param_cy[30]*param_cy[30])*fPost[30] +
             0.5*(param_cx[24]*param_cx[24]+param_cy[24]*param_cy[24])*fPost[24] +
             0.5*(param_cx[17]*param_cx[17]+param_cy[17]*param_cy[17])*fPost[17] + 
             0.5*(param_cx[10]*param_cx[10]+param_cy[10]*param_cy[10])*fPost[10] +
             0.5*(param_cx[ 4]*param_cx[ 4]+param_cy[ 4]*param_cy[ 4])*fPost[ 4] +
             0.5*(param_cx[ 0]*param_cx[ 0]+param_cy[ 0]*param_cy[ 0])*fPost[ 0] +
             0.5*(param_cx[29]*param_cx[29]+param_cy[29]*param_cy[29])*fPost[29] +
             0.5*(param_cx[23]*param_cx[23]+param_cy[23]*param_cy[23])*fPost[23] +
             0.5*(param_cx[16]*param_cx[16]+param_cy[16]*param_cy[16])*fPost[16] +
             0.5*(param_cx[ 9]*param_cx[ 9]+param_cy[ 9]*param_cy[ 9])*fPost[ 9] +
             0.5*(param_cx[ 3]*param_cx[ 3]+param_cy[ 3]*param_cy[ 3])*fPost[ 3] +
             0.5*(param_cx[22]*param_cx[22]+param_cy[22]*param_cy[22])*fPost[22] +
             0.5*(param_cx[15]*param_cx[15]+param_cy[15]*param_cy[15])*fPost[15] +
             0.5*(param_cx[ 8]*param_cx[ 8]+param_cy[ 8]*param_cy[ 8])*fPost[ 8];
  
        a1 = -2.*N*UNIT*UNIT;
        b1 = (29./2.)*(ptildex-Ox)*UNIT*UNIT;
        d1 = 3.*(ptildex-Ox);

        b2 = (Etilde-OE)*(29./2.)*UNIT*UNIT-(281./4.)*N*UNIT*UNIT*UNIT*UNIT;
        d2 = 3.*(Etilde-OE)-(29./2.)*N*UNIT*UNIT;

        E = -d2/b2;

        px = (b1*d2-d1*b2)/(a1*b2);

        phi[21] = 1.+param_cx[21]*px+0.5*(param_cx[21]*param_cx[21]+param_cy[21]*param_cy[21])*E;   
        phi[14] = 1.+param_cx[14]*px+0.5*(param_cx[14]*param_cx[14]+param_cy[14]*param_cy[14])*E;
        phi[28] = 1.+param_cx[28]*px+0.5*(param_cx[28]*param_cx[28]+param_cy[28]*param_cy[28])*E;

        S = phi[21]+phi[14]+phi[28];

        fPost[21] = phi[21]*N/S;  
        fPost[14] = phi[14]*N/S;  
        fPost[28] = phi[28]*N/S;
  
        ////////////FINE LOWER LAYER 3////////////////////////
  
      } // end if ( threadIdx.x==2 )      
  
    //__syncthreads();
 
    //////////////////////////////////////////////////////////
    // move populations from previous to current lattice
    
    p_nxt[( 0 * NX_d*NY_d) + idx_cur] = fPost[ 0];  //    curY;
    p_nxt[( 1 * NX_d*NY_d) + idx_cur] = fPost[ 1];  //    curY;
    p_nxt[( 2 * NX_d*NY_d) + idx_cur] = fPost[ 2];  //    curY;
    p_nxt[( 3 * NX_d*NY_d) + idx_cur] = fPost[ 3];  //    curY;
    p_nxt[( 4 * NX_d*NY_d) + idx_cur] = fPost[ 4];  //    curY;
    p_nxt[( 5 * NX_d*NY_d) + idx_cur] = fPost[ 5];  //    curY;
    p_nxt[( 6 * NX_d*NY_d) + idx_cur] = fPost[ 6];  //    curY;
    p_nxt[( 7 * NX_d*NY_d) + idx_cur] = fPost[ 7];  //    curY;
    p_nxt[( 8 * NX_d*NY_d) + idx_cur] = fPost[ 8];  //    curY;
    p_nxt[( 9 * NX_d*NY_d) + idx_cur] = fPost[ 9];  //    curY;
    p_nxt[(10 * NX_d*NY_d) + idx_cur] = fPost[10];  //    curY;
    p_nxt[(11 * NX_d*NY_d) + idx_cur] = fPost[11];  //    curY;
    p_nxt[(12 * NX_d*NY_d) + idx_cur] = fPost[12];  //    curY;
    p_nxt[(13 * NX_d*NY_d) + idx_cur] = fPost[13];  //    curY;
    p_nxt[(14 * NX_d*NY_d) + idx_cur] = fPost[14];  //    curY;
    p_nxt[(15 * NX_d*NY_d) + idx_cur] = fPost[15];  //    curY;
    p_nxt[(16 * NX_d*NY_d) + idx_cur] = fPost[16];  //    curY;
    p_nxt[(17 * NX_d*NY_d) + idx_cur] = fPost[17];  //    curY;
    p_nxt[(18 * NX_d*NY_d) + idx_cur] = fPost[18];  //    curY;
    p_nxt[(19 * NX_d*NY_d) + idx_cur] = fPost[19];  //    curY;
    p_nxt[(20 * NX_d*NY_d) + idx_cur] = fPost[20];  //    curY;
    p_nxt[(21 * NX_d*NY_d) + idx_cur] = fPost[21];  //    curY;
    p_nxt[(22 * NX_d*NY_d) + idx_cur] = fPost[22];  //    curY;
    p_nxt[(23 * NX_d*NY_d) + idx_cur] = fPost[23];  //    curY;
    p_nxt[(24 * NX_d*NY_d) + idx_cur] = fPost[24];  //    curY;
    p_nxt[(25 * NX_d*NY_d) + idx_cur] = fPost[25];  //    curY;
    p_nxt[(26 * NX_d*NY_d) + idx_cur] = fPost[26];  //    curY;
    p_nxt[(27 * NX_d*NY_d) + idx_cur] = fPost[27];  //    curY;
    p_nxt[(28 * NX_d*NY_d) + idx_cur] = fPost[28];  //    curY;
    p_nxt[(29 * NX_d*NY_d) + idx_cur] = fPost[29];  //    curY;
    p_nxt[(30 * NX_d*NY_d) + idx_cur] = fPost[30];  //    curY;
    p_nxt[(31 * NX_d*NY_d) + idx_cur] = fPost[31];  //    curY;
    p_nxt[(32 * NX_d*NY_d) + idx_cur] = fPost[32];  //    curY;
    p_nxt[(33 * NX_d*NY_d) + idx_cur] = fPost[33];  //    curY;
    p_nxt[(34 * NX_d*NY_d) + idx_cur] = fPost[34];  //    curY;
    p_nxt[(35 * NX_d*NY_d) + idx_cur] = fPost[35];  //    curY;
    p_nxt[(36 * NX_d*NY_d) + idx_cur] = fPost[36];  //    curY;
    
  } // end if (idx_cur < LSIZEX_d*NY_d)
 
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Global function BC.
__global__ void bcT (
      data_t *p_nxt,
      data_t *p_prv
    ) {
  
  long idx_cur;       // Index of current site.
  
  data_t localPop[NPOP];  // Array where local populations are stored 
                          // during operations.
    
  //////////////////////////////////////////////////////////////////////////////
  data_t a1,a2,a3;
  data_t b1,b2;
  data_t c1,c2,c3;
  data_t d1,d2,d3;
  
  data_t utilde1,vtilde1;
  data_t utilde2,vtilde2;
  data_t utilde3;
  
  data_t term1,term2;
  
  data_t temptilde1,temptilde2,temptilde3;
  data_t localTempfic;
  
  data_t px,py,E,S,N,Ox,Oy,OE;
  data_t ptildex,ptildey,Etilde,massapost;
  
#ifndef ADIABATIC
  data_t xa[ENNE+2], ya[ENNE+2];
#endif
  
  data_t rhoi, forcey, scalar;
  
  int k;

  long idx3;
  
  data_t phi[NPOP];
  data_t fPre[NPOP], fPost[NPOP];
  
  data_t  rho, temp, u, v;
    
  ////////////////////////////////////////////////////////////////////////////
  // Set index of current site.      
  idx_cur = blockIdx.y*blockDim.y*NY_d + threadIdx.y*NY_d + threadIdx.x + (LSIZEY_d-3);    

  // This condition is set to block final threads when LSIZEX_d is not divisible by BC_BLOCK_DIMY
  if (idx_cur < LSIZEX_d*NY_d) {

    ////////////////////////////////////////////////////////////////////////////
    // Load populations of prv lattice in localPop array.
    
    fPre[ 0] = p_prv[ ( 0 * NX_d*NY_d) + idx_cur ];
    fPre[ 1] = p_prv[ ( 1 * NX_d*NY_d) + idx_cur ];
    fPre[ 2] = p_prv[ ( 2 * NX_d*NY_d) + idx_cur ];
    fPre[ 3] = p_prv[ ( 3 * NX_d*NY_d) + idx_cur ];
    fPre[ 4] = p_prv[ ( 4 * NX_d*NY_d) + idx_cur ];
    fPre[ 5] = p_prv[ ( 5 * NX_d*NY_d) + idx_cur ];
    fPre[ 6] = p_prv[ ( 6 * NX_d*NY_d) + idx_cur ];
    fPre[ 7] = p_prv[ ( 7 * NX_d*NY_d) + idx_cur ];
    fPre[ 8] = p_prv[ ( 8 * NX_d*NY_d) + idx_cur ];
    fPre[ 9] = p_prv[ ( 9 * NX_d*NY_d) + idx_cur ];
    fPre[10] = p_prv[ (10 * NX_d*NY_d) + idx_cur ];
    fPre[11] = p_prv[ (11 * NX_d*NY_d) + idx_cur ];
    fPre[12] = p_prv[ (12 * NX_d*NY_d) + idx_cur ];
    fPre[13] = p_prv[ (13 * NX_d*NY_d) + idx_cur ];
    fPre[14] = p_prv[ (14 * NX_d*NY_d) + idx_cur ];
    fPre[15] = p_prv[ (15 * NX_d*NY_d) + idx_cur ];
    fPre[16] = p_prv[ (16 * NX_d*NY_d) + idx_cur ];
    fPre[17] = p_prv[ (17 * NX_d*NY_d) + idx_cur ];
    fPre[18] = p_prv[ (18 * NX_d*NY_d) + idx_cur ];
    fPre[19] = p_prv[ (19 * NX_d*NY_d) + idx_cur ];
    fPre[20] = p_prv[ (20 * NX_d*NY_d) + idx_cur ];
    fPre[21] = p_prv[ (21 * NX_d*NY_d) + idx_cur ];
    fPre[22] = p_prv[ (22 * NX_d*NY_d) + idx_cur ];
    fPre[23] = p_prv[ (23 * NX_d*NY_d) + idx_cur ];
    fPre[24] = p_prv[ (24 * NX_d*NY_d) + idx_cur ];
    fPre[25] = p_prv[ (25 * NX_d*NY_d) + idx_cur ];
    fPre[26] = p_prv[ (26 * NX_d*NY_d) + idx_cur ];
    fPre[27] = p_prv[ (27 * NX_d*NY_d) + idx_cur ];
    fPre[28] = p_prv[ (28 * NX_d*NY_d) + idx_cur ];
    fPre[29] = p_prv[ (29 * NX_d*NY_d) + idx_cur ];
    fPre[30] = p_prv[ (30 * NX_d*NY_d) + idx_cur ];
    fPre[31] = p_prv[ (31 * NX_d*NY_d) + idx_cur ];
    fPre[32] = p_prv[ (32 * NX_d*NY_d) + idx_cur ];
    fPre[33] = p_prv[ (33 * NX_d*NY_d) + idx_cur ];
    fPre[34] = p_prv[ (34 * NX_d*NY_d) + idx_cur ];
    fPre[35] = p_prv[ (35 * NX_d*NY_d) + idx_cur ];
    fPre[36] = p_prv[ (36 * NX_d*NY_d) + idx_cur ];
      
    ////////////////////////////////////////////////////////////////////////////
    // Load populations of succ lattice in localPop array.
    
    fPost[ 0] = p_nxt[ ( 0 * NX_d*NY_d) + idx_cur ];
    fPost[ 1] = p_nxt[ ( 1 * NX_d*NY_d) + idx_cur ];
    fPost[ 2] = p_nxt[ ( 2 * NX_d*NY_d) + idx_cur ];
    fPost[ 3] = p_nxt[ ( 3 * NX_d*NY_d) + idx_cur ];
    fPost[ 4] = p_nxt[ ( 4 * NX_d*NY_d) + idx_cur ];
    fPost[ 5] = p_nxt[ ( 5 * NX_d*NY_d) + idx_cur ];
    fPost[ 6] = p_nxt[ ( 6 * NX_d*NY_d) + idx_cur ];
    fPost[ 7] = p_nxt[ ( 7 * NX_d*NY_d) + idx_cur ];
    fPost[ 8] = p_nxt[ ( 8 * NX_d*NY_d) + idx_cur ];
    fPost[ 9] = p_nxt[ ( 9 * NX_d*NY_d) + idx_cur ];
    fPost[10] = p_nxt[ (10 * NX_d*NY_d) + idx_cur ];
    fPost[11] = p_nxt[ (11 * NX_d*NY_d) + idx_cur ];
    fPost[12] = p_nxt[ (12 * NX_d*NY_d) + idx_cur ];
    fPost[13] = p_nxt[ (13 * NX_d*NY_d) + idx_cur ];
    fPost[14] = p_nxt[ (14 * NX_d*NY_d) + idx_cur ];
    fPost[15] = p_nxt[ (15 * NX_d*NY_d) + idx_cur ];
    fPost[16] = p_nxt[ (16 * NX_d*NY_d) + idx_cur ];
    fPost[17] = p_nxt[ (17 * NX_d*NY_d) + idx_cur ];
    fPost[18] = p_nxt[ (18 * NX_d*NY_d) + idx_cur ];
    fPost[19] = p_nxt[ (19 * NX_d*NY_d) + idx_cur ];
    fPost[20] = p_nxt[ (20 * NX_d*NY_d) + idx_cur ];
    fPost[21] = p_nxt[ (21 * NX_d*NY_d) + idx_cur ];
    fPost[22] = p_nxt[ (22 * NX_d*NY_d) + idx_cur ];
    fPost[23] = p_nxt[ (23 * NX_d*NY_d) + idx_cur ];
    fPost[24] = p_nxt[ (24 * NX_d*NY_d) + idx_cur ];
    fPost[25] = p_nxt[ (25 * NX_d*NY_d) + idx_cur ];
    fPost[26] = p_nxt[ (26 * NX_d*NY_d) + idx_cur ];
    fPost[27] = p_nxt[ (27 * NX_d*NY_d) + idx_cur ];
    fPost[28] = p_nxt[ (28 * NX_d*NY_d) + idx_cur ];
    fPost[29] = p_nxt[ (29 * NX_d*NY_d) + idx_cur ];
    fPost[30] = p_nxt[ (30 * NX_d*NY_d) + idx_cur ];
    fPost[31] = p_nxt[ (31 * NX_d*NY_d) + idx_cur ];
    fPost[32] = p_nxt[ (32 * NX_d*NY_d) + idx_cur ];
    fPost[33] = p_nxt[ (33 * NX_d*NY_d) + idx_cur ];
    fPost[34] = p_nxt[ (34 * NX_d*NY_d) + idx_cur ];
    fPost[35] = p_nxt[ (35 * NX_d*NY_d) + idx_cur ];
    fPost[36] = p_nxt[ (36 * NX_d*NY_d) + idx_cur ];
      
    ////////////////////////////////////////////////////////////////////////////
   
    //upper bord
      
    idx3 = idx_cur-threadIdx.x-1; // index of population in y=LSIZEY_d-4
    
    /////////PROCEDURE OF EXTRAPOLATION-INTERPOLATION////////////

    //     ya[2]=tempfic[idx3];   
    //------------- the line above to be replaced by the following set of equations
    //-------------- localPop is read from fPost[idx3]
    //-------------- THE SAME TO BE DONE FOR THE UPPER LAYER...
          
    localPop[ 0] = p_nxt[ ( 0 * NX_d*NY_d) + idx3 ];
    localPop[ 1] = p_nxt[ ( 1 * NX_d*NY_d) + idx3 ];
    localPop[ 2] = p_nxt[ ( 2 * NX_d*NY_d) + idx3 ];
    localPop[ 3] = p_nxt[ ( 3 * NX_d*NY_d) + idx3 ];
    localPop[ 4] = p_nxt[ ( 4 * NX_d*NY_d) + idx3 ];
    localPop[ 5] = p_nxt[ ( 5 * NX_d*NY_d) + idx3 ];
    localPop[ 6] = p_nxt[ ( 6 * NX_d*NY_d) + idx3 ];
    localPop[ 7] = p_nxt[ ( 7 * NX_d*NY_d) + idx3 ];
    localPop[ 8] = p_nxt[ ( 8 * NX_d*NY_d) + idx3 ];
    localPop[ 9] = p_nxt[ ( 9 * NX_d*NY_d) + idx3 ];
    localPop[10] = p_nxt[ (10 * NX_d*NY_d) + idx3 ];
    localPop[11] = p_nxt[ (11 * NX_d*NY_d) + idx3 ];
    localPop[12] = p_nxt[ (12 * NX_d*NY_d) + idx3 ];
    localPop[13] = p_nxt[ (13 * NX_d*NY_d) + idx3 ];
    localPop[14] = p_nxt[ (14 * NX_d*NY_d) + idx3 ];
    localPop[15] = p_nxt[ (15 * NX_d*NY_d) + idx3 ];
    localPop[16] = p_nxt[ (16 * NX_d*NY_d) + idx3 ];
    localPop[17] = p_nxt[ (17 * NX_d*NY_d) + idx3 ];
    localPop[18] = p_nxt[ (18 * NX_d*NY_d) + idx3 ];
    localPop[19] = p_nxt[ (19 * NX_d*NY_d) + idx3 ];
    localPop[20] = p_nxt[ (20 * NX_d*NY_d) + idx3 ];
    localPop[21] = p_nxt[ (21 * NX_d*NY_d) + idx3 ];
    localPop[22] = p_nxt[ (22 * NX_d*NY_d) + idx3 ];
    localPop[23] = p_nxt[ (23 * NX_d*NY_d) + idx3 ];
    localPop[24] = p_nxt[ (24 * NX_d*NY_d) + idx3 ];
    localPop[25] = p_nxt[ (25 * NX_d*NY_d) + idx3 ];
    localPop[26] = p_nxt[ (26 * NX_d*NY_d) + idx3 ];
    localPop[27] = p_nxt[ (27 * NX_d*NY_d) + idx3 ];
    localPop[28] = p_nxt[ (28 * NX_d*NY_d) + idx3 ];
    localPop[29] = p_nxt[ (29 * NX_d*NY_d) + idx3 ];
    localPop[30] = p_nxt[ (30 * NX_d*NY_d) + idx3 ];
    localPop[31] = p_nxt[ (31 * NX_d*NY_d) + idx3 ];
    localPop[32] = p_nxt[ (32 * NX_d*NY_d) + idx3 ];
    localPop[33] = p_nxt[ (33 * NX_d*NY_d) + idx3 ];
    localPop[34] = p_nxt[ (34 * NX_d*NY_d) + idx3 ];
    localPop[35] = p_nxt[ (35 * NX_d*NY_d) + idx3 ];
    localPop[36] = p_nxt[ (36 * NX_d*NY_d) + idx3 ];
    
    rho = 0.0; 
    u = 0.0; 
    v = 0.0;
    
    for ( k = 0 ; k < NPOP ; k++) {
      rho = rho + localPop[k];
      u   = u   + param_cx[k] * localPop[k];
      v   = v   + param_cy[k] * localPop[k];
    }

    rhoi = 1.0 / rho;

    forcey = rho * GRAVITY * DELTAT;

    u = u * rhoi;
    v = v * rhoi;

    temp = 0.0;
    
    for ( k = 0 ; k < NPOP ; k++ ) {
      scalar = (param_cx[k] - u) * (param_cx[k] - u) + 
               (param_cy[k] - v) * (param_cy[k] - v);
      temp = temp + 0.5*scalar*localPop[k];
    }
    
    temp = temp * rhoi;

    localTempfic = temp +
                   (forcex*forcex+forcey*forcey) * 
                   rhoi * rhoi/8.0;
      
    // ---- end of replacement --------------------------------------------------

#ifdef ADIABATIC

    temptilde1 = localTempfic ;
    temptilde2 = localTempfic ;
    temptilde3 = localTempfic ;

#else

    temptilde1 = TEMPWALLUP;
    temptilde2 = TEMPWALLUP;
    temptilde3 = TEMPWALLUP;

    xa[1] = 3.0;  //this point seems crucial! 
    xa[2] = 4.0;   
    ya[1] = TEMPWALLUP;
    ya[2] = localTempfic;

    xa[0] = 2.0;//I want it here
    _polint(xa,ya);
    temptilde2 = ya[0];

    xa[0]=2.0;//I want it here
    _polint(xa,ya);
    temptilde2 = ya[0];

    xa[0]=1.0;//I want it here
    _polint(xa,ya);
    temptilde1 = ya[0];

#endif
 
    //x velocity
    utilde1=0.0;
    utilde2=0.0;
    utilde3=0.0;

    ///y velocity
    vtilde1=0.0;
    vtilde2=0.0;
      
    if (threadIdx.x == 2) {   

      ////////////UPPER LAYER 1 ///////////////////


      N= ( fPre[21] + fPre[14] + fPre[28] ) + 
         ( fPre[ 7] + fPre[13] + fPre[20] + 
           fPre[27] + fPre[33] ) +
         ( fPre[ 2] + fPre[ 6] + fPre[12] + fPre[19] + 
           fPre[26] + fPre[32] + fPre[36] );

      massapost = N +
                  fPost[35] + fPost[31] + fPost[25] +
                  fPost[11] + fPost[ 5] + fPost[ 1] + fPost[ 2] +
                  fPost[ 6] + fPost[12] + fPost[19] + fPost[26] +
                  fPost[32] + fPost[36] + fPost[ 7] + fPost[13] +
                  fPost[20] + fPost[27] + fPost[33] + fPost[14] +
                  fPost[21] + fPost[28] + fPost[18];

      ptildex = massapost*(utilde1)-0.5*FORCEPOIS*DELTAT;
      ptildey = massapost*(vtilde1)-0.5*GRAVITY*massapost*DELTAT;

      Etilde = (temptilde1)*massapost+0.5*(ptildex*ptildex+ptildey*ptildey)/massapost;


      Ox = param_cx[35]*fPost[35] + param_cx[31]*fPost[31] + param_cx[25]*fPost[25] +
           param_cx[11]*fPost[11] + param_cx[ 5]*fPost[ 5] + param_cx[ 1]*fPost[ 1] +
           param_cx[ 2]*fPost[ 2] + param_cx[ 6]*fPost[ 6] + param_cx[12]*fPost[12] +
           param_cx[19]*fPost[19] + param_cx[26]*fPost[26] + param_cx[32]*fPost[32] +
           param_cx[36]*fPost[36] + param_cx[ 7]*fPost[ 7] + param_cx[13]*fPost[13] +
           param_cx[20]*fPost[20] + param_cx[27]*fPost[27] + param_cx[33]*fPost[33] +
           param_cx[14]*fPost[14] + param_cx[21]*fPost[21] + param_cx[28]*fPost[28] ;

      Oy = (param_cy[35]*fPost[35] + param_cy[31]*fPost[31] + param_cy[25]*fPost[25] +
            param_cy[11]*fPost[11] + param_cy[ 5]*fPost[ 5] + param_cy[ 1]*fPost[ 1] +
            param_cy[ 2]*fPost[ 2] + param_cy[ 6]*fPost[ 6] + param_cy[12]*fPost[12] +
            param_cy[19]*fPost[19] + param_cy[26]*fPost[26] + param_cy[32]*fPost[32] +
            param_cy[36]*fPost[36] + param_cy[ 7]*fPost[ 7] + param_cy[13]*fPost[13] +
            param_cy[20]*fPost[20] + param_cy[27]*fPost[27] + param_cy[33]*fPost[33] +
            param_cy[14]*fPost[14] + param_cy[21]*fPost[21] + param_cy[28]*fPost[28]);
  
      OE = 0.5*(param_cx[35]*param_cx[35]+param_cy[35]*param_cy[35])*fPost[35]+
           0.5*(param_cx[31]*param_cx[31]+param_cy[31]*param_cy[31])*fPost[31]+
           0.5*(param_cx[25]*param_cx[25]+param_cy[25]*param_cy[25])*fPost[25]+
           0.5*(param_cx[11]*param_cx[11]+param_cy[11]*param_cy[11])*fPost[11]+
           0.5*(param_cx[ 5]*param_cx[ 5]+param_cy[ 5]*param_cy[ 5])*fPost[ 5]+
           0.5*(param_cx[ 1]*param_cx[ 1]+param_cy[ 1]*param_cy[ 1])*fPost[ 1]+
           0.5*(param_cx[ 2]*param_cx[ 2]+param_cy[ 2]*param_cy[ 2])*fPost[ 2]+
           0.5*(param_cx[ 6]*param_cx[ 6]+param_cy[ 6]*param_cy[ 6])*fPost[ 6]+
           0.5*(param_cx[12]*param_cx[12]+param_cy[12]*param_cy[12])*fPost[12]+
           0.5*(param_cx[19]*param_cx[19]+param_cy[19]*param_cy[19])*fPost[19]+ 
           0.5*(param_cx[26]*param_cx[26]+param_cy[26]*param_cy[26])*fPost[26]+
           0.5*(param_cx[32]*param_cx[32]+param_cy[32]*param_cy[32])*fPost[32]+
           0.5*(param_cx[36]*param_cx[36]+param_cy[36]*param_cy[36])*fPost[36]+
           0.5*(param_cx[ 7]*param_cx[ 7]+param_cy[ 7]*param_cy[ 7])*fPost[ 7]+
           0.5*(param_cx[13]*param_cx[13]+param_cy[13]*param_cy[13])*fPost[13]+
           0.5*(param_cx[20]*param_cx[20]+param_cy[20]*param_cy[20])*fPost[20]+
           0.5*(param_cx[27]*param_cx[27]+param_cy[27]*param_cy[27])*fPost[27]+
           0.5*(param_cx[33]*param_cx[33]+param_cy[33]*param_cy[33])*fPost[33]+
           0.5*(param_cx[14]*param_cx[14]+param_cy[14]*param_cy[14])*fPost[14]+
           0.5*(param_cx[21]*param_cx[21]+param_cy[21]*param_cy[21])*fPost[21]+
           0.5*(param_cx[28]*param_cx[28]+param_cy[28]*param_cy[28])*fPost[28];

      a1 = -26.*(ptildex-Ox)*UNIT;
      b1 = -40.*N*UNIT*UNIT;
      c1 = 47.*(ptildex-Ox)*UNIT*UNIT;
      d1 = 15.*(ptildex-Ox);

      a2 = -26.*(ptildey-Oy)*UNIT-54.*N*UNIT*UNIT;
      c2 = 47.*(ptildey-Oy)*UNIT*UNIT+91.*N*UNIT*UNIT*UNIT;
      d2 = 15.*(ptildey-Oy)+26.*N*UNIT;

      a3 = -26.*(Etilde-OE)*UNIT+91.*N*UNIT*UNIT*UNIT;
      c3 = 47.*(Etilde-OE)*UNIT*UNIT-N*UNIT*UNIT*UNIT*UNIT*367./2.;
      d3 = 15.*(Etilde-OE)-47.*N*UNIT*UNIT;

      py = (-c3*d2+c2*d3)/(-a3*c2+a2*c3);
      px = (a2*c3*d1-a2*c1*d3-a3*c2*d1-c3*a1*d2+c1*a3*d2+a1*c2*d3)/(b1*(a3*c2-a2*c3));

      E  = (-a3*d2+a2*d3)/(a3*c2-a2*c3);
  
      phi[17] = 1.+param_cx[17]*px+param_cy[17]*py+0.5*(param_cx[17]*param_cx[17]+param_cy[17]*param_cy[17])*E;
      phi[16] = 1.+param_cx[16]*px+param_cy[16]*py+0.5*(param_cx[16]*param_cx[16]+param_cy[16]*param_cy[16])*E;
      phi[15] = 1.+param_cx[15]*px+param_cy[15]*py+0.5*(param_cx[15]*param_cx[15]+param_cy[15]*param_cy[15])*E;
      phi[24] = 1.+param_cx[24]*px+param_cy[24]*py+0.5*(param_cx[24]*param_cx[24]+param_cy[24]*param_cy[24])*E;
      phi[23] = 1.+param_cx[23]*px+param_cy[23]*py+0.5*(param_cx[23]*param_cx[23]+param_cy[23]*param_cy[23])*E;
      phi[22] = 1.+param_cx[22]*px+param_cy[22]*py+0.5*(param_cx[22]*param_cx[22]+param_cy[22]*param_cy[22])*E;
      phi[10] = 1.+param_cx[10]*px+param_cy[10]*py+0.5*(param_cx[10]*param_cx[10]+param_cy[10]*param_cy[10])*E;
      phi[ 9] = 1.+param_cx[ 9]*px+param_cy[ 9]*py+0.5*(param_cx[ 9]*param_cx[ 9]+param_cy[ 9]*param_cy[ 9])*E;
      phi[ 8] = 1.+param_cx[ 8]*px+param_cy[ 8]*py+0.5*(param_cx[ 8]*param_cx[ 8]+param_cy[ 8]*param_cy[ 8])*E;
      phi[29] = 1.+param_cx[29]*px+param_cy[29]*py+0.5*(param_cx[29]*param_cx[29]+param_cy[29]*param_cy[29])*E;
      phi[30] = 1.+param_cx[30]*px+param_cy[30]*py+0.5*(param_cx[30]*param_cx[30]+param_cy[30]*param_cy[30])*E;
      phi[ 3] = 1.+param_cx[ 3]*px+param_cy[ 3]*py+0.5*(param_cx[ 3]*param_cx[ 3]+param_cy[ 3]*param_cy[ 3])*E;
      phi[ 4] = 1.+param_cx[ 4]*px+param_cy[ 4]*py+0.5*(param_cx[ 4]*param_cx[ 4]+param_cy[ 4]*param_cy[ 4])*E;
      phi[34] = 1.+param_cx[34]*px+param_cy[34]*py+0.5*(param_cx[34]*param_cx[34]+param_cy[34]*param_cy[34])*E;
      phi[ 0] = 1.+param_cx[ 0]*px+param_cy[ 0]*py+0.5*(param_cx[ 0]*param_cx[ 0]+param_cy[ 0]*param_cy[ 0])*E;

      S = phi[17] + phi[16] + phi[15] + phi[24] + phi[23] + phi[22] +
          phi[10] + phi[ 9] + phi[ 8] + phi[29] + phi[30] + phi[ 3] +
          phi[ 4] + phi[34] + phi[ 0]; 

      fPost[17] = phi[17]*N/S; 
      fPost[16] = phi[16]*N/S;  
      fPost[15] = phi[15]*N/S;  
             
      fPost[24] = phi[24]*N/S;  
      fPost[23] = phi[23]*N/S;
      fPost[22] = phi[22]*N/S;  
             
      fPost[10] = phi[10]*N/S;  
      fPost[ 9] = phi[ 9]*N/S;  
      fPost[ 8] = phi[ 8]*N/S;
             
      fPost[29] = phi[29]*N/S;  
      fPost[30] = phi[30]*N/S;  
             
      fPost[ 3] = phi[ 3]*N/S;  
      fPost[ 4] = phi[ 4]*N/S;  

      fPost[34] = phi[34]*N/S;  
      fPost[ 0] = phi[ 0]*N/S;  

      //////////FINE UPPER  LAYER1////////////////////
  
    } // end if (threadIdx.x == 2)
      
    else
      
      if (threadIdx.x == 1) {  
  
        //////////UPPER LAYER 2/////////////////////////////////

        N = ( fPre[21] + fPre[14] + fPre[28] ) +
            ( fPre[ 7] + fPre[13] + fPre[20]   +
              fPre[27] + fPre[33] );

        massapost = N +
                    fPost[34] + fPost[30] + fPost[24] +
                    fPost[17] + fPost[10] + fPost[ 4] +
                    fPost[ 0] + fPost[35] + fPost[31] +
                    fPost[25] + fPost[11] + fPost[ 5] +
                    fPost[ 1] + fPost[ 2] + fPost[ 6] +
                    fPost[12] + fPost[19] + fPost[26] +
                    fPost[32] + fPost[36] + fPost[ 7] +
                    fPost[13] + fPost[20] + fPost[27] +
                    fPost[33] + fPost[14] + fPost[21] +
                    fPost[28] + fPost[18];

        ptildex = massapost*(utilde2) - 0.5*FORCEPOIS*DELTAT;
        ptildey = massapost*(vtilde2) - 0.5*GRAVITY*massapost*DELTAT;

        Etilde  = (temptilde2)*massapost + 
                  0.5*(ptildex*ptildex+ptildey*ptildey)/massapost;    

        Ox = param_cx[34]*fPost[34] + param_cx[30]*fPost[30] + param_cx[24]*fPost[24] +
             param_cx[17]*fPost[17] + param_cx[10]*fPost[10] + param_cx[ 4]*fPost[ 4] +
             param_cx[ 0]*fPost[ 0] + param_cx[35]*fPost[35] + param_cx[31]*fPost[31] +
             param_cx[25]*fPost[25] + param_cx[11]*fPost[11] + param_cx[ 5]*fPost[ 5] +
             param_cx[ 1]*fPost[ 1] + param_cx[ 2]*fPost[ 2] + param_cx[ 6]*fPost[ 6] +
             param_cx[12]*fPost[12] + param_cx[19]*fPost[19] + param_cx[26]*fPost[26] +
             param_cx[32]*fPost[32] + param_cx[36]*fPost[36] + param_cx[ 7]*fPost[ 7] +
             param_cx[13]*fPost[13] + param_cx[20]*fPost[20] + param_cx[27]*fPost[27] +
             param_cx[33]*fPost[33] + param_cx[14]*fPost[14] + param_cx[21]*fPost[21] +
             param_cx[28]*fPost[28];

        Oy = param_cy[34]*fPost[34] + param_cy[30]*fPost[30] + param_cy[24]*fPost[24] +
             param_cy[17]*fPost[17] + param_cy[10]*fPost[10] + param_cy[ 4]*fPost[ 4] +
             param_cy[ 0]*fPost[ 0] + param_cy[35]*fPost[35] + param_cy[31]*fPost[31] +
             param_cy[25]*fPost[25] + param_cy[11]*fPost[11] + param_cy[ 5]*fPost[ 5] +
             param_cy[ 1]*fPost[ 1] + param_cy[ 2]*fPost[ 2] + param_cy[ 6]*fPost[ 6] +
             param_cy[12]*fPost[12] + param_cy[19]*fPost[19] + param_cy[26]*fPost[26] +
             param_cy[32]*fPost[32] + param_cy[36]*fPost[36] + param_cy[ 7]*fPost[ 7] +
             param_cy[13]*fPost[13] + param_cy[20]*fPost[20] + param_cy[27]*fPost[27] +
             param_cy[33]*fPost[33] + param_cy[14]*fPost[14] + param_cy[21]*fPost[21] +
             param_cy[28]*fPost[28];

        OE = 0.5 * (param_cx[34]*param_cx[34] + param_cy[34]*param_cy[34]) * fPost[34] +
             0.5 * (param_cx[30]*param_cx[30] + param_cy[30]*param_cy[30]) * fPost[30] +
             0.5 * (param_cx[24]*param_cx[24] + param_cy[24]*param_cy[24]) * fPost[24] +
             0.5 * (param_cx[17]*param_cx[17] + param_cy[17]*param_cy[17]) * fPost[17] +
             0.5 * (param_cx[10]*param_cx[10] + param_cy[10]*param_cy[10]) * fPost[10] +
             0.5 * (param_cx[ 4]*param_cx[ 4] + param_cy[ 4]*param_cy[ 4]) * fPost[ 4] +
             0.5 * (param_cx[ 0]*param_cx[ 0] + param_cy[ 0]*param_cy[ 0]) * fPost[ 0] +
             0.5 * (param_cx[35]*param_cx[35] + param_cy[35]*param_cy[35]) * fPost[35] +
             0.5 * (param_cx[31]*param_cx[31] + param_cy[31]*param_cy[31]) * fPost[31] +
             0.5 * (param_cx[25]*param_cx[25] + param_cy[25]*param_cy[25]) * fPost[25] +
             0.5 * (param_cx[11]*param_cx[11] + param_cy[11]*param_cy[11]) * fPost[11] +
             0.5 * (param_cx[ 5]*param_cx[ 5] + param_cy[ 5]*param_cy[ 5]) * fPost[ 5] +
             0.5 * (param_cx[ 1]*param_cx[ 1] + param_cy[ 1]*param_cy[ 1]) * fPost[ 1] +
             0.5 * (param_cx[ 2]*param_cx[ 2] + param_cy[ 2]*param_cy[ 2]) * fPost[ 2] +
             0.5 * (param_cx[ 6]*param_cx[ 6] + param_cy[ 6]*param_cy[ 6]) * fPost[ 6] +
             0.5 * (param_cx[12]*param_cx[12] + param_cy[12]*param_cy[12]) * fPost[12] +
             0.5 * (param_cx[19]*param_cx[19] + param_cy[19]*param_cy[19]) * fPost[19] + 
             0.5 * (param_cx[26]*param_cx[26] + param_cy[26]*param_cy[26]) * fPost[26] +
             0.5 * (param_cx[32]*param_cx[32] + param_cy[32]*param_cy[32]) * fPost[32] +
             0.5 * (param_cx[36]*param_cx[36] + param_cy[36]*param_cy[36]) * fPost[36] +
             0.5 * (param_cx[ 7]*param_cx[ 7] + param_cy[ 7]*param_cy[ 7]) * fPost[ 7] +
             0.5 * (param_cx[13]*param_cx[13] + param_cy[13]*param_cy[13]) * fPost[13] +
             0.5 * (param_cx[20]*param_cx[20] + param_cy[20]*param_cy[20]) * fPost[20] +
             0.5 * (param_cx[27]*param_cx[27] + param_cy[27]*param_cy[27]) * fPost[27] +
             0.5 * (param_cx[33]*param_cx[33] + param_cy[33]*param_cy[33]) * fPost[33] +
             0.5 * (param_cx[14]*param_cx[14] + param_cy[14]*param_cy[14]) * fPost[14] +
             0.5 * (param_cx[21]*param_cx[21] + param_cy[21]*param_cy[21]) * fPost[21] +
             0.5 * (param_cx[28]*param_cx[28] + param_cy[28]*param_cy[28]) * fPost[28];
    
        a1 = -19.*(ptildex-Ox)*UNIT;
        b1 = -12.*N*UNIT*UNIT;
        c1 = (59/2.)*(ptildex-Ox)*UNIT*UNIT;
        d1 = 8.*(ptildex-Ox);

        a2 = -19.*(ptildey-Oy)*UNIT-47.*N*UNIT*UNIT;
        c2 = (59/2.)*(ptildey-Oy)*UNIT*UNIT+(147./2.)*N*UNIT*UNIT*UNIT;
        d2 = 8.*(ptildey-Oy)+19.*N*UNIT;

        a3 = -19.*(Etilde-OE)*UNIT+(147./2.)*N*UNIT*UNIT*UNIT;
        c3 = (59./2.)*(Etilde-OE)*UNIT*UNIT-N*UNIT*UNIT*UNIT*UNIT*475./4.;
        d3 = 8.*(Etilde-OE)-(59./2.)*N*UNIT*UNIT;

        py = (-c3*d2+c2*d3)/(-a3*c2+a2*c3);

        px = (a2*c3*d1-a2*c1*d3-a3*c2*d1-c3*a1*d2+c1*a3*d2+a1*c2*d3)/(b1*(a3*c2-a2*c3));

        E  = (-a3*d2+a2*d3)/(a3*c2-a2*c3);

        phi[16] = 1.+param_cx[16]*px+param_cy[16]*py+0.5*(param_cx[16]*param_cx[16]+param_cy[16]*param_cy[16])*E;
        phi[15] = 1.+param_cx[15]*px+param_cy[15]*py+0.5*(param_cx[15]*param_cx[15]+param_cy[15]*param_cy[15])*E;
        phi[23] = 1.+param_cx[23]*px+param_cy[23]*py+0.5*(param_cx[23]*param_cx[23]+param_cy[23]*param_cy[23])*E;
        phi[22] = 1.+param_cx[22]*px+param_cy[22]*py+0.5*(param_cx[22]*param_cx[22]+param_cy[22]*param_cy[22])*E;
        phi[ 9] = 1.+param_cx[ 9]*px+param_cy[ 9]*py+0.5*(param_cx[ 9]*param_cx[ 9]+param_cy[ 9]*param_cy[ 9])*E;
        phi[ 8] = 1.+param_cx[ 8]*px+param_cy[ 8]*py+0.5*(param_cx[ 8]*param_cx[ 8]+param_cy[ 8]*param_cy[ 8])*E;
        phi[29] = 1.+param_cx[29]*px+param_cy[29]*py+0.5*(param_cx[29]*param_cx[29]+param_cy[29]*param_cy[29])*E;
        phi[ 3] = 1.+param_cx[ 3]*px+param_cy[ 3]*py+0.5*(param_cx[ 3]*param_cx[ 3]+param_cy[ 3]*param_cy[ 3])*E;    
            
        S = phi[16] + phi[15] + phi[23] + phi[22] + phi[ 9] + phi[ 8] + phi[29] + phi[ 3];

        fPost[16] = phi[16]*N/S;  
        fPost[15] = phi[15]*N/S;  
        
        fPost[23] = phi[23]*N/S;
        fPost[22] = phi[22]*N/S;  

        fPost[ 9] = phi[ 9]*N/S;  
        fPost[ 8] = phi[ 8]*N/S;
        
        fPost[29] = phi[29]*N/S; 
        fPost[ 3] = phi[ 3]*N/S; 

        ////////////FINE UPPER LAYER 2////////////////////////
  
      } // end if (threadIdx.x == 1) 
      
      else { // if (threadIdx.x == 0) 
  
        //N=(f2[idx2].p[21]+f2[idx2].p[14]+f2[idx2].p[28]); new

        term1 = param_cy[29]*fPost[29] + param_cy[23]*fPost[23] +
                param_cy[16]*fPost[16] + param_cy[ 9]*fPost[ 9] +
                param_cy[ 3]*fPost[ 3] + param_cy[34]*fPost[34] +
                param_cy[30]*fPost[30] + param_cy[24]*fPost[24] +
                param_cy[17]*fPost[17] + param_cy[10]*fPost[10] +
                param_cy[ 4]*fPost[ 4] + param_cy[ 0]*fPost[ 0] +
                param_cy[35]*fPost[35] + param_cy[31]*fPost[31] +
                param_cy[25]*fPost[25] + param_cy[11]*fPost[11] +
                param_cy[ 5]*fPost[ 5] + param_cy[ 1]*fPost[ 1] +
                param_cy[ 2]*fPost[ 2] + param_cy[ 6]*fPost[ 6] +
                param_cy[12]*fPost[12] + param_cy[19]*fPost[19] +
                param_cy[26]*fPost[26] + param_cy[32]*fPost[32] +
                param_cy[36]*fPost[36] + param_cy[ 7]*fPost[ 7] +
                param_cy[13]*fPost[13] + param_cy[20]*fPost[20] +
                param_cy[27]*fPost[27] + param_cy[33]*fPost[33] +
                param_cy[14]*fPost[14] + param_cy[21]*fPost[21] +
                param_cy[28]*fPost[28];

        term2 = fPost[29] + fPost[23] + fPost[16] +
                fPost[ 9] + fPost[ 3] + fPost[34] +
                fPost[30] + fPost[24] + fPost[17] +
                fPost[10] + fPost[ 4] + fPost[ 0] +
                fPost[35] + fPost[31] + fPost[25] +
                fPost[11] + fPost[ 5] + fPost[ 1] +
                fPost[ 2] + fPost[ 6] + fPost[12] +
                fPost[19] + fPost[26] + fPost[32] +
                fPost[36] + fPost[ 7] + fPost[13] +
                fPost[20] + fPost[27] + fPost[33] +
                fPost[14] + fPost[21] + fPost[28] +
                fPost[18] + 
                (fPre[21] +  fPre[14]+ fPre[28]);


        N = -term1/(-3.*UNIT)-0.5*GRAVITY*DELTAT*(term2)/(-3.*UNIT);

        fPost[18] = fPost[18] - N + (fPre[21] + fPre[14] + fPre[28]);
  
        massapost = N +
                    fPost[29] + fPost[23] + fPost[16] +
                    fPost[ 9] + fPost[ 3] + fPost[34] +
                    fPost[30] + fPost[24] + fPost[17] +
                    fPost[10] + fPost[ 4] + fPost[ 0] +
                    fPost[35] + fPost[31] + fPost[25] +
                    fPost[11] + fPost[ 5] + fPost[ 1] +
                    fPost[ 2] + fPost[ 6] + fPost[12] +
                    fPost[19] + fPost[26] + fPost[32] +
                    fPost[36] + fPost[ 7] + fPost[13] +
                    fPost[20] + fPost[27] + fPost[33] +
                    fPost[14] + fPost[21] + fPost[28] +
                    fPost[18];

        ptildex = massapost*(utilde3) - 0.5*FORCEPOIS*DELTAT;

        ptildey = -3.*N*UNIT+
                  param_cy[29]*fPost[29] +
                  param_cy[23]*fPost[23] + param_cy[16]*fPost[16] +
                  param_cy[ 9]*fPost[ 9] + param_cy[ 3]*fPost[ 3] +
                  param_cy[34]*fPost[34] + param_cy[30]*fPost[30] +
                  param_cy[24]*fPost[24] + param_cy[17]*fPost[17] +
                  param_cy[10]*fPost[10] + param_cy[ 4]*fPost[ 4] +
                  param_cy[ 0]*fPost[ 0] + param_cy[35]*fPost[35] +
                  param_cy[31]*fPost[31] + param_cy[25]*fPost[25] +
                  param_cy[11]*fPost[11] + param_cy[ 5]*fPost[ 5] +
                  param_cy[ 1]*fPost[ 1] + param_cy[ 2]*fPost[ 2] +
                  param_cy[ 6]*fPost[ 6] + param_cy[12]*fPost[12] +
                  param_cy[19]*fPost[19] + param_cy[26]*fPost[26] +
                  param_cy[32]*fPost[32] + param_cy[36]*fPost[36] +
                  param_cy[ 7]*fPost[ 7] + param_cy[13]*fPost[13] +
                  param_cy[20]*fPost[20] + param_cy[27]*fPost[27] +
                  param_cy[33]*fPost[33] + param_cy[14]*fPost[14] +
                  param_cy[21]*fPost[21] + param_cy[28]*fPost[28];

        Etilde=(temptilde3)*massapost+0.5*(ptildex*ptildex+ptildey*ptildey)/massapost;
  
        Ox = param_cx[29]*fPost[29] + param_cx[23]*fPost[23] +
             param_cx[16]*fPost[16] + param_cx[ 9]*fPost[ 9] +
             param_cx[ 3]*fPost[ 3] + param_cx[34]*fPost[34] +
             param_cx[30]*fPost[30] + param_cx[24]*fPost[24] +
             param_cx[17]*fPost[17] + param_cx[10]*fPost[10] +
             param_cx[ 4]*fPost[ 4] + param_cx[ 0]*fPost[ 0] +
             param_cx[35]*fPost[35] + param_cx[31]*fPost[31] +
             param_cx[25]*fPost[25] + param_cx[11]*fPost[11] +
             param_cx[ 5]*fPost[ 5] + param_cx[ 1]*fPost[ 1] +
             param_cx[ 2]*fPost[ 2] + param_cx[ 6]*fPost[ 6] +
             param_cx[12]*fPost[12] + param_cx[19]*fPost[19] +
             param_cx[26]*fPost[26] + param_cx[32]*fPost[32] +
             param_cx[36]*fPost[36] + param_cx[ 7]*fPost[ 7] +
             param_cx[13]*fPost[13] + param_cx[20]*fPost[20] +
             param_cx[27]*fPost[27] + param_cx[33]*fPost[33] +
             param_cx[14]*fPost[14] + param_cx[21]*fPost[21] +
             param_cx[28]*fPost[28];
  
        OE = 0.5*(param_cx[29]*param_cx[29]+param_cy[29]*param_cy[29])*fPost[29]+
             0.5*(param_cx[23]*param_cx[23]+param_cy[23]*param_cy[23])*fPost[23]+
             0.5*(param_cx[16]*param_cx[16]+param_cy[16]*param_cy[16])*fPost[16]+
             0.5*(param_cx[ 9]*param_cx[ 9]+param_cy[ 9]*param_cy[ 9])*fPost[ 9]+
             0.5*(param_cx[ 3]*param_cx[ 3]+param_cy[ 3]*param_cy[ 3])*fPost[ 3]+
             0.5*(param_cx[34]*param_cx[34]+param_cy[34]*param_cy[34])*fPost[34]+
             0.5*(param_cx[30]*param_cx[30]+param_cy[30]*param_cy[30])*fPost[30]+
             0.5*(param_cx[24]*param_cx[24]+param_cy[24]*param_cy[24])*fPost[24]+
             0.5*(param_cx[17]*param_cx[17]+param_cy[17]*param_cy[17])*fPost[17]+
             0.5*(param_cx[10]*param_cx[10]+param_cy[10]*param_cy[10])*fPost[10]+
             0.5*(param_cx[ 4]*param_cx[ 4]+param_cy[ 4]*param_cy[ 4])*fPost[ 4]+
             0.5*(param_cx[ 0]*param_cx[ 0]+param_cy[ 0]*param_cy[ 0])*fPost[ 0]+
             0.5*(param_cx[35]*param_cx[35]+param_cy[35]*param_cy[35])*fPost[35]+
             0.5*(param_cx[31]*param_cx[31]+param_cy[31]*param_cy[31])*fPost[31]+
             0.5*(param_cx[25]*param_cx[25]+param_cy[25]*param_cy[25])*fPost[25]+
             0.5*(param_cx[11]*param_cx[11]+param_cy[11]*param_cy[11])*fPost[11]+
             0.5*(param_cx[ 5]*param_cx[ 5]+param_cy[ 5]*param_cy[ 5])*fPost[ 5]+
             0.5*(param_cx[ 1]*param_cx[ 1]+param_cy[ 1]*param_cy[ 1])*fPost[ 1]+
             0.5*(param_cx[ 2]*param_cx[ 2]+param_cy[ 2]*param_cy[ 2])*fPost[ 2]+
             0.5*(param_cx[ 6]*param_cx[ 6]+param_cy[ 6]*param_cy[ 6])*fPost[ 6]+
             0.5*(param_cx[12]*param_cx[12]+param_cy[12]*param_cy[12])*fPost[12]+
             0.5*(param_cx[19]*param_cx[19]+param_cy[19]*param_cy[19])*fPost[19]+ 
             0.5*(param_cx[26]*param_cx[26]+param_cy[26]*param_cy[26])*fPost[26]+
             0.5*(param_cx[32]*param_cx[32]+param_cy[32]*param_cy[32])*fPost[32]+
             0.5*(param_cx[36]*param_cx[36]+param_cy[36]*param_cy[36])*fPost[36]+
             0.5*(param_cx[ 7]*param_cx[ 7]+param_cy[ 7]*param_cy[ 7])*fPost[ 7]+
             0.5*(param_cx[13]*param_cx[13]+param_cy[13]*param_cy[13])*fPost[13]+
             0.5*(param_cx[20]*param_cx[20]+param_cy[20]*param_cy[20])*fPost[20]+
             0.5*(param_cx[27]*param_cx[27]+param_cy[27]*param_cy[27])*fPost[27]+
             0.5*(param_cx[33]*param_cx[33]+param_cy[33]*param_cy[33])*fPost[33]+
             0.5*(param_cx[14]*param_cx[14]+param_cy[14]*param_cy[14])*fPost[14]+
             0.5*(param_cx[21]*param_cx[21]+param_cy[21]*param_cy[21])*fPost[21]+
             0.5*(param_cx[28]*param_cx[28]+param_cy[28]*param_cy[28])*fPost[28];
  
        a1 = -2.*N*UNIT*UNIT;
        b1 = (29./2.)*(ptildex-Ox)*UNIT*UNIT;
        d1 = 3.*(ptildex-Ox);

        b2 = (Etilde-OE)*(29./2.)*UNIT*UNIT-(281./4.)*N*UNIT*UNIT*UNIT*UNIT;
        d2 = 3.*(Etilde-OE)-(29./2.)*N*UNIT*UNIT;

        E  = -d2/b2;

        px = (b1*d2-d1*b2)/(a1*b2);

        phi[15] = 1.+param_cx[15]*px+0.5*(param_cx[15]*param_cx[15]+param_cy[15]*param_cy[15])*E;   
        phi[22] = 1.+param_cx[22]*px+0.5*(param_cx[22]*param_cx[22]+param_cy[22]*param_cy[22])*E;
        phi[ 8] = 1.+param_cx[ 8]*px+0.5*(param_cx[ 8]*param_cx[ 8]+param_cy[ 8]*param_cy[ 8])*E;

        S = phi[15] + phi[22] + phi[ 8];

        fPost[15] = phi[15]*N/S;  
        fPost[22] = phi[22]*N/S;  
        fPost[ 8] = phi[ 8]*N/S;
  
      } // end if (threadIdx.x == 0)
      
    //__syncthreads();
 
    //////////////////////////////////////////////////////////
    // move populations from previous to current lattice
    
    p_nxt[( 0 * NX_d*NY_d) + idx_cur] = fPost[ 0];  //    curY;
    p_nxt[( 1 * NX_d*NY_d) + idx_cur] = fPost[ 1];  //    curY;
    p_nxt[( 2 * NX_d*NY_d) + idx_cur] = fPost[ 2];  //    curY;
    p_nxt[( 3 * NX_d*NY_d) + idx_cur] = fPost[ 3];  //    curY;
    p_nxt[( 4 * NX_d*NY_d) + idx_cur] = fPost[ 4];  //    curY;
    p_nxt[( 5 * NX_d*NY_d) + idx_cur] = fPost[ 5];  //    curY;
    p_nxt[( 6 * NX_d*NY_d) + idx_cur] = fPost[ 6];  //    curY;
    p_nxt[( 7 * NX_d*NY_d) + idx_cur] = fPost[ 7];  //    curY;
    p_nxt[( 8 * NX_d*NY_d) + idx_cur] = fPost[ 8];  //    curY;
    p_nxt[( 9 * NX_d*NY_d) + idx_cur] = fPost[ 9];  //    curY;
    p_nxt[(10 * NX_d*NY_d) + idx_cur] = fPost[10];  //    curY;
    p_nxt[(11 * NX_d*NY_d) + idx_cur] = fPost[11];  //    curY;
    p_nxt[(12 * NX_d*NY_d) + idx_cur] = fPost[12];  //    curY;
    p_nxt[(13 * NX_d*NY_d) + idx_cur] = fPost[13];  //    curY;
    p_nxt[(14 * NX_d*NY_d) + idx_cur] = fPost[14];  //    curY;
    p_nxt[(15 * NX_d*NY_d) + idx_cur] = fPost[15];  //    curY;
    p_nxt[(16 * NX_d*NY_d) + idx_cur] = fPost[16];  //    curY;
    p_nxt[(17 * NX_d*NY_d) + idx_cur] = fPost[17];  //    curY;
    p_nxt[(18 * NX_d*NY_d) + idx_cur] = fPost[18];  //    curY;
    p_nxt[(19 * NX_d*NY_d) + idx_cur] = fPost[19];  //    curY;
    p_nxt[(20 * NX_d*NY_d) + idx_cur] = fPost[20];  //    curY;
    p_nxt[(21 * NX_d*NY_d) + idx_cur] = fPost[21];  //    curY;
    p_nxt[(22 * NX_d*NY_d) + idx_cur] = fPost[22];  //    curY;
    p_nxt[(23 * NX_d*NY_d) + idx_cur] = fPost[23];  //    curY;
    p_nxt[(24 * NX_d*NY_d) + idx_cur] = fPost[24];  //    curY;
    p_nxt[(25 * NX_d*NY_d) + idx_cur] = fPost[25];  //    curY;
    p_nxt[(26 * NX_d*NY_d) + idx_cur] = fPost[26];  //    curY;
    p_nxt[(27 * NX_d*NY_d) + idx_cur] = fPost[27];  //    curY;
    p_nxt[(28 * NX_d*NY_d) + idx_cur] = fPost[28];  //    curY;
    p_nxt[(29 * NX_d*NY_d) + idx_cur] = fPost[29];  //    curY;
    p_nxt[(30 * NX_d*NY_d) + idx_cur] = fPost[30];  //    curY;
    p_nxt[(31 * NX_d*NY_d) + idx_cur] = fPost[31];  //    curY;
    p_nxt[(32 * NX_d*NY_d) + idx_cur] = fPost[32];  //    curY;
    p_nxt[(33 * NX_d*NY_d) + idx_cur] = fPost[33];  //    curY;
    p_nxt[(34 * NX_d*NY_d) + idx_cur] = fPost[34];  //    curY;
    p_nxt[(35 * NX_d*NY_d) + idx_cur] = fPost[35];  //    curY;
    p_nxt[(36 * NX_d*NY_d) + idx_cur] = fPost[36];  //    curY;
    
  } // end if (idx_cur < LSIZEX_d*NY_d)
 
} 

#endif /* _BC_CUDA_H */

