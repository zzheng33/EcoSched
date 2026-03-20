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

#ifndef _COLLIDE_CUDA_H
#define _COLLIDE_CUDA_H

#define FLOPs 6472

////////////////////////////////////////////////////////////////////////////////
// Device function COLLIDE.

__device__ void _collide_site(
        data_t localPop[NPOP],
        data_t * __restrict__ const p_nxt,
        const data_t * __restrict__ const p_prv,
        int idx_cur,
        int yheight
  ){ 
  int bits;
  unsigned int i;
  
  data_t projection2xx,   projection2xy,   projection2yy,
      	 projection3xxx,  projection3xxy,  projection3xyy,
      	 projection3yyy,  projection4xxxx, projection4xxxy,
      	 projection4xxyy, projection4xyyy, projection4yyyy;
  
  data_t projection2xy_t_2,   projection3xxy_t_3,  projection3xyy_t_3,
         projection4xxxy_t_4, projection4xxyy_t_6, projection4xyyy_t_4;
  
  data_t finalProj0,     finalProj1x,    finalProj1y,
      	 finalProj2xx,   finalProj2xy,   finalProj2yy,
      	 finalProj3xxx,  finalProj3xxy,  finalProj3xyy,
      	 finalProj3yyy,  finalProj4xxxx, finalProj4xxxy,
      	 finalProj4xxyy, finalProj4xyyy, finalProj4yyyy;
  
  data_t finalProj2xy_t_2,   finalProj3xxy_t_3,  finalProj3xyy_t_3,
         finalProj4xxxy_t_4, finalProj4xxyy_t_6, finalProj4xyyy_t_4;
  
  data_t  rhoi, scalar, temprhoi, tempi, theta, forcey, omeganow,
          modc2, modu2, scalar2, theta_1, theta_1_3, theta_1_04, theta_1_2_08;
  
  data_t  tau1, tau2, tau3;
  data_t  tau1i, tau2i, tau3i;
  
  data_t  Hermite0,   Hermite1, Hermite2,   Hermite3, Hermite4;
  data_t  eqHermite2, eqHermite3, eqHermite4, Collisional;
  
  data_t  isBoundary, isBulk;  
  
  data_t  rho, v, u, temp;
  
  data_t popTemp;

  	//////////////////////////////////////////////////////////////////////////////
  	// Computation of the projections, rho, u and v.
  
  	rho = VZERO; v = VZERO; u = VZERO;
  
  	for(i = 0; i < NPOP; i++) {
  	  popTemp = localPop[i]; 
    
  	  rho = rho + popTemp;    	      	      	      	  
  	  u   = u   + popTemp * param_cx[i];
  	  v   = v   + popTemp * param_cy[i];
  	}
  
  	//////////////////////////////////////////////////////////////////////////////
  	// Temperature is computed.
  
  	rhoi = VONES / rho;
  	u    = u * rhoi;
  	v    = v * rhoi;
  
  	//////////////////////////////////////////////////////////////////////////////
  	// isBulk is computed
  	//yheight  = blockIdx.x * blockDim.x + threadIdx.x;
  	bits   = 8*sizeof(int) - 1;
  	isBulk = 1 - ( ( ((yheight-3) >> bits) | ((GSIZEY_d-3-yheight-1) >> bits) ) & 1);
  
  	projection2xx   = VZERO;
  	projection2xy   = VZERO;
  	projection2yy   = VZERO;
  	projection3xxx  = VZERO;
  	projection3xxy  = VZERO;
  	projection3xyy  = VZERO;
  	projection3yyy  = VZERO;
  	projection4xxxx = VZERO;
  	projection4xxxy = VZERO;
  	projection4xxyy = VZERO;
  	projection4xyyy = VZERO;
  	projection4yyyy = VZERO;
  
  	temp = VZERO;
  
  	for(i = 0; i < NPOP; i++) { 
  	  popTemp = localPop[i];      
  
  	  scalar =  (param_cx[ i ] - u) * (param_cx[ i ] - u) +
  	            (param_cy[ i ] - v) * (param_cy[ i ] - v);
  	  temp = temp + VHALF*scalar*popTemp;
    
  	  projection2xx   += popTemp * param_H2xx[ i ];
  	  projection2xy   += popTemp * param_H2xy[ i ];
  	  projection2yy   += popTemp * param_H2yy[ i ];
    
  	  projection3xxx  += popTemp * param_H3xxx[ i ];
  	  projection3xxy  += popTemp * param_H3xxy[ i ];
  	  projection3xyy  += popTemp * param_H3xyy[ i ];
  	  projection3yyy  += popTemp * param_H3yyy[ i ];
    
  	  projection4xxxx += popTemp * param_H4xxxx[ i ];
  	  projection4xxxy += popTemp * param_H4xxxy[ i ];
  	  projection4xxyy += popTemp * param_H4xxyy[ i ];
  	  projection4xyyy += popTemp * param_H4xyyy[ i ];
  	  projection4yyyy += popTemp * param_H4yyyy[ i ];
  	}
  
  	temp = temp * rhoi;
  
  	//////////////////////////////////////////////////////////////////////////////
  	// WARNING: Here we assume that tau's are the same at all boundary sites...
  
  	tempi      = VONES / temp;
  
  	isBoundary = VONES - isBulk;    
  
  	temprhoi   = rhoi * tempi ;
  
  	tau1  = (VHALF + VCONST1 * temprhoi) * isBulk + VTAU1B * isBoundary;
  	tau1i = VHALF / tau1;
  
  	tau2  = (VHALF + VCONST2 * temprhoi) * isBulk + VTAU2B * isBoundary;
  	tau2i = _06 / tau2 ;
  
  	tau3  = ( VHALF + VCONST3 ) * isBulk + VTAU3B * isBoundary;
  	tau3i = _24 / tau3 ;
  
  	//----- replacement for SHIFT ------------------------------------------------
  	omeganow = tau1 * rhoi;
  
  	forcey   = GRAVITY * DELTAT * rho;
  
  	u = u + forcex * omeganow;
  	v = v + forcey * omeganow;    
  	//----- replacement for adjustTemp -------------------------------------------
  	temp = temp + VHALF*(VONES-tau1)*tau1*(forcex*forcex+forcey*forcey) * rhoi*rhoi;
  	//---- replacement for equili ------------------------------------------------
  	//---- WARNING: equili results are NOT used -> no replacement...
  
  	//////////////////////////////////////////////////////////////////////////////
  	// Following variables needed for the final projection.
  
  	modu2   = u * u + v * v;
  	theta   = temp;
  	theta_1 = theta - VONES;
  	theta_1_3 = theta_1 * VTHRE;
  	theta_1_04 = theta_1 * _04;
  	theta_1_2_08 = theta_1 * theta_1 * _08;
  	projection2xy_t_2 = projection2xy * VTWOS;
  	projection3xxy_t_3 = projection3xxy * VTHRE;
  	projection3xyy_t_3 = projection3xyy * VTHRE;
  	projection4xxxy_t_4 = projection4xxxy * VFOUR;
  	projection4xxyy_t_6 = projection4xxyy * VSIXS;
  	projection4xyyy_t_4 = projection4xyyy * VFOUR;
  
  	for(i = 0; i < NPOP; i++) {
  	  scalar  = (param_cx[ i ] * u + param_cy[ i ]  * v);
  	  scalar2 = scalar * scalar;
  	  modc2   = (param_cx[ i ] * param_cx[ i ] + param_cy[ i ] * param_cy[ i ]);
    
  	  Hermite2 = projection2xx     * param_H2xx[ i ] +
  	             projection2xy_t_2 * param_H2xy[ i ] +
  	             projection2yy     * param_H2yy[ i ];
    
  	  Hermite3 = projection3xxx     * param_H3xxx[ i ] +
	               projection3xxy_t_3 * param_H3xxy[ i ] +
	               projection3xyy_t_3 * param_H3xyy[ i ] +
	               projection3yyy     * param_H3yyy[ i ];
    
	    Hermite4 = projection4xxxx * param_H4xxxx[ i ]     +
	               projection4xxxy_t_4 * param_H4xxxy[ i ] +
	               projection4xxyy_t_6 * param_H4xxyy[ i ] +
      		       projection4xyyy_t_4 * param_H4xyyy[ i ] +
               	 projection4yyyy * param_H4yyyy[ i ];
    
      eqHermite2 = rho * 
          	       (
                	   (scalar2 - modu2 ) +
                	   theta_1 * (modc2 - VTWOS)
                       );
    
      eqHermite3 = rho *
    	             ( scalar *
                     ( scalar2 -
                       VTHRE  * modu2 +
          		         theta_1_3 * (modc2 - VFOUR)
          		       )
          	       );
    
      eqHermite4 = VTWO4 * rho *
                   (
                     ( scalar2 * scalar2 -
                       VSIXS  * scalar2 * modu2 +
                       VTHRE  * modu2 * modu2
                     ) *
                     ( _24 ) +
                     ( theta_1_04 ) *
                     ( (modc2 - VFOUR) * (scalar2 - modu2) -
                        VTWOS * scalar2
                     ) +
                     ( theta_1_2_08 ) *
                     ( modc2 * modc2 - VEIGH * modc2 + VEIGH )
                   );
    
    	Collisional = -forcex * param_cx[ i ] -
                     forcey * param_cy[ i ] +
                     (Hermite2 - eqHermite2) * ( tau1i ) +
                     (Hermite3 - eqHermite3) * ( tau2i ) +
                     (Hermite4 - eqHermite4) * ( tau3i );
    
      localPop[ i ] =  localPop[ i ] - Collisional * param_ww[ i ];
  	}
  
  	//////////////////////////////////////////////////////////////////////////////
  
  	finalProj0     = VZERO; finalProj1x    = VZERO; finalProj1y    = VZERO;
  	finalProj2xx   = VZERO; finalProj2xy   = VZERO; finalProj2yy   = VZERO;
  	finalProj3xxx  = VZERO; finalProj3xxy  = VZERO; finalProj3xyy  = VZERO;
  	finalProj3yyy  = VZERO; finalProj4xxxx = VZERO; finalProj4xxxy = VZERO;
  	finalProj4xxyy = VZERO; finalProj4xyyy = VZERO; finalProj4yyyy = VZERO;
  
  	for(i = 0; i < NPOP; i++) {
  	  popTemp = localPop[ i ];
  
  	  finalProj0     += popTemp * param_H0[ i ];
    
  	  finalProj1x    += popTemp * param_H1x[ i ];
  	  finalProj1y    += popTemp * param_H1y[ i ];
    
  	  finalProj2xx   += popTemp * param_H2xx[ i ];
  	  finalProj2xy   += popTemp * param_H2xy[ i ];
  	  finalProj2yy   += popTemp * param_H2yy[ i ];
    
  	  finalProj3xxx  += popTemp * param_H3xxx[ i ];
  	  finalProj3xxy  += popTemp * param_H3xxy[ i ];
  	  finalProj3xyy  += popTemp * param_H3xyy[ i ];
  	  finalProj3yyy  += popTemp * param_H3yyy[ i ];
    
  	  finalProj4xxxx += popTemp * param_H4xxxx[ i ];
  	  finalProj4xxxy += popTemp * param_H4xxxy[ i ];
  	  finalProj4xxyy += popTemp * param_H4xxyy[ i ];
  	  finalProj4xyyy += popTemp * param_H4xyyy[ i ];
  	  finalProj4yyyy += popTemp * param_H4yyyy[ i ];
  	}
    
  	//////////////////////////////////////////////////////////////////////////////

  	finalProj2xy_t_2 = finalProj2xy * VTWOS;
  	finalProj3xxy_t_3 = finalProj3xxy * VTHRE;
  	finalProj3xyy_t_3 = finalProj3xyy * VTHRE;
  	finalProj4xxxy_t_4 = finalProj4xxxy * VFOUR;
  	finalProj4xxyy_t_6 = finalProj4xxyy * VSIXS;
  	finalProj4xyyy_t_4 = finalProj4xyyy * VFOUR;

  	for(i = 0; i < NPOP; i++) {
  	  Hermite0 =  finalProj0 * param_H0[ i ];
    
  	  Hermite1 =  finalProj1x * param_H1x[ i ] +
          	      finalProj1y * param_H1y[ i ];
    
  	  Hermite2 =  finalProj2xx     * param_H2xx[ i ] +
              	      finalProj2xy_t_2 * param_H2xy[ i ] +
              	      finalProj2yy     * param_H2yy[ i ];
    
  	  Hermite3 =  finalProj3xxx     * param_H3xxx[ i ] +
              	      finalProj3xxy_t_3 * param_H3xxy[ i ] +
                      finalProj3xyy_t_3 * param_H3xyy[ i ] +
                      finalProj3yyy     * param_H3yyy[ i ];
    
  	  Hermite4 =  finalProj4xxxx     * param_H4xxxx[ i ] +
                      finalProj4xxxy_t_4 * param_H4xxxy[ i ] +
                      finalProj4xxyy_t_6 * param_H4xxyy[ i ] +
                      finalProj4xyyy_t_4 * param_H4xyyy[ i ] +
                      finalProj4yyyy     * param_H4yyyy[ i ];
    
    
  	  p_nxt[(  i * NX_d*NY_d) + idx_cur] = 
                      ( Hermite0 + Hermite1 + Hermite2 * ( _02 ) +
                        Hermite3 * ( _06 )  + Hermite4 * ( _24 )
                      ) * param_ww[i];
  	}
  //}
}

////////////////////////////////////////////////////////////////////////////////

__device__ void _collide(
        data_t * __restrict__ const p_nxt,
        const data_t * __restrict__ const p_prv,
        int idx_cur,
        int yheight
  ){

  int i;
  data_t  localPop[NPOP];   // Array where local populations are stored
                              // during operations.

  for(i = 0; i < NPOP; i++) {
    localPop[i] = p_prv[ ( i * NX_d*NY_d) + idx_cur ];
  }

  _collide_site(localPop, p_nxt, p_prv, idx_cur, yheight);
}

////////////////////////////////////////////////////////////////////////////////

__global__ void collideT (
              data_t * __restrict__ const p_nxt,
        const data_t * __restrict__ const p_prv,
        const int yoff
  ) {
    int idx_cur, yheight;
    idx_cur =  ( blockIdx.y  * blockDim.y * NY_d ) +   // Offset columns.
               ( blockIdx.x  * blockDim.x      ) +   // Offset Y block.
               ( threadIdx.y * NY_d              ) +   // Index of Y thread.
               ( threadIdx.x                   );    // Index of X thread.

    if( (threadIdx.x < blockDim.x) && ((blockIdx.y*blockDim.y+threadIdx.y) < LSIZEX_d) ) {
      yheight  = yoff + (blockIdx.x * blockDim.x + threadIdx.x + LSIZEY_d-hy-1);
      _collide(p_nxt,p_prv,idx_cur,yheight);
    }
  }


__global__ void collideB (
              data_t * __restrict__ const p_nxt,
        const data_t * __restrict__ const p_prv,
        const int yoff
  ) {
    int idx_cur, yheight;
    idx_cur =  ( blockIdx.y  * blockDim.y * NY_d ) +   // Offset columns.
               ( blockIdx.x  * blockDim.x      ) +   // Offset Y block.
               ( threadIdx.y * NY_d              ) +   // Index of Y thread.
               ( threadIdx.x                   );    // Index of X thread.

    if( (threadIdx.x < blockDim.x) && ((blockIdx.y*blockDim.y+threadIdx.y) < LSIZEX_d) ) {
      yheight  = yoff + (blockIdx.x * blockDim.x + threadIdx.x);
      _collide(p_nxt,p_prv,idx_cur,yheight);
    }
  }


#endif /* _COLLIDE_CUDA_H */
