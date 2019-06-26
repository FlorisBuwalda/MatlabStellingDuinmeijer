/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
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
 */
#include <iostream>
#include <thread>
#include <mutex>
#include <cassert>
#include <condition_variable>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include "cuda.h"
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
 /* Includes, cuda */
#include <cuda.h>
#include <cublas_v2.h>
#include <windows.h>
#define CLOCK_REALTIME 0
//struct timespec { long tv_sec; long tv_nsec; };    //header par

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cyclicbarrier.hpp>

#ifndef __CUDACC__  
#define __CUDACC__
#endif
using namespace std;

 // Convenience function for checking CUDA runtime API results
 // can be wrapped around any runtime API call. No-op in release builds.
inline

cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
 		assert(result == cudaSuccess);
	}
#endif
	return result;
}

////// parameters
constexpr int BLOCK_SIZE_x = 32;  // number of threads per block in x-dir
constexpr int BLOCK_SIZE_y = 32;  // number of threads per block in y-dir

//constexpr int threads[] = { 1,0 };
//constexpr int iter = 10;
//constexpr int ni = 1;
//constexpr int n = 3 * 32 * ni + 2;
//constexpr int L = n + 1;          //Domain length
//constexpr int W = n + 1;          //Domain width
//constexpr float dx_h = (float)W / ((float)(n + 1));                // inter grid distance in x - direction
//constexpr float dy_h = (float)L / ((float)(n + 1));                // inter grid distance in y - direction

int n;
constexpr float dx_h = 1;
constexpr float dy_h = 1;


__constant__ int  n_d;
__constant__ float dx;
__constant__ float dy;

__constant__ float Hstart ;    //Rest water depth
constexpr float Hstart_h = 1;

constexpr float g_h = (float)9.8;   // gravitational constant
__constant__ float g;                

constexpr float tstep = 1;             // maximum timestep
constexpr float dt = (float).01;                // first step is maximum timestep





__constant__ float cf;
constexpr float cf_h = 0;                    // Bottom friction factor

constexpr float droppar_h = .5;
__constant__ float droppar;

constexpr int ndrops = 1;              // maximum number of water drops
constexpr int dropstep = 5;            // drop interval

bool timer = false;
constexpr float safety = (float).9;



	__global__ void update( float *h, __int8 *upos, __int8 *vpos, float *U, float *V, float dt )
{
	
		
		__shared__      float s_h[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 4-wide halo
		//__shared__     float s_hy[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 4-wide halo
		//__shared__     float s_hx[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 4-wide halo
		__shared__  __int8 s_upos[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 2-wide halo
		__shared__  __int8 s_vpos[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 2-wide halo
		__shared__      float s_U[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 2-wide halo
		__shared__      float s_V[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 2-wide halo

		//int i = threadIdx.x;
		//int j = blockIdx.x*blockDim.y + threadIdx.y;
		int j = blockIdx.y*blockDim.y + threadIdx.y;
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		int si = threadIdx.x + 1; // local i for shared memory access + halo offset
		int sj = threadIdx.y + 1; // local j for shared memory access
		float utemp, vtemp;
		float s_hx, s_hxmin, s_hy, s_hymin;
		int globalIdx =  (j+1) * n_d + i+1;
		
		//Boundaries
		if (threadIdx.x ==0) {
			
			   s_h[sj][si-1] =    h[globalIdx-1];
			s_upos[sj][si-1] = upos[globalIdx-1];
			s_vpos[sj][si-1] = vpos[globalIdx-1];
			   s_U[sj][si-1] =	  U[globalIdx-1];
			   s_V[sj][si-1] =    V[globalIdx-1];

			         s_h[sj][si + BLOCK_SIZE_x ] =    h[globalIdx + BLOCK_SIZE_x ];
				  s_upos[sj][si + BLOCK_SIZE_x ] = upos[globalIdx + BLOCK_SIZE_x ];
				  s_vpos[sj][si + BLOCK_SIZE_x ] = vpos[globalIdx + BLOCK_SIZE_x ];
				     s_U[sj][si + BLOCK_SIZE_x ] =    U[globalIdx + BLOCK_SIZE_x ];
				     s_V[sj][si + BLOCK_SIZE_x ] =    V[globalIdx + BLOCK_SIZE_x ];
		}
		if (threadIdx.y==0) {
			s_h[sj-1][si] =    h[globalIdx - n_d];
		 s_upos[sj-1][si] = upos[globalIdx - n_d];
		 s_vpos[sj-1][si] = vpos[globalIdx - n_d];
			s_U[sj-1][si] =    U[globalIdx - n_d];
			s_V[sj-1][si] =    V[globalIdx - n_d];

			s_h[sj+BLOCK_SIZE_y][si] =    h[globalIdx +n_d*(BLOCK_SIZE_y)];
		 s_upos[sj+BLOCK_SIZE_y][si] = upos[globalIdx +n_d*(BLOCK_SIZE_y)];
		 s_vpos[sj+BLOCK_SIZE_y][si] = vpos[globalIdx +n_d*(BLOCK_SIZE_y)];
			s_U[sj+BLOCK_SIZE_y][si] =    U[globalIdx +n_d*(BLOCK_SIZE_y)];
			s_V[sj+BLOCK_SIZE_y][si] =    V[globalIdx +n_d*(BLOCK_SIZE_y)];
		}

		// copy global variables into shared memory
		   s_h[sj][si] = h[globalIdx];
		s_upos[sj][si] = upos[globalIdx];
		s_vpos[sj][si] = vpos[globalIdx]; 
		   s_U[sj][si] = U[globalIdx];
		   s_V[sj][si] = V[globalIdx];

		__syncthreads();
		
		// fill in periodic images in shared memory array 
		//if (i < 4) {
		//	s_f[sj][si - 4] = s_f[sj][si + mx - 5];
		//	s_f[sj][si + mx] = s_f[sj][si + 1];
		//}

		//__syncthreads();

		//update Hx and Hy
	/*	 s_hx[sj][si] =
			s_upos[sj][si] * s_h[sj][si]
			+ (1 - s_upos[sj][si]) *s_h[sj][si + 1];

		s_hy[sj][si] =
			s_vpos[sj][si] * s_h[sj][si]
			+ (1 - s_vpos[sj][si]) *s_h[sj+1][si];*/

		//update U (no sync necessary)
		utemp = s_U[sj][si] - g * dt / dx * (s_h[sj][si + 1] - s_h[sj][si])
			      - s_upos[sj][si] * dt / dx * (s_U[sj][si] - s_U[sj][si - 1])*(s_U[sj][si] + s_U[sj][si - 1]) / 2
			      - s_vpos[sj][si] * dt / dy * (s_U[sj][si] - s_U[sj - 1][si])*(s_V[sj-1][si] + s_V[sj - 1][si+1]) / 2
			- (1 - s_upos[sj][si]) * dt / dx * (s_U[sj][si + 1] - s_U[sj][si])*(s_U[sj][si] + s_U[sj][si + 1]) / 2
			- (1 - s_vpos[sj][si]) * dt / dy * (s_U[sj + 1][si] - s_U[sj][si])*(s_V[sj][si] + s_V[sj ][si+1]) / 2;

		__syncthreads();

		//write temp values to shared memory after sync and update upos
		s_U[sj][si] = utemp;
		s_upos[sj][si] = (__int8)(utemp > 0);

		__syncthreads();
		//now that 
		 s_hx =
			          (s_upos[sj][si] * s_h[sj][si]
				+ (1 - s_upos[sj][si]) *s_h[sj][si + 1]);

		 s_hxmin =
			           (s_upos[sj][si - 1] * s_h[sj][si - 1]
				+ (1 - s_upos[sj][si - 1]) *s_h[sj][si]);

		//write back to global memory
		U[globalIdx] = utemp;

		//update V
		vtemp = s_V[sj][si] - g * dt / dy * (s_h[sj + 1][si] - s_h[sj][si])
			- s_vpos[sj][si] * dt / dy * (s_V[sj][si] - s_V[sj - 1][si])*(s_V[sj][si] + s_V[sj - 1][si]) / 2
			- s_upos[sj][si] * dt / dx * (s_V[sj][si] - s_V[sj][si-1])  *(s_U[sj + 1][si - 1] + s_U[sj][si - 1]) / 2
			- (1-s_vpos[sj][si]) * dt / dy * (s_V[sj+1][si] - s_V[sj ][si])*(s_V[sj][si] + s_V[sj + 1][si]) / 2
			- (1-s_upos[sj][si]) * dt / dx * (s_V[sj][si+1] - s_V[sj][si])  *(s_U[sj + 1][si ] + s_U[sj][si ]) / 2;

		__syncthreads();

		s_V[sj][si] =  vtemp;
		s_vpos[sj][si]= (__int8)(vtemp > 0);

		__syncthreads();

		V[globalIdx] = vtemp;
		
		//calculate hy
		s_hy =
			s_vpos[sj][si] * s_h[sj][si]
			+ (1 - s_vpos[sj][si]) *s_h[sj + 1][si];

		s_hymin =
			s_vpos[sj - 1][si] * s_h[sj - 1][si]
			+ (1 - s_vpos[sj - 1][si]) *s_h[sj][si];

		// update h
		s_h[sj][si] = s_h[sj][si] - dt / dx * (s_hx * s_U[sj][si] - s_hxmin * s_U[sj][si - 1])
								  - dt / dy * (s_hy * s_V[sj][si] - s_hymin * s_V[sj - 1][si]);

			/*s_h[sj][si] = s_h[sj][si] - dt / dx * s_hx[sj][si] * s_U[sj][si] - s_hx[sj][si - 1] * s_U[sj][si - 1]
				- dt / dy * s_hy[sj][si] * s_V[sj][si] - s_hy[sj - 1][si] * s_V[sj - 1][si];*/

		__syncthreads();
		//write h back to global memory
		h[globalIdx] = s_h[sj][si];
		
		__syncthreads();
			
	}

	__global__ void updatenobool(float *h, float *U, float *V, float dt)
	{


		__shared__      float s_h[BLOCK_SIZE_y + 2][BLOCK_SIZE_x + 2]; // 4-wide halo
		//__shared__     float s_hy[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 4-wide halo
		//__shared__     float s_hx[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 4-wide halo	
		__shared__      float s_U[BLOCK_SIZE_y + 2][BLOCK_SIZE_x + 2]; // 2-wide halo
		__shared__      float s_V[BLOCK_SIZE_y + 2][BLOCK_SIZE_x + 2]; // 2-wide halo

		//int i = threadIdx.x;
		//int j = blockIdx.x*blockDim.y + threadIdx.y;
		int j = blockIdx.y*blockDim.y + threadIdx.y;
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		int si = threadIdx.x + 1; // local i for shared memory access + halo offset
		int sj = threadIdx.y + 1; // local j for shared memory access
		float utemp, vtemp;
		float s_hx, s_hxmin, s_hy, s_hymin;
		__int8 s_upos, s_vpos;
		int globalIdx = (j + 1) * n_d + i + 1;

		//Boundaries
		if (threadIdx.x == 0) {

			s_h[sj][si - 1] = h[globalIdx - 1];
			s_U[sj][si - 1] = U[globalIdx - 1];
			s_V[sj][si - 1] = V[globalIdx - 1];

			s_h[sj][si + BLOCK_SIZE_x] = h[globalIdx + BLOCK_SIZE_x];
			s_U[sj][si + BLOCK_SIZE_x] = U[globalIdx + BLOCK_SIZE_x];
			s_V[sj][si + BLOCK_SIZE_x] = V[globalIdx + BLOCK_SIZE_x];
		}
		if (threadIdx.y == 0) {
			s_h[sj - 1][si] = h[globalIdx - n_d];
			
			s_U[sj - 1][si] = U[globalIdx - n_d];
			s_V[sj - 1][si] = V[globalIdx - n_d];

			s_h[sj + BLOCK_SIZE_y][si] = h[globalIdx + n_d * (BLOCK_SIZE_y)];
			
			s_U[sj + BLOCK_SIZE_y][si] = U[globalIdx + n_d * (BLOCK_SIZE_y)];
			s_V[sj + BLOCK_SIZE_y][si] = V[globalIdx + n_d * (BLOCK_SIZE_y)];
		}

		// copy global variables into shared memory
		s_h[sj][si] = h[globalIdx];
		
		s_U[sj][si] = U[globalIdx];
		s_V[sj][si] = V[globalIdx];
		s_upos = s_U[sj][si] >= 0;
		s_vpos = s_V[sj][si] >= 0;
		__syncthreads();
		
		
		// fill in periodic images in shared memory array 
		//if (i < 4) {
		//	s_f[sj][si - 4] = s_f[sj][si + mx - 5];
		//	s_f[sj][si + mx] = s_f[sj][si + 1];
		//}

		//__syncthreads();

		//update Hx and Hy
	/*	 s_hx[sj][si] =
			s_upos[sj][si] * s_h[sj][si]
			+ (1 - s_upos[sj][si]) *s_h[sj][si + 1];
		
		s_hy[sj][si] =
			s_vpos[sj][si] * s_h[sj][si]
			+ (1 - s_vpos[sj][si]) *s_h[sj+1][si];*/

			//update U (no sync necessary)
		utemp = s_U[sj][si] - g * dt / dx * (s_h[sj][si + 1] - s_h[sj][si])
			- s_upos * dt / dx * (s_U[sj][si] - s_U[sj][si - 1])*(s_U[sj][si] + s_U[sj][si - 1]) / 2
			- s_vpos * dt / dy * (s_U[sj][si] - s_U[sj - 1][si])*(s_V[sj - 1][si] + s_V[sj - 1][si + 1]) / 2
			- (1 - s_upos) * dt / dx * (s_U[sj][si + 1] - s_U[sj][si])*(s_U[sj][si] + s_U[sj][si + 1]) / 2
			- (1 - s_vpos) * dt / dy * (s_U[sj + 1][si] - s_U[sj][si])*(s_V[sj][si] + s_V[sj][si + 1]) / 2;

		__syncthreads();

		//write temp values to shared memory after sync and update upos
		s_U[sj][si] = utemp;
		s_upos = (utemp > 0);

		__syncthreads();
		//now that 
		s_hx =
			(s_upos * s_h[sj][si]
				+ (1 - s_upos) *s_h[sj][si + 1]);

		s_hxmin =
			((s_U[sj][si - 1]>=0) * s_h[sj][si - 1]
				+ (s_U[sj][si - 1]<0) *s_h[sj][si]);

		//write back to global memory
		U[globalIdx] = utemp;

		//update V
		vtemp = s_V[sj][si] - g * dt / dy * (s_h[sj + 1][si] - s_h[sj][si])
			- s_vpos * dt / dy * (s_V[sj][si] - s_V[sj - 1][si])*(s_V[sj][si] + s_V[sj - 1][si]) / 2
			- s_upos * dt / dx * (s_V[sj][si] - s_V[sj][si - 1])  *(s_U[sj + 1][si - 1] + s_U[sj][si - 1]) / 2
			- (1 - s_vpos) * dt / dy * (s_V[sj + 1][si] - s_V[sj][si])*(s_V[sj][si] + s_V[sj + 1][si]) / 2
			- (1 - s_upos) * dt / dx * (s_V[sj][si + 1] - s_V[sj][si])  *(s_U[sj + 1][si] + s_U[sj][si]) / 2;

		__syncthreads();

		s_V[sj][si] = vtemp;
		s_vpos = (vtemp >= 0);

		__syncthreads();

		V[globalIdx] = vtemp;

		//calculate hy
		s_hy =
			s_vpos * s_h[sj][si]
			+ (1 - s_vpos) *s_h[sj + 1][si];

		s_hymin =
			(s_V[sj - 1][si]>=0) * s_h[sj - 1][si]
			+ (s_V[sj - 1][si] < 0) *s_h[sj][si];

		__syncthreads();
		// update h
		s_h[sj][si] = s_h[sj][si] - dt / dx * (s_hx * s_U[sj][si] - s_hxmin * s_U[sj][si - 1])
			- dt / dy * (s_hy * s_V[sj][si] - s_hymin * s_V[sj - 1][si]);

		/*s_h[sj][si] = s_h[sj][si] - dt / dx * s_hx[sj][si] * s_U[sj][si] - s_hx[sj][si - 1] * s_U[sj][si - 1]
			- dt / dy * s_hy[sj][si] * s_V[sj][si] - s_hy[sj - 1][si] * s_V[sj - 1][si];*/

		__syncthreads();
		//write h back to global memory
		h[globalIdx] = s_h[sj][si];

		__syncthreads();

	}

	__global__ void updateborders(float *h, float *U, float *V, float dt, int iter)
	{


		__shared__      float s_h[BLOCK_SIZE_y + 2][BLOCK_SIZE_x + 2]; // 4-wide halo
		//__shared__     float s_hy[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 4-wide halo
		//__shared__     float s_hx[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 4-wide halo	
		__shared__      float s_U[BLOCK_SIZE_y + 2][BLOCK_SIZE_x + 2]; // 2-wide halo
		__shared__      float s_V[BLOCK_SIZE_y + 2][BLOCK_SIZE_x + 2]; // 2-wide halo

		//int i = threadIdx.x;
		//int j = blockIdx.x*blockDim.y + threadIdx.y;
		int j = blockIdx.y*blockDim.y + threadIdx.y;
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		int si = threadIdx.x + 1; // local i for shared memory access + halo offset
		int sj = threadIdx.y + 1; // local j for shared memory access
		float utemp, vtemp;
		float s_hx, s_hxmin, s_hy, s_hymin;
		__int8 s_upos, s_vpos;
		int globalIdx = (j + 1) * n_d + i + 1;

		//Boundaries
		if (threadIdx.x == 0) {

			s_h[sj][si - 1] = h[globalIdx - 1];
			s_U[sj][si - 1] = U[globalIdx - 1];
			s_V[sj][si - 1] = V[globalIdx - 1];

			s_h[sj][si + BLOCK_SIZE_x] = h[globalIdx + BLOCK_SIZE_x];
			s_U[sj][si + BLOCK_SIZE_x] = U[globalIdx + BLOCK_SIZE_x];
			s_V[sj][si + BLOCK_SIZE_x] = V[globalIdx + BLOCK_SIZE_x];
		}
		if (threadIdx.y == 0) {
			s_h[sj - 1][si] = h[globalIdx - n_d];
			s_U[sj - 1][si] = U[globalIdx - n_d];
			s_V[sj - 1][si] = V[globalIdx - n_d];

			s_h[sj + BLOCK_SIZE_y][si] = h[globalIdx + n_d * (BLOCK_SIZE_y)];
			s_U[sj + BLOCK_SIZE_y][si] = U[globalIdx + n_d * (BLOCK_SIZE_y)];
			s_V[sj + BLOCK_SIZE_y][si] = V[globalIdx + n_d * (BLOCK_SIZE_y)];
		}

		// copy global variables into shared memory
		s_h[sj][si] = h[globalIdx];
		s_U[sj][si] = U[globalIdx];
		s_V[sj][si] = V[globalIdx];
		s_upos = s_U[sj][si] >= 0;
		s_vpos = s_V[sj][si] >= 0;

		__syncthreads();

		for (int k = 0; k < iter; k++) {

			if (threadIdx.x == 0) {

				s_h[sj][si - 1] = h[globalIdx - 1];				
				s_U[sj][si - 1] = U[globalIdx - 1];
				s_V[sj][si - 1] = V[globalIdx - 1];
				s_h[sj][si + BLOCK_SIZE_x] = h[globalIdx + BLOCK_SIZE_x];
				s_U[sj][si + BLOCK_SIZE_x] = U[globalIdx + BLOCK_SIZE_x];
				s_V[sj][si + BLOCK_SIZE_x] = V[globalIdx + BLOCK_SIZE_x];
			}
			if (threadIdx.y == 0) {
				s_h[sj - 1][si] = h[globalIdx - n_d];				
				s_U[sj - 1][si] = U[globalIdx - n_d];
				s_V[sj - 1][si] = V[globalIdx - n_d];
				s_h[sj + BLOCK_SIZE_y][si] = h[globalIdx + n_d * (BLOCK_SIZE_y)];				
				s_U[sj + BLOCK_SIZE_y][si] = U[globalIdx + n_d * (BLOCK_SIZE_y)];
				s_V[sj + BLOCK_SIZE_y][si] = V[globalIdx + n_d * (BLOCK_SIZE_y)];
			}
			

				//update U (no sync necessary)
			utemp = s_U[sj][si] - g * dt / dx * (s_h[sj][si + 1] - s_h[sj][si])
				- s_upos * dt / dx * (s_U[sj][si] - s_U[sj][si - 1])*(s_U[sj][si] + s_U[sj][si - 1]) / 2
				- s_vpos * dt / dy * (s_U[sj][si] - s_U[sj - 1][si])*(s_V[sj - 1][si] + s_V[sj - 1][si + 1]) / 2
				- (1 - s_upos) * dt / dx * (s_U[sj][si + 1] - s_U[sj][si])*(s_U[sj][si] + s_U[sj][si + 1]) / 2
				- (1 - s_vpos) * dt / dy * (s_U[sj + 1][si] - s_U[sj][si])*(s_V[sj][si] + s_V[sj][si + 1]) / 2;

			__syncthreads();

			//write temp values to shared memory after sync and update upos
			s_U[sj][si] = utemp;
			s_upos = (utemp > 0);

			__syncthreads();
			//now that 
			s_hx =
				(s_upos * s_h[sj][si]
					+ (1 - s_upos) *s_h[sj][si + 1]);

			s_hxmin =
				((s_U[sj][si - 1] >= 0) * s_h[sj][si - 1]
					+ (s_U[sj][si - 1] < 0) *s_h[sj][si]);

			//write back to global memory
		//	U[globalIdx] = utemp;

			//update V
			vtemp = s_V[sj][si] - g * dt / dy * (s_h[sj + 1][si] - s_h[sj][si])
				- s_vpos * dt / dy * (s_V[sj][si] - s_V[sj - 1][si])*(s_V[sj][si] + s_V[sj - 1][si]) / 2
				- s_upos * dt / dx * (s_V[sj][si] - s_V[sj][si - 1])  *(s_U[sj + 1][si - 1] + s_U[sj][si - 1]) / 2
				- (1 - s_vpos) * dt / dy * (s_V[sj + 1][si] - s_V[sj][si])*(s_V[sj][si] + s_V[sj + 1][si]) / 2
				- (1 - s_upos) * dt / dx * (s_V[sj][si + 1] - s_V[sj][si])  *(s_U[sj + 1][si] + s_U[sj][si]) / 2;

			

			__syncthreads();

			s_vpos = (vtemp >= 0);
			s_V[sj][si] = vtemp;
			

			__syncthreads();

			//V[globalIdx] = vtemp;

			//calculate hy
			s_hy =
				s_vpos * s_h[sj][si]
				+ (1 - s_vpos) *s_h[sj + 1][si];

			s_hymin =
				(s_V[sj - 1][si] >= 0) * s_h[sj - 1][si]
				+ (s_V[sj - 1][si] < 0) *s_h[sj][si];

			__syncthreads();
			// update h
			s_h[sj][si] = s_h[sj][si] - dt / dx * (s_hx * s_U[sj][si] - s_hxmin * s_U[sj][si - 1])
				- dt / dy * (s_hy * s_V[sj][si] - s_hymin * s_V[sj - 1][si]);

			/*s_h[sj][si] = s_h[sj][si] - dt / dx * s_hx[sj][si] * s_U[sj][si] - s_hx[sj][si - 1] * s_U[sj][si - 1]
				- dt / dy * s_hy[sj][si] * s_V[sj][si] - s_hy[sj - 1][si] * s_V[sj - 1][si];*/

			__syncthreads();
			
			if (threadIdx.x == 0) {

				h[globalIdx] = s_h[sj][si];				
				U[globalIdx] =  utemp;
				V[globalIdx] = vtemp;

				h[globalIdx + BLOCK_SIZE_x-1] = s_h[sj][si + BLOCK_SIZE_x-1];
				U[globalIdx + BLOCK_SIZE_x - 1] =  s_U[sj][si + BLOCK_SIZE_x - 1];
				V[globalIdx + BLOCK_SIZE_x-1] = s_V[sj][si + BLOCK_SIZE_x-1];
			}
			if (threadIdx.y == 0) {
				h[globalIdx ] = s_h[sj ][si];
				U[globalIdx] =  s_U[sj][si];
				V[globalIdx ] = s_V[sj ][si];

				h[globalIdx + (n_d) * (BLOCK_SIZE_y-1)] = s_h[sj + BLOCK_SIZE_y-1][si];
				U[globalIdx + (n_d) * (BLOCK_SIZE_y - 1)] =  s_U[sj + BLOCK_SIZE_y - 1][si];
				V[globalIdx + (n_d) * (BLOCK_SIZE_y-1)] = s_V[sj + BLOCK_SIZE_y-1][si];
			}
			__syncthreads();
			
		}
		V[globalIdx] = vtemp;
		//U[globalIdx] = utemp;
		h[globalIdx] = s_h[sj][si];
	}

	/* void syncthreads(int threadcount)
	{
		std::condition_variable wake;
		
		std::unique_lock<std::mutex> lck(mu);

		barrier++;		

		
			

		if (barrier == threadcount)
		{
			barrier = 0;			
			wake.notify_all();
			
			
			
		}
		else
			while(barrier<threadcount)
		wake.wait(lck);
		
		assert(barrier != threadcount);

		if (barrier == --threadcount)
		{
			wake.notify_all();
		}
		else
		{
			wake.wait(lck, [this]() { return barrier == threadcount; });
		}



	} 
	*/

	void updatecputhread(float* h, float* U, float* V, float dt, int tid, int numthreads,int iter, cbar::cyclicbarrier* cb)
	{
		int BLOCK_SIZE_x = n-2;
		int BLOCK_SIZE_y = ceil((float)(n-2) / (float)numthreads);
		
		float g = g_h;
		float dx = dx_h;
		float dy = dy_h;

		float** s_h = NULL; float** s_U = NULL; float** s_V = NULL; __int8** s_upos = NULL; __int8** s_vpos = NULL;
		s_h = new float* [(BLOCK_SIZE_y + 2)]; s_U = new float* [(BLOCK_SIZE_y + 2)]; s_V = new float* [(BLOCK_SIZE_y + 2)];
		s_upos = new __int8* [(BLOCK_SIZE_y + 2)]; s_vpos = new __int8* [(BLOCK_SIZE_y + 2)];

				// Create a row for every pointer 
				for (int k = 0; k <= BLOCK_SIZE_y+2; k++)
		{ 
			   s_h[k] = new float[BLOCK_SIZE_x+2];
			   s_U[k] = new float[BLOCK_SIZE_x+2];
			   s_V[k] = new float[BLOCK_SIZE_x+2];
			s_upos[k] = new __int8[BLOCK_SIZE_x + 2];
			s_vpos[k] = new __int8[BLOCK_SIZE_x + 2];
		}

				for (int z = 0; z < iter; z++) {

					for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
						for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
							int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;

							//Boundaries
							if (si == 1) {

								s_h[sj][si - 1] = h[globalIdx - 1];
								s_U[sj][si - 1] = U[globalIdx - 1];
								s_V[sj][si - 1] = V[globalIdx - 1];

								s_h[sj][si + BLOCK_SIZE_x] = h[globalIdx + BLOCK_SIZE_x];
								s_U[sj][si + BLOCK_SIZE_x] = U[globalIdx + BLOCK_SIZE_x];
								s_V[sj][si + BLOCK_SIZE_x] = V[globalIdx + BLOCK_SIZE_x];
							}
							if (sj == 1) {
								s_h[sj - 1][si] = h[globalIdx - n];
								s_U[sj - 1][si] = U[globalIdx - n];
								s_V[sj - 1][si] = V[globalIdx - n];

								s_h[sj + BLOCK_SIZE_y][si] = h[globalIdx + n * (BLOCK_SIZE_y)];
								s_U[sj + BLOCK_SIZE_y][si] = U[globalIdx + n * (BLOCK_SIZE_y)];
								s_V[sj + BLOCK_SIZE_y][si] = V[globalIdx + n * (BLOCK_SIZE_y)];
							}

							// copy global variables into shared memory
							s_h[sj][si] = h[globalIdx];
							s_U[sj][si] = U[globalIdx];
							s_V[sj][si] = V[globalIdx];
							s_upos[sj][si] = s_U[sj][si] >= 0;
							s_vpos[sj][si] = s_V[sj][si] >= 0;
						}
					}

					cb->await();//

						//update U
					for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
						for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
							int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;

							//update U (no sync necessary)
							U[globalIdx] = s_U[sj][si] - g * dt / dx * (s_h[sj][si + 1] - s_h[sj][si])
								- s_upos[sj][si] * dt / dx * (s_U[sj][si] - s_U[sj][si - 1]) * (s_U[sj][si] + s_U[sj][si - 1]) / 2
								- s_vpos[sj][si] * dt / dy * (s_U[sj][si] - s_U[sj - 1][si]) * (s_V[sj - 1][si] + s_V[sj - 1][si + 1]) / 2
								- (1 - s_upos[sj][si]) * dt / dx * (s_U[sj][si + 1] - s_U[sj][si]) * (s_U[sj][si] + s_U[sj][si + 1]) / 2
								- (1 - s_vpos[sj][si]) * dt / dy * (s_U[sj + 1][si] - s_U[sj][si]) * (s_V[sj][si] + s_V[sj][si + 1]) / 2;

							s_upos[sj][si] = (U[globalIdx] >= 0);
						}
					}
					cb->await();//syncthreads(numthreads);

					//write temp values to shared memory after sync and update upos
					for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
						for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
							int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
							s_U[sj][si] = U[globalIdx];
							
						}
					}
					cb->await();//syncthreads(numthreads);

					//update V
					for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
						for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
							int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
							V[globalIdx] = s_V[sj][si] - g * dt / dy * (s_h[sj + 1][si] - s_h[sj][si])
								- s_vpos[sj][si] * dt / dy * (s_V[sj][si] - s_V[sj - 1][si]) * (s_V[sj][si] + s_V[sj - 1][si]) / 2
								- s_upos[sj][si] * dt / dx * (s_V[sj][si] - s_V[sj][si - 1]) * (s_U[sj + 1][si - 1] + s_U[sj][si - 1]) / 2
								- (1 - s_vpos[sj][si]) * dt / dy * (s_V[sj + 1][si] - s_V[sj][si]) * (s_V[sj][si] + s_V[sj + 1][si]) / 2
								- (1 - s_upos[sj][si]) * dt / dx * (s_V[sj][si + 1] - s_V[sj][si]) * (s_U[sj + 1][si] + s_U[sj][si]) / 2;

							s_vpos[sj][si] = (V[globalIdx] >= 0);
						}
					}
					cb->await();//syncthreads(numthreads);


					for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
						for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
							int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
							s_V[sj][si] = V[globalIdx];
							
						}
					}
					cb->await();//syncthreads(numthreads);
					//update H
					for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
						for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
							int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
							//calculate hy
							float s_hy =
								s_vpos[sj][si] * s_h[sj][si]
								+ (1 - s_vpos[sj][si]) * s_h[sj + 1][si];

							float s_hymin =
								(s_V[sj - 1][si] >= 0) * s_h[sj - 1][si]
								+ (s_V[sj - 1][si] < 0) * s_h[sj][si];

							float s_hx =
								(s_upos[sj][si] * s_h[sj][si]
									+ (1 - s_upos[sj][si]) * s_h[sj][si + 1]);

							float s_hxmin =
								((s_U[sj][si - 1] >= 0) * s_h[sj][si - 1]
									+ (s_U[sj][si - 1] < 0) * s_h[sj][si]);


							// update h
							h[globalIdx] = s_h[sj][si] - dt / dx * (s_hx * s_U[sj][si] - s_hxmin * s_U[sj][si - 1])
								- dt / dy * (s_hy * s_V[sj][si] - s_hymin * s_V[sj - 1][si]);




						}

					}
					cb->await();//syncthreads(numthreads);
				}
	}

	void updatecputhreadborder(float* h, float* U, float* V, float dt, int tid, int numthreads, int iter, cbar::cyclicbarrier* cb)
	{  
		const int BLOCK_SIZE_x = n - 2;
		const int BLOCK_SIZE_y = ceil((float)(n - 2) / (float)numthreads);

		const float g = g_h;
		const float dx = dx_h;
		const float dy = dy_h;

		float** s_h = NULL; float** s_U = NULL; float** s_V = NULL; __int8** s_upos = NULL; __int8** s_vpos = NULL;
		s_h = new float*[(BLOCK_SIZE_y + 2)]; s_U = new float*[(BLOCK_SIZE_y + 2)]; s_V = new float*[(BLOCK_SIZE_y + 2)];
		s_upos = new __int8*[(BLOCK_SIZE_y + 2)]; s_vpos = new __int8*[(BLOCK_SIZE_y + 2)];

		// Create a row for every pointer 
		for (int k = 0; k <= BLOCK_SIZE_y + 2; k++)
		{
			s_h[k] = new float[BLOCK_SIZE_x + 2];
			s_U[k] = new float[BLOCK_SIZE_x + 2];
			s_V[k] = new float[BLOCK_SIZE_x + 2];
			s_upos[k] = new __int8[BLOCK_SIZE_x + 2];
			s_vpos[k] = new __int8[BLOCK_SIZE_x + 2];
		}

        
		
			for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
				for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
				int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
				// copy global variables into shared memory
				s_h[sj][si] = h[globalIdx];
				s_U[sj][si] = U[globalIdx];
				s_V[sj][si] = V[globalIdx];
				s_upos[sj][si] = s_U[sj][si] >= 0;
				s_vpos[sj][si] = s_V[sj][si] >= 0;
			}
		}
		for (int z = 0; z < iter; z++) {
			
			
				for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
					for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
					int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;

					//Boundaries
					if (si == 1) {

						s_h[sj][si - 1] = h[globalIdx - 1];
						s_U[sj][si - 1] = U[globalIdx - 1];
						s_V[sj][si - 1] = V[globalIdx - 1];

						s_h[sj][si + BLOCK_SIZE_x] = h[globalIdx + BLOCK_SIZE_x];
						s_U[sj][si + BLOCK_SIZE_x] = U[globalIdx + BLOCK_SIZE_x];
						s_V[sj][si + BLOCK_SIZE_x] = V[globalIdx + BLOCK_SIZE_x];
					}
					if (sj == 1) {
						s_h[sj - 1][si] = h[globalIdx - n];
						s_U[sj - 1][si] = U[globalIdx - n];
						s_V[sj - 1][si] = V[globalIdx - n];

						s_h[sj + BLOCK_SIZE_y][si] = h[globalIdx + n * (BLOCK_SIZE_y)];
						s_U[sj + BLOCK_SIZE_y][si] = U[globalIdx + n * (BLOCK_SIZE_y)];
						s_V[sj + BLOCK_SIZE_y][si] = V[globalIdx + n * (BLOCK_SIZE_y)];
					}

					
				}
			}

			cb->await();//

				//update U

			for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
			for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
				
					int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;

					//update U (no sync necessary)
					U[globalIdx] = s_U[sj][si] - g * dt / dx * (s_h[sj][si + 1] - s_h[sj][si])
						- s_upos[sj][si] * dt / dx * (s_U[sj][si] - s_U[sj][si - 1]) * (s_U[sj][si] + s_U[sj][si - 1]) / 2
						- s_vpos[sj][si] * dt / dy * (s_U[sj][si] - s_U[sj - 1][si]) * (s_V[sj - 1][si] + s_V[sj - 1][si + 1]) / 2
						- (1 - s_upos[sj][si]) * dt / dx * (s_U[sj][si + 1] - s_U[sj][si]) * (s_U[sj][si] + s_U[sj][si + 1]) / 2
						- (1 - s_vpos[sj][si]) * dt / dy * (s_U[sj + 1][si] - s_U[sj][si]) * (s_V[sj][si] + s_V[sj][si + 1]) / 2;

					s_upos[sj][si] = (U[globalIdx] >= 0);
				}
			}
			cb->await();//syncthreads(numthreads);

			//write temp values to shared memory after sync and update upos
			for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
			for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
				
					int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
					s_U[sj][si] = U[globalIdx];

				}
			}
			cb->await();//syncthreads(numthreads);

			//update V
			
				for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
					for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
					int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
					V[globalIdx] = s_V[sj][si] - g * dt / dy * (s_h[sj + 1][si] - s_h[sj][si])
						- s_vpos[sj][si] * dt / dy * (s_V[sj][si] - s_V[sj - 1][si]) * (s_V[sj][si] + s_V[sj - 1][si]) / 2
						- s_upos[sj][si] * dt / dx * (s_V[sj][si] - s_V[sj][si - 1]) * (s_U[sj + 1][si - 1] + s_U[sj][si - 1]) / 2
						- (1 - s_vpos[sj][si]) * dt / dy * (s_V[sj + 1][si] - s_V[sj][si]) * (s_V[sj][si] + s_V[sj + 1][si]) / 2
						- (1 - s_upos[sj][si]) * dt / dx * (s_V[sj][si + 1] - s_V[sj][si]) * (s_U[sj + 1][si] + s_U[sj][si]) / 2;

					s_vpos[sj][si] = (V[globalIdx] >= 0);
				}
			}
			cb->await();//syncthreads(numthreads);


			for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
			for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
				
					int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
					s_V[sj][si] = V[globalIdx];

				}
			}
			cb->await();//syncthreads(numthreads);
			//update H
			for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
			for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
				
					int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
					//calculate hy
					float s_hy =
						s_vpos[sj][si] * s_h[sj][si]
						+ (1 - s_vpos[sj][si]) * s_h[sj + 1][si];

					float s_hymin =
						(s_V[sj - 1][si] >= 0) * s_h[sj - 1][si]
						+ (s_V[sj - 1][si] < 0) * s_h[sj][si];

					float s_hx =
						(s_upos[sj][si] * s_h[sj][si]
							+ (1 - s_upos[sj][si]) * s_h[sj][si + 1]);

					float s_hxmin =
						((s_U[sj][si - 1] >= 0) * s_h[sj][si - 1]
							+ (s_U[sj][si - 1] < 0) * s_h[sj][si]);


					// update h
					h[globalIdx] = s_h[sj][si] - dt / dx * (s_hx * s_U[sj][si] - s_hxmin * s_U[sj][si - 1])
						- dt / dy * (s_hy * s_V[sj][si] - s_hymin * s_V[sj - 1][si]);
					



				}

			}
			cb->await();//syncthreads(numthreads);
			
				for (int sj = 1; sj < BLOCK_SIZE_y + 1; sj++) {
					for (int si = 1; si < BLOCK_SIZE_x + 1; si++) {
					int globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
					s_h[sj][si] = h[globalIdx];

				}
			}
			cb->await();
		}
	}
	
	__int8 *initializeBoolArray() {
		__int8 *ptr = 0;
		//printf("Initializing bool array \n");
		checkCuda(cudaMalloc(&ptr, n * n * sizeof(__int8)));
		//checkCudaError("Malloc for matrix on device failed !");

		return ptr;

	}

	float *initializeFloatArray(){
 //   void initializearrays(float *H, float  *U, float  *V, /*float  *Hx, float  *Hy,*/ __int8 *Upos, __int8 *Vpos) {
		float *ptr = 0;
		//printf("Initializing float array \n");
	checkCuda(cudaMalloc(&ptr, n * n * sizeof(float)));
	//checkCudaError("Malloc for matrix on device failed !");

	return ptr;
	

	}

	__global__	void fillarrays(float *H, /*float  *Hx, float  *Hy,*/ __int8 *Upos, __int8 *Vpos) {

		//int i = threadIdx.x;
		//int j = blockIdx.x*blockDim.y + threadIdx.y;
		int j = blockIdx.y*blockDim.y + threadIdx.y;
		int i = blockIdx.x*blockDim.x + threadIdx.x;		
		int globalIdx = (j+1) * n_d + i+1;
		//printf("globalidx = %d \n", globalIdx);
		       H[globalIdx] = Hstart;
			  // U[globalIdx] = 0;
			   //V[globalIdx] = 0;
			Upos[globalIdx] = 1;
			Vpos[globalIdx] = 1;
			if (i == 0) {
			    H[globalIdx-1] = Hstart;			
		     Upos[globalIdx-1] = 1;
			 Vpos[globalIdx-1] = 1;
			    H[globalIdx +n_d-2] = Hstart;
			 Upos[globalIdx +n_d-2] = 1;
			 Vpos[globalIdx +n_d-2] = 1;
			}
			if (j == 0) {
				   H[globalIdx - n_d] = Hstart;
				Upos[globalIdx - n_d] = 1;
				Vpos[globalIdx - n_d] = 1;
				   H[globalIdx + n_d*(n_d-2) ] = Hstart;
				Upos[globalIdx + n_d*(n_d-2) ] = 1;
				Vpos[globalIdx + n_d*(n_d-2) ] = 1;
			}
			__syncthreads();

	}

	__global__	void printIdx() {

		//int i = threadIdx.x;
		//int j = blockIdx.x*blockDim.y + threadIdx.y;
		int j = blockIdx.y*blockDim.y + threadIdx.y;
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		int globalIdx = j * n_d + i;
		//printf("globalidx = %d \n", globalIdx);
		printf("i = %d \n j= %d \n globalidx = %d \n", i,j,globalIdx);

		

	}

	__global__ void Waterdrop(float *H,float height, int width, float step) {

		
		int j = blockIdx.y*blockDim.y + threadIdx.y;
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		float x = -1 + i * step;
		float y = -1 + j * step;
		
		
		float D = (1+4*droppar)/5* height *expf(-5*(x*x + y * y));
		int globalIdx = (j + 1 + droppar * (n_d - width)) * n_d + i + 1 + droppar * (n_d - width);
		 
		H[globalIdx] = H[globalIdx] + D;
		//printf("globalIdx: %d x: %.3f y: %.3f ", globalIdx, x, y);
		//printf("Idx: %d x: %d y: %d D: %d ", globalIdx, x , y ,  D);
		//printf("x: %.2f y: %.2f D: %.2f \n ",  x , y ,  D);
		//printf("i: %d j: %d \n" ,i ,j);
		//printf("D: %.3f \n" ,D);
		__syncthreads();
		//float height = 1.5*Hstart;
		//int width = (int)(n-2)/2;
		
		//for (float i = -1; i < 1; i = i + 2 / (width - 1));
		/*[x, y] = ndgrid(-1:(2 / (width - 1)) : 1);
		D = height * exp(-5 * (x. ^ 2 + y. ^ 2));
		w = size(D, 1);
		i = ceil(rand*(n - w)) + (1:w);
		j = ceil(rand*(n - w)) + (1:w);
		H(i, j) = H(i, j) + (1 + 4 * rand) / 5 * D;*/
   }

void copyConstants(){

    checkCuda(cudaMemcpyToSymbol(n_d, &n, sizeof(int), 0, cudaMemcpyHostToDevice)); //grid size
	checkCuda(cudaMemcpyToSymbol(g, &g_h, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(dx, &dx_h, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(dy, &dy_h, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(cf, &cf_h, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(Hstart, &Hstart_h, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(droppar, &droppar_h, sizeof(float), 0, cudaMemcpyHostToDevice));
}

void CudaCheckError() {
	// make the host block until the device is finished with foo
	cudaDeviceSynchronize();

	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	}

void initializeWaterdrop(float *H) {

	int width = (int)floor(((float)n - 2) / 3);
	dim3 gridSizeDrop(ceil((float)width / 32), ceil((float)width / 32));
	dim3 blockSizeDrop(fmin(32,width),fmin(32,width));
	float height = 1.5*Hstart_h;
	float step = 2 / (float(width)-1);
	
	Waterdrop << <gridSizeDrop, blockSizeDrop >> > (H, height, width, step);
	
	CudaCheckError();
	
}

void showMatrix(const char *name, float *a, int n, int m)
{

	long x, y;

	for (y = 0; y < m; y++)
	{
		for (x = 0; x < n; x++){
			//printf("%s[%02ld][%02ld]=%6.2f  ", name, y, x, a[y*n + x]);
			if (a[y*n + x] == 0 || a[y*n + x] == 1)
			{
				printf("%d  ", (int)a[y*n + x]);
			}
			else {
				printf("%6.2f  ", a[y*n + x]);
			}
			}
		printf("\n");
	}
	printf("\n");

}

void copyfromCudaMatrix(float *h_a, float *d_a, int n, int m)
{
	//printf("Copying result back... ");
	checkCuda(cudaMemcpy(h_a, d_a, n * m * sizeof(float), cudaMemcpyDeviceToHost));
	//printf("success! \n");
	//checkCudaError("Matrix copy from device failed !");
}

void runprogram(int iter)
{

	
	// Print device and precision
	cudaDeviceProp prop;
	checkCuda(cudaGetDeviceProperties(&prop, 0));
	printf("\nDevice Name: %s\n", prop.name);
	printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);	
	printf("Shared Memory per Block: %d bytes \n", prop.sharedMemPerBlock);
	printf("Shared Memory per SM: %d bytes \n", prop.sharedMemPerMultiprocessor);
	printf("SM count: %d \n", prop.multiProcessorCount);
	printf("Max threads per SM: %d \n", prop.maxThreadsPerMultiProcessor);
	printf("Max threads per block: %d \n", prop.maxThreadsPerBlock);
	printf("block size: %d \n", BLOCK_SIZE_x*BLOCK_SIZE_y);
	printf("Max registers per thread: %d \n", prop.regsPerMultiprocessor / (prop.maxThreadsPerMultiProcessor / (BLOCK_SIZE_x*BLOCK_SIZE_y)) / (BLOCK_SIZE_x*BLOCK_SIZE_y));
	dim3 gridSize((n-2) / (BLOCK_SIZE_x), (n-2) / BLOCK_SIZE_y);
	dim3 blockSize(BLOCK_SIZE_x, BLOCK_SIZE_y);
	int blockmem = ((BLOCK_SIZE_y + 2)*(BLOCK_SIZE_x + 2)*(3 * sizeof(float) + 2 * sizeof(__int8)));
	printf("Block memory: %d bytes \n", blockmem);
	//check if blocks fit in shared memory:
	if (blockmem > prop.sharedMemPerBlock) {
		throw "Block size too large!! \n";
	}
	
	if ((prop.maxThreadsPerMultiProcessor / (BLOCK_SIZE_x*BLOCK_SIZE_y))*blockmem < 16* pow(2, 10)  )
	{
		checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		printf("Configured for 48kb of L1 cache \n");
		
	}
	printf("n = %d \n", n);
	if (n * n * (3 * sizeof(float) + 2 * sizeof(__int8)) > prop.totalGlobalMem)
	{
		throw "Device out of memory!! max size = %d " ,  sqrt( prop.totalGlobalMem/ (3 * sizeof(float) + 2 * sizeof(__int8) ) );
	}
	float *H = initializeFloatArray();
	float *U = initializeFloatArray();
	float *V = initializeFloatArray();
	__int8 *Upos = initializeBoolArray();
	__int8 *Vpos = initializeBoolArray();
	
	copyConstants();

	printf("filling arrays... ");
		fillarrays << <gridSize, blockSize >> > (H,  Upos, Vpos);
		CudaCheckError();
		printf("success! \n");

		float *H_h = 0;
		cudaMallocHost(&H_h, n * n * sizeof(float));

		printf("initializing water drop... ");
		initializeWaterdrop(H);
		printf("success! \n");

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		for (int i = 0; i < iter; i++) {
			update << <gridSize, blockSize >> > (H, Upos, Vpos, U, V, dt);
			//cudaThreadSynchronize();
		}

		//cudaDeviceSynchronize();
		CudaCheckError();

		
	
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
	
		printf("update time: %.7f \n" , milliseconds/1000);
		copyfromCudaMatrix(H_h, H, n, n);
		if (n == 34)
		showMatrix("H", H_h, n, n);

		checkCuda(cudaFree(H));
		checkCuda(cudaFree(U));
		checkCuda(cudaFree(V));
		checkCuda(cudaFree(Upos));
		checkCuda(cudaFree(Vpos));
		cudaDeviceReset();
}

void runprogrambenchmark( int iter, int updatetype)
{

	cudaEvent_t start2;
	cudaEventCreate(&start2);
	cudaEventRecord(start2);

	// Print device and precision
	cudaDeviceProp prop;
	checkCuda(cudaGetDeviceProperties(&prop, 0));

	dim3 gridSize((n - 2) / (BLOCK_SIZE_x), (n - 2) / BLOCK_SIZE_y);
	dim3 blockSize(BLOCK_SIZE_x, BLOCK_SIZE_y);
	int blockmem = ((BLOCK_SIZE_y + 2)*(BLOCK_SIZE_x + 2)*(3 * sizeof(float) + 2 * sizeof(__int8)));
	//check if blocks fit in shared memory:
	if (blockmem > prop.sharedMemPerBlock) {
		throw "Block size too large!! \n";
	}

	if ((prop.maxThreadsPerMultiProcessor / (BLOCK_SIZE_x*BLOCK_SIZE_y))*blockmem < 16 * pow(2, 10))
	{
		checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));


	}

	if (n * n * (3 * sizeof(float) + 2 * sizeof(__int8)) > prop.totalGlobalMem)
	{
		throw "Device out of memory!! max size = %d ", sqrt(prop.totalGlobalMem / (3 * sizeof(float) + 2 * sizeof(__int8)));
	}
	float *H = initializeFloatArray();
	float *U = initializeFloatArray();
	float *V = initializeFloatArray();
	__int8 *Upos = initializeBoolArray();
	__int8 *Vpos = initializeBoolArray();

	copyConstants();


	fillarrays << <gridSize, blockSize >> > (H, Upos, Vpos);
	CudaCheckError();


	float *H_h = 0;
	cudaMallocHost(&H_h, n * n * sizeof(float));
	initializeWaterdrop(H);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	if (updatetype == 1) {
		for (int i = 0; i < iter; i++) {
			update << <gridSize, blockSize >> > (H, Upos, Vpos, U, V, dt);

		}
	}
	else if (updatetype == 2){
		for (int i = 0; i < iter; i++) {
			updatenobool << <gridSize, blockSize >> > (H, U, V, dt);

		}
		}
	else if (updatetype ==3)
		updateborders << <gridSize, blockSize >> > (H, U, V, dt, iter);
	
	/*switch (updatetype) {
	case 1:
		cudaEventRecord(start);
		
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
	case 2:
		cudaEventRecord(start);
		for (int i = 0; i < iter; i++) {
			updatenobool << <gridSize, blockSize >> > (H, U, V, dt);
			cudaDeviceSynchronize();
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
	case 3:
		cudaEventRecord(start);
		updateborders << <gridSize, blockSize >> > (H, U, V, dt, iter);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
	}*/
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	CudaCheckError();
	if (n == 34) {
		copyfromCudaMatrix(H_h, H, n, n);
		showMatrix("H", H_h, n, n);
	}


	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("%.9f ", milliseconds/1000 );

	
	
	float milliseconds2 = 0;
	cudaEventElapsedTime(&milliseconds2, start2, stop);
	printf("%.9f ", milliseconds2 / 1000);

	cudaEventDestroy(start);
	cudaEventDestroy(start2);
	cudaEventDestroy(stop);
	checkCuda(cudaFree(H));
	checkCuda(cudaFree(U));
	checkCuda(cudaFree(V));
	checkCuda(cudaFree(Upos));
	checkCuda(cudaFree(Vpos));
	cudaDeviceReset();
}

void RunProgramCPU(int iter,int maxthreads) {

	//std::mutex mu;
	//int barrier = 0;
	// Print device and precision
	cudaDeviceProp prop;
	checkCuda(cudaGetDeviceProperties(&prop, 0));

	dim3 gridSize((n - 2) / (BLOCK_SIZE_x), (n - 2) / BLOCK_SIZE_y);
	dim3 blockSize(BLOCK_SIZE_x, BLOCK_SIZE_y);
	int blockmem = ((BLOCK_SIZE_y + 2) * (BLOCK_SIZE_x + 2) * (3 * sizeof(float) + 2 * sizeof(__int8)));
	//check if blocks fit in shared memory:
	if (blockmem > prop.sharedMemPerBlock) {
		throw "Block size too large!! \n";
	}

	if ((prop.maxThreadsPerMultiProcessor / (BLOCK_SIZE_x * BLOCK_SIZE_y)) * blockmem < 16 * pow(2, 10))
	{
		checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));


	}

	if (n * n * (3 * sizeof(float) + 2 * sizeof(__int8)) > prop.totalGlobalMem)
	{
		throw "Device out of memory!! max size = %d ", sqrt(prop.totalGlobalMem / (3 * sizeof(float) + 2 * sizeof(__int8)));
	}

	float* H = initializeFloatArray();
	float* U = initializeFloatArray();
	float* V = initializeFloatArray();
	__int8* Upos = initializeBoolArray();
	__int8* Vpos = initializeBoolArray();

	copyConstants();


	fillarrays << <gridSize, blockSize >> > (H, Upos, Vpos);
	CudaCheckError();
	initializeWaterdrop(H);

	float* H_h = 0;
	float* V_h = 0;
	float* U_h = 0;
	cudaMallocHost(&H_h, n * n * sizeof(float));
	cudaMallocHost(&U_h, n * n * sizeof(float));
	cudaMallocHost(&V_h, n * n * sizeof(float));
	copyfromCudaMatrix(H_h, H, n, n);
	copyfromCudaMatrix(U_h, U, n, n);
	copyfromCudaMatrix(V_h, V, n, n);
	//showMatrix("H", H_h, n, n);
	unsigned numthreads;
	if (maxthreads>0)
	{
		numthreads = maxthreads;
	}
	else
	numthreads = std::thread::hardware_concurrency();
	if ((n-2) % numthreads != 0)
		while ((n-2) % numthreads != 0)
			--numthreads;

	std::thread* t = NULL;
	t = new std::thread[numthreads];
	auto cb = new cbar::cyclicbarrier(numthreads); //syncthreads(numthreads);	
	auto timestart = clock();
	for (int i = 0; i < numthreads; i++) {
	
		t[i] = std::thread(updatecputhreadborder, H_h,U_h,V_h,dt,i,numthreads,iter,cb);
	}

	for (int i = 0; i < numthreads; i++) {
		t[i].join();
	}
	float timestop = float(clock() - timestart)/ CLOCKS_PER_SEC;

	if (n == 34) {
		showMatrix("H", H_h, n, n);
	}
	printf("%.9f ",timestop);
	cudaDeviceReset();
	

}

int main()
{	
	
	constexpr int ns[] = { 1  , 2, 4, 8, 16, 32};
	constexpr int iter = 10;
	constexpr int threads[] = { 0,1 };
		//int maxthreads = 1;
		for (const int ni : ns) {
			 n = 3 * 32 * ni + 2;
			
			//n = 34;
			
			
			printf("n = %d GPU: ", n);
			
			//runprogram(iter);
			runprogrambenchmark(iter,1);
			
			//printf("nobool: ");
			runprogrambenchmark(iter,2);
			
			//printf("borders: ");
			//runprogrambenchmark(iter,3);

			//RunProgramCPU(iter,0);
			//if (ni<20)
			printf("CPU: ");
			
			for (const int maxthreads:threads)
			RunProgramCPU(iter, maxthreads);
			
			printf(" \n");
		}
		cudaDeviceReset();
	return 0;
}  