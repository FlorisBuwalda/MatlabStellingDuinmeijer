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
using namespace std::chrono;
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <windows.h>
#define CLOCK_REALTIME 0
//struct timespec { long tv_sec; long tv_nsec; };    //header par
#include <cuda_runtime.h>


// Utilities and timing functions
//#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
//#include <helper_cuda.h>         // helper functions for CUDA error check

#include <iostream>
#include <stdio.h>
#include <assert.h>
//#include <cyclicbarrier.hpp>

#ifndef __CUDACC__  
#define __CUDACC__
#endif
using namespace std;
#include<iostream>
#include<sstream>
#include<fstream>
#include<iomanip>
#include <windows.h>
#include <conio.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <cusparse.h>
#include <cublas_v2.h>


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

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>


#pragma comment(lib,"cublas.lib")
#pragma comment(lib,"cusparse.lib")

int n;

float dx_h;// = 1;
float dy_h;// = 1;


//constexpr int n = 10 * 32 * 3;

__constant__ int  n_d;
__constant__ float dx;
__constant__ float dy;

__constant__ float Hstart ;    //Rest water depth
 float Hstart_h = 1;

constexpr float g_h = (float)9.8;   // gravitational constant
__constant__ float g;                

constexpr float tstep = 1;             // maximum timestep
 float dt = .1;//(float);                // first step is maximum timestep

__constant__ float cf;
constexpr float cf_h = 0;                    // Bottom friction factor

constexpr float droppar_h = .5;
__constant__ float droppar;

constexpr int ndrops = 1;              // maximum number of water drops
constexpr int dropstep = 5;            // drop interval

bool timer = false;
constexpr float safety = (float).9;

constexpr float droogval_h = .05 ;
__constant__ float droogval;

constexpr float offset = +.03; // D_max = H+offset

constexpr int BLOCK_SIZE_x = 32;  // number of threads per block in x-dir
constexpr int BLOCK_SIZE_y = 32;  // number of threads per block in y-dir
#define MAX_THREADS_PER_BLOCK 1024
#define MIN_BLOCKS_PER_MP 2

// launch parameters
constexpr bool showarray =false;
constexpr bool realtimeplot =false;
constexpr bool tidal_case = false;
constexpr bool benchmark = true;

constexpr int plotstep =3;
constexpr int ns[] = {3};// , 2, 4, 8, 11};// {32, 64, 128};

const int iter = 1000;// 12 * 3600 / (.6 / (2 * sqrt(2 * g_h*(5))));
constexpr int threads[] = {1,2,4,8};
constexpr int benchaverages = 1;
constexpr int cpubenchthresh = 5;
constexpr float alpha = 0.5; //0 = implicit
constexpr int implicitfactor = 2;
constexpr bool cgoutput = false;

int plotdelay = 100;

double** benchmarkresult;




//	__global__ void update( float *h, bool *upos, bool *vpos, float *U, float *V, float dt )
//{
//	
//		
//		__shared__      float s_h[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 4-wide halo
//		//__shared__     float s_hy[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 4-wide halo
//		//__shared__     float s_hx[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 4-wide halo
//		__shared__  __int8 s_upos[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 2-wide halo
//		__shared__  __int8 s_vpos[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 2-wide halo
//		__shared__      float s_U[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 2-wide halo
//		__shared__      float s_V[BLOCK_SIZE_y+2][BLOCK_SIZE_x+2]; // 2-wide halo
//
//		//int i = threadIdx.x;
//		//int j = blockIdx.x*blockDim.y + threadIdx.y;
//		int j = blockIdx.y*blockDim.y + threadIdx.y;
//		int i = blockIdx.x*blockDim.x + threadIdx.x;
//		int si = threadIdx.x + 1; // local i for shared memory access + halo offset
//		int sj = threadIdx.y + 1; // local j for shared memory access
//		float utemp, vtemp;
//		float s_hx, s_hxmin, s_hy, s_hymin;
//		int globalIdx =  (j+1) * n_d + i+1;
//		
//		//Boundaries
//		if (threadIdx.x ==0) {
//			
//			   s_h[sj][si-1] =    h[globalIdx-1];
//			s_upos[sj][si-1] = upos[globalIdx-1];
//			s_vpos[sj][si-1] = vpos[globalIdx-1];
//			   s_U[sj][si-1] =	  U[globalIdx-1];
//			   s_V[sj][si-1] =    V[globalIdx-1];
//
//			         s_h[sj][si + BLOCK_SIZE_x ] =    h[globalIdx + BLOCK_SIZE_x ];
//				  s_upos[sj][si + BLOCK_SIZE_x ] = upos[globalIdx + BLOCK_SIZE_x ];
//				  s_vpos[sj][si + BLOCK_SIZE_x ] = vpos[globalIdx + BLOCK_SIZE_x ];
//				     s_U[sj][si + BLOCK_SIZE_x ] =    U[globalIdx + BLOCK_SIZE_x ];
//				     s_V[sj][si + BLOCK_SIZE_x ] =    V[globalIdx + BLOCK_SIZE_x ];
//		}
//		if (threadIdx.y==0) {
//			s_h[sj-1][si] =    h[globalIdx - n_d];
//		 s_upos[sj-1][si] = upos[globalIdx - n_d];
//		 s_vpos[sj-1][si] = vpos[globalIdx - n_d];
//			s_U[sj-1][si] =    U[globalIdx - n_d];
//			s_V[sj-1][si] =    V[globalIdx - n_d];
//
//			s_h[sj+BLOCK_SIZE_y][si] =    h[globalIdx +n_d*(BLOCK_SIZE_y)];
//		 s_upos[sj+BLOCK_SIZE_y][si] = upos[globalIdx +n_d*(BLOCK_SIZE_y)];
//		 s_vpos[sj+BLOCK_SIZE_y][si] = vpos[globalIdx +n_d*(BLOCK_SIZE_y)];
//			s_U[sj+BLOCK_SIZE_y][si] =    U[globalIdx +n_d*(BLOCK_SIZE_y)];
//			s_V[sj+BLOCK_SIZE_y][si] =    V[globalIdx +n_d*(BLOCK_SIZE_y)];
//		}
//
//		// copy global variables into shared memory
//		   s_h[sj][si] = h[globalIdx];
//		s_upos[sj][si] = upos[globalIdx];
//		s_vpos[sj][si] = vpos[globalIdx]; 
//		   s_U[sj][si] = U[globalIdx];
//		   s_V[sj][si] = V[globalIdx];
//
//		__syncthreads();
//		
//		// fill in periodic images in shared memory array 
//		//if (i < 4) {
//		//	s_f[sj][si - 4] = s_f[sj][si + mx - 5];
//		//	s_f[sj][si + mx] = s_f[sj][si + 1];
//		//}
//
//		//__syncthreads();
//
//		//update Hx and Hy
//	/*	 s_hx[sj][si] =
//			s_upos[sj][si] * s_h[sj][si]
//			+ (1 - s_upos[sj][si]) *s_h[sj][si + 1];
//
//		s_hy[sj][si] =
//			s_vpos[sj][si] * s_h[sj][si]
//			+ (1 - s_vpos[sj][si]) *s_h[sj+1][si];*/
//
//		//update U (no sync necessary)
//		//utemp = s_U[sj][si] - g * dt / dx * (s_h[sj][si + 1] - s_h[sj][si])
//		//	      - s_upos[sj][si] * dt / dx * (s_U[sj][si] - s_U[sj][si - 1])*(s_U[sj][si] + s_U[sj][si - 1]) / 2
//		//	      - s_vpos[sj][si] * dt / dy * (s_U[sj][si] - s_U[sj - 1][si])*(s_V[sj-1][si] + s_V[sj - 1][si+1]) / 2
//		//	- (1 - s_upos[sj][si]) * dt / dx * (s_U[sj][si + 1] - s_U[sj][si])*(s_U[sj][si] + s_U[sj][si + 1]) / 2
//		//	- (1 - s_vpos[sj][si]) * dt / dy * (s_U[sj + 1][si] - s_U[sj][si])*(s_V[sj][si] + s_V[sj ][si+1]) / 2;
//
//		utemp = s_U[sj][si] - g * dt / dx * (s_h[sj][si + 1] - s_h[sj][si]) +
//			(s_upos[sj][si] ? -dt / dx * (s_U[sj][si] - s_U[sj][si - 1]) * (s_U[sj][si] + s_U[sj][si - 1]) / 2 :
//				      -dt / dx * (s_U[sj][si + 1] - s_U[sj][si]) * (s_U[sj][si] + s_U[sj][si + 1]) / 2) +
//				(s_vpos[sj][si] ? -dt / dy * (s_U[sj][si] - s_U[sj - 1][si]) * (s_V[sj - 1][si] + s_V[sj - 1][si + 1]) / 2 :
//					      -dt / dy * (s_U[sj + 1][si] - s_U[sj][si]) * (s_V[sj][si] + s_V[sj][si + 1]) / 2);
//
//		__syncthreads();
//
//		U[globalIdx] = utemp;
//		//write temp values to shared memory after sync and update upos
//		s_U[sj][si] = utemp;
//		s_upos[sj][si] = (__int8)(utemp >= 0);
//		
//
//		__syncthreads();
//		//now that 
//		 s_hx =
//			          (s_upos[sj][si] * s_h[sj][si]
//				+ (1 - s_upos[sj][si]) *s_h[sj][si + 1]);
//
//		 s_hxmin =
//			           (s_upos[sj][si - 1] * s_h[sj][si - 1]
//				+ (1 - s_upos[sj][si - 1]) *s_h[sj][si]);
//
//		//write back to global memory
//		
//
//		//update V
//		/*vtemp = s_V[sj][si] - g * dt / dy * (s_h[sj + 1][si] - s_h[sj][si])
//			- s_vpos[sj][si] * dt / dy * (s_V[sj][si] - s_V[sj - 1][si])*(s_V[sj][si] + s_V[sj - 1][si]) / 2
//			- s_upos[sj][si] * dt / dx * (s_V[sj][si] - s_V[sj][si-1])  *(s_U[sj + 1][si - 1] + s_U[sj][si - 1]) / 2
//			- (1-s_vpos[sj][si]) * dt / dy * (s_V[sj+1][si] - s_V[sj ][si])*(s_V[sj][si] + s_V[sj + 1][si]) / 2
//			- (1-s_upos[sj][si]) * dt / dx * (s_V[sj][si+1] - s_V[sj][si])  *(s_U[sj + 1][si ] + s_U[sj][si ]) / 2;*/
//
//		vtemp = s_V[sj][si] - g * dt / dy * (s_h[sj + 1][si] - s_h[sj][si])
//			+ (s_vpos[sj][si] ? -dt / dy * (s_V[sj][si] - s_V[sj - 1][si]) * (s_V[sj][si] + s_V[sj - 1][si]) / 2 :
//				                -dt / dy * (s_V[sj + 1][si] - s_V[sj][si]) * (s_V[sj][si] + s_V[sj + 1][si]) / 2)
//			+ (s_upos[sj][si] ? -dt / dx * (s_V[sj][si] - s_V[sj][si - 1]) * (s_U[sj + 1][si - 1] + s_U[sj][si - 1]) / 2 :
//				                -dt / dx * (s_V[sj][si + 1] - s_V[sj][si]) * (s_U[sj + 1][si] + s_U[sj][si]) / 2);
//		
//		__syncthreads();
//		V[globalIdx] = vtemp;
//		s_V[sj][si] =  vtemp;
//		s_vpos[sj][si]= (__int8)(vtemp >= 0);
//
//		__syncthreads();
//
//		
//		
//		//calculate hy
//		s_hy =
//			s_vpos[sj][si] * s_h[sj][si]
//			+ (1 - s_vpos[sj][si]) *s_h[sj + 1][si];
//
//		s_hymin =
//			s_vpos[sj - 1][si] * s_h[sj - 1][si]
//			+ (1 - s_vpos[sj - 1][si]) *s_h[sj][si];
//
//
//		// update h
//		s_h[sj][si] = s_h[sj][si] - dt / dx * (s_hx * s_U[sj][si] - s_hxmin * s_U[sj][si - 1])
//								  - dt / dy * (s_hy * s_V[sj][si] - s_hymin * s_V[sj - 1][si]);
//
//			/*s_h[sj][si] = s_h[sj][si] - dt / dx * s_hx[sj][si] * s_U[sj][si] - s_hx[sj][si - 1] * s_U[sj][si - 1]
//				- dt / dy * s_hy[sj][si] * s_V[sj][si] - s_hy[sj - 1][si] * s_V[sj - 1][si];*/
//
//		
//		//write h back to global memory
//		h[globalIdx] = s_h[sj][si];
//		
//		__syncthreads();
//			
//	}

std::mutex m;
std::condition_variable cv;
bool ready = true;
bool processed = false;


void syncthreads(int even, int* pointer, int numthreads, int tid) {
	
	std::unique_lock<std::mutex> lk(m);
	if (even == 1) {
		

		// after the wait, we own the lock.
		//printf("worker %d \n", tid);
		pointer[0]++;

		// Send data back to main()
		if (pointer[0] >= numthreads) {

			//std::cout << "Worker thread signals data processing completed\n";
			pointer[0] = 0;
			processed = true;
			ready = false;
			// Manual unlocking is done before notifying, to avoid waking up
			// the waiting thread only to block again (see notify_one for details)
			lk.unlock();
			cv.notify_all();
		}
		cv.wait(lk, [] {return processed; });
		
	}
	if (even == 2) {
		
		// after the wait, we own the lock.
		//printf("worker %d, sync2 \n", tid);
		pointer[0]++;

		// Send data back to main()
		if (pointer[0] >= numthreads) {	
			//std::cout << "Worker thread signals data processing completed\n";
			pointer[0] = 0;
			processed = false;
			ready = true;
			// Manual unlocking is done before notifying, to avoid waking up
			// the waiting thread only to block again (see notify_one for details)
			lk.unlock();
			cv.notify_all();
		}
		cv.wait(lk, [] {return ready; });
	}
}

__global__ void
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
calcUVexpl(float *h, float *U, float *V, float* Utemp, float* Vtemp, __int8* kfUtemp, __int8* kfVtemp, float *D, float* Hx, float* Hy, float dt, float alpha, bool tidal_case = false, float tideheight = 0)
{



	__shared__       float   s_h[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4]; // 4-wide halo
	__shared__       float   s_D[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4];
	__shared__       float   s_U[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4]; // 4-wide halo
	__shared__       float   s_V[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4]; // 4-wide halo		
	__shared__		__int8 s_kfU[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4];
	__shared__		__int8 s_kfV[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4];



	//int i = threadIdx.x;
	//int j = blockIdx.x*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int si = threadIdx.x + 2; // local i for shared memory access + halo offset
	unsigned int sj = threadIdx.y + 2; // local j for shared memory access
	unsigned int globalIdx = (j + 2) * n_d + i + 2;

	float s_hx, s_hy, s_dx, s_dy;
	
	

	s_U[sj][si] = U[globalIdx];
	s_V[sj][si] = V[globalIdx];
	s_h[sj][si] = h[globalIdx];
	s_D[sj][si] = D[globalIdx];

	//Borders
	if (threadIdx.y == 0 | threadIdx.y == 1) {

		/*if (blockIdx.x == 0 && blockIdx.y ==0) {
			Htemp[globalIdx - 2 * n_d] = 1.5;
			Htemp[globalIdx + n_d * BLOCK_SIZE_y] = 1.5;
		}*/

		s_h[sj - 2][si] = h[globalIdx - 2 * n_d];
		s_U[sj - 2][si] = U[globalIdx - 2 * n_d];
		s_V[sj - 2][si] = V[globalIdx - 2 * n_d];
		s_D[sj - 2][si] = D[globalIdx - 2 * n_d];

		s_h[sj + BLOCK_SIZE_y][si] = h[globalIdx + n_d * (BLOCK_SIZE_y)];
		s_U[sj + BLOCK_SIZE_y][si] = U[globalIdx + n_d * (BLOCK_SIZE_y)];
		s_V[sj + BLOCK_SIZE_y][si] = V[globalIdx + n_d * (BLOCK_SIZE_y)];
		s_D[sj + BLOCK_SIZE_y][si] = D[globalIdx + n_d * (BLOCK_SIZE_y)];

		if (threadIdx.x == 0) {
			for (int offset = 1; offset <= 2; offset++) {
				s_h[sj - 2][si - offset] = h[globalIdx - 2 * n_d - offset];
				s_U[sj - 2][si - offset] = U[globalIdx - 2 * n_d - offset];
				s_V[sj - 2][si - offset] = V[globalIdx - 2 * n_d - offset];

				s_h[sj - 2][BLOCK_SIZE_x + offset + 1] = h[globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1];
				s_U[sj - 2][BLOCK_SIZE_x + offset + 1] = U[globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1];
				s_V[sj - 2][BLOCK_SIZE_x + offset + 1] = V[globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1];

				s_h[BLOCK_SIZE_y + sj][si - offset] = h[globalIdx + (BLOCK_SIZE_y)* n_d - offset];
				s_U[BLOCK_SIZE_y + sj][si - offset] = U[globalIdx + (BLOCK_SIZE_y)* n_d - offset];
				s_V[BLOCK_SIZE_y + sj][si - offset] = V[globalIdx + (BLOCK_SIZE_y)* n_d - offset];

				s_h[BLOCK_SIZE_y + sj][BLOCK_SIZE_x + offset + 1] = h[globalIdx + (BLOCK_SIZE_y)* n_d + BLOCK_SIZE_x + offset - 1];
				s_U[BLOCK_SIZE_y + sj][BLOCK_SIZE_x + offset + 1] = U[globalIdx + (BLOCK_SIZE_y)* n_d + BLOCK_SIZE_x + offset - 1];
				s_V[BLOCK_SIZE_y + sj][BLOCK_SIZE_x + offset + 1] = V[globalIdx + (BLOCK_SIZE_y)* n_d + BLOCK_SIZE_x + offset - 1];






			}
		}
	}

	if ((threadIdx.y == 2 | threadIdx.y == 3) /*& threadIdx.y < BLOCK_SIZE_y*/) {

		unsigned int jtemp = blockIdx.y*blockDim.y + threadIdx.x;
		unsigned int itemp = blockIdx.x*blockDim.x + threadIdx.y;
		unsigned int si2 = threadIdx.y; // local i for shared memory access + halo offset
		unsigned int sj2 = threadIdx.x + 2; // local j for shared memory access
		unsigned int globalIdx2 = (jtemp + 2)* n_d + itemp;

		s_h[sj2][si2 - 2] = h[globalIdx2 - 2];
		s_U[sj2][si2 - 2] = U[globalIdx2 - 2];
		s_V[sj2][si2 - 2] = V[globalIdx2 - 2];
		s_D[sj2][si2 - 2] = D[globalIdx2 - 2];

		s_h[sj2][si2 + BLOCK_SIZE_x] = h[globalIdx2 + BLOCK_SIZE_x];/*BLOCK_SIZE_x*/
		s_U[sj2][si2 + BLOCK_SIZE_x] = U[globalIdx2 + BLOCK_SIZE_x];
		s_V[sj2][si2 + BLOCK_SIZE_x] = V[globalIdx2 + BLOCK_SIZE_x];
		s_D[sj2][si2 + BLOCK_SIZE_x] = D[globalIdx2 + BLOCK_SIZE_x];


	}

	__syncthreads();

	// hxy and kfuv on the borders
	if (threadIdx.y == 0 | threadIdx.y == 1) {


		s_kfU[sj - 2][si] = 1 - (
			     ((s_U[sj - 2][si] > 0) *(s_h[sj - 2][si]      - D[globalIdx - n_d * 2])
				+ (s_U[sj - 2][si] < 0) *(s_h[sj - 2][si + 1]  - D[globalIdx+1 - n_d * 2])
				+ (s_U[sj - 2][si] == 0) *fmaxf((s_h[sj - 2][si] - D[globalIdx - n_d * 2]), (s_h[sj - 2][si + 1] - D[globalIdx + 1 - n_d * 2])))
			< droogval);

		s_kfV[sj - 2][si] = 1 -
			     (((s_V[sj - 2][si] > 0)* (s_h[sj - 2][si] - D[globalIdx - n_d * 2])
				+ (s_V[sj - 2][si] < 0) * (s_h[sj - 1][si] - D[globalIdx - n_d * 1])
				+ (s_V[sj - 2][si] == 0) * fmaxf((s_h[sj - 2][si] - D[globalIdx - n_d * 2]), (s_h[sj - 1][si] - D[globalIdx - n_d * 1])) )
				< droogval);


		s_kfU[sj + BLOCK_SIZE_y][si] = 1 - (
			    ((s_U[sj + BLOCK_SIZE_y][si] > 0) *(s_h[sj + BLOCK_SIZE_y][si]-D[globalIdx + n_d * (BLOCK_SIZE_y)])+
			     (s_U[sj + BLOCK_SIZE_y][si] < 0) *(s_h[sj + BLOCK_SIZE_y][si + 1] - D[globalIdx + 1+ n_d * (BLOCK_SIZE_y)]) +
				(s_U[sj + BLOCK_SIZE_y][si] == 0) *fmaxf((s_h[sj + BLOCK_SIZE_y][si] - D[globalIdx + n_d * (BLOCK_SIZE_y)]), (s_h[sj + BLOCK_SIZE_y][si + 1] - D[globalIdx + 1 + n_d * (BLOCK_SIZE_y)]))) < droogval);

		s_kfV[sj + BLOCK_SIZE_y][si] = 1 - ( ((blockIdx.y != gridDim.y - 1) ?
			      ((s_V[sj + BLOCK_SIZE_y][si] > 0)* (s_h[sj + BLOCK_SIZE_y][si]- D[globalIdx + n_d * (BLOCK_SIZE_y )])
				 + (s_V[sj + BLOCK_SIZE_y][si] < 0) * (h[globalIdx + n_d * (BLOCK_SIZE_y + 1)]- D[globalIdx + n_d * (BLOCK_SIZE_y + 1)])
				+ (s_V[sj + BLOCK_SIZE_y][si] == 0) * fmaxf((h[globalIdx + n_d * (BLOCK_SIZE_y + 1)] - D[globalIdx + n_d * (BLOCK_SIZE_y + 1)]), (s_h[sj + BLOCK_SIZE_y][si] - D[globalIdx + n_d * (BLOCK_SIZE_y)]) ) ) :
					  (s_h[sj + BLOCK_SIZE_y][si] - D[globalIdx + n_d * (BLOCK_SIZE_y)]) ) < droogval );


	}

	if ((threadIdx.y == 2 | threadIdx.y == 3) /*& threadIdx.y < BLOCK_SIZE_y*/) {

		unsigned int jtemp = blockIdx.y*blockDim.y + threadIdx.x;
		unsigned int itemp = blockIdx.x*blockDim.x + threadIdx.y;
		unsigned int si2 = threadIdx.y; // local i for shared memory access + halo offset
		unsigned int sj2 = threadIdx.x + 2; // local j for shared memory access
		unsigned int globalIdx2 = (jtemp + 2)* n_d + itemp;




		s_kfU[sj2][si2 - 2] = 1 - (((s_U[sj2][si2 - 2] > 0) *(s_h[sj2][si2 - 2]-D[globalIdx2-2])
								  + (s_U[sj2][si2 - 2] < 0) *(s_h[sj2][si2 - 1]-D[globalIdx2-1])
		+ (s_U[sj2][si2 - 2] == 0) *fmaxf(s_h[sj2][si2 - 1] - D[globalIdx2 - 1], (s_h[sj2][si2 - 2]-D[globalIdx2-2]))) < droogval);


		s_kfV[sj2][si2 - 2] =       1 - (((s_V[sj2][si2 - 2] > 0)  * (s_h[sj2][si2 - 2]-D[globalIdx2-2])
								    + (s_V[sj2][si2 - 2] < 0)  * (s_h[sj2 + 1][si2 - 2]-D[globalIdx2-2+n_d])
			+ (s_V[sj2][si2 - 2] == 0)* fmaxf((s_h[sj2][si2 - 2]-D[globalIdx2-2]), (s_h[sj2 + 1][si2 - 2]-D[globalIdx2-2+n_d]))) < droogval);

		s_kfU[sj2][si2 + BLOCK_SIZE_x] = 1 - (((blockIdx.x != gridDim.x - 1) ?
			      (s_U[sj2][si2 + BLOCK_SIZE_x] > 0) *(s_h[sj2][si2 + BLOCK_SIZE_x]- D[globalIdx2 + BLOCK_SIZE_x ])
			+ (s_U[sj2][si2 + BLOCK_SIZE_x] < 0) *(h[globalIdx2 + BLOCK_SIZE_x + 1] - D[globalIdx2 + BLOCK_SIZE_x + 1])
			+ (s_U[sj2][si2 + BLOCK_SIZE_x] == 0) *fmaxf((s_h[sj2][si2 + BLOCK_SIZE_x] - D[globalIdx2 + BLOCK_SIZE_x]), (h[globalIdx2 + BLOCK_SIZE_x + 1] - D[globalIdx2 + BLOCK_SIZE_x + 1])) : s_h[sj2][si2 + BLOCK_SIZE_x]-D[globalIdx2 + BLOCK_SIZE_x]) < droogval);

		s_kfV[sj2][si2 + BLOCK_SIZE_x] =         1 - (((s_V[sj2][si2 + BLOCK_SIZE_x] > 0) * (s_h[sj2][si2 + BLOCK_SIZE_x]-D[globalIdx2+BLOCK_SIZE_x])
			                                    + (s_V[sj2][si2 + BLOCK_SIZE_x] < 0)  *     (s_h[sj2 + 1][si2 + BLOCK_SIZE_x] - D[globalIdx2 + BLOCK_SIZE_x+n_d])
			+ (s_V[sj2][si2 + BLOCK_SIZE_x] == 0) * fmaxf(s_h[sj2 + 1][si2 + BLOCK_SIZE_x] - D[globalIdx2 + BLOCK_SIZE_x + n_d], (s_h[sj2][si2 + BLOCK_SIZE_x] - D[globalIdx2 + BLOCK_SIZE_x]))) < droogval);

		if (tidal_case && blockIdx.x == 0) {
			  s_h[sj2][si2-1 ] = Hstart /*- D[globalIdx2 - 1]*/ + tideheight;
			  h[globalIdx2 - 1] = Hstart + tideheight;
			s_kfU[sj2][si2 -1] = 1;

			if (sj2 == 2 | sj2 == 3) {
				  s_h[sj2 - 2][si2 - 1] = Hstart /*- D[globalIdx2 - 1]*/ + tideheight;
				  h[globalIdx2 - 1 - 2*n_d] = Hstart + tideheight;
				s_kfU[sj2 - 2][si2 - 1] = 1;
				

				s_h[sj2 + BLOCK_SIZE_y][si2 - 1] = Hstart /*- D[globalIdx2 - 1]*/ + tideheight;
				h[globalIdx2 - 1 + BLOCK_SIZE_y*n_d] = Hstart + tideheight;
				s_kfU[sj2 + BLOCK_SIZE_y][si2 - 1] = 1;
				
			}
		}


	}




	__syncthreads();
	s_hx =  (s_U[sj][si] > 0) *s_h[sj][si] +
		    (s_U[sj][si] < 0) *s_h[sj][si + 1] +
		    (s_U[sj][si] == 0) *fmaxf(s_h[sj][si], s_h[sj][si + 1]);

	Hx[globalIdx] = s_hx;

	s_dx = (s_U[sj][si] > 0)*s_D[sj][si] +
		   (s_U[sj][si] < 0)*s_D[sj][si + 1] +
		 (s_U[sj][si] == 0) *fmaxf(s_D[sj][si], s_D[sj][si + 1]);

	s_hy =    (s_V[sj][si] > 0)* s_h[sj][si]
		   + (s_V[sj][si] < 0) * s_h[sj + 1][si]
		  + (s_V[sj][si] == 0) * fmaxf(s_h[sj + 1][si], s_h[sj][si]);

	Hy[globalIdx] = s_hy;

	s_dy = (s_V[sj][si] > 0) * s_D[sj][si]
		 + (s_V[sj][si] < 0) * s_D[sj + 1][si]
		+ (s_V[sj][si] == 0) * fmaxf(s_D[sj + 1][si], s_D[sj][si]);


	__syncthreads();

	//wetting/drying
	s_kfU[sj][si] = 1 - ((s_hx-s_dx) < droogval);
	s_kfV[sj][si] = 1 - ((s_hy-s_dy) < droogval);


	__syncthreads();

	if (blockIdx.x == gridDim.x - 1 && sj == BLOCK_SIZE_x + 1) {
		s_kfU[si][sj] = 0;
	}
	if (blockIdx.y == gridDim.y - 1 && sj == BLOCK_SIZE_y + 1) {
		s_kfV[sj][si] = 0;
	}


	__syncthreads();

	//update U 
	Utemp[globalIdx] = 
		s_kfU[sj][si] * (s_U[sj][si] - g * dt / dx * alpha*(s_h[sj][si + 1] - s_h[sj][si] /*+ D[globalIdx + 1] - D[globalIdx]*/) +
		-dt / (dx*(1 + s_kfU[sj][si - 1] * s_kfU[sj][si + 1])) * (s_kfU[sj][si + 1] * (s_U[sj][si + 1] - s_U[sj][si]) + s_kfU[sj][si - 1] * (s_U[sj][si] - s_U[sj][si - 1]))* s_U[sj][si] +
		(s_V[sj][si] > 0) * -dt / ((1 + s_kfU[sj - 2][si])*dy) * s_kfU[sj - 1][si] * ((1 + 2 * s_kfU[sj - 2][si])*s_U[sj][si] - (1 + 3 * s_kfU[sj - 2][si])*s_U[sj - 1][si] + s_kfU[sj - 2][si] * s_U[sj - 2][si])  *(s_V[sj - 1][si] + s_V[sj - 1][si + 1]) / 2
		+ (s_V[sj][si] < 0) * -dt / ((1 + s_kfU[sj + 2][si])*dy) * s_kfU[sj + 1][si] * (-(1 + 2 * s_kfU[sj + 2][si])*s_U[sj][si] + (1 + 3 * s_kfU[sj + 2][si])*s_U[sj + 1][si] - s_kfU[sj + 2][si] * s_U[sj + 2][si]) * (s_V[sj][si] + s_V[sj][si + 1]) / 2);

	// update V

	Vtemp[globalIdx] = 
		s_kfV[sj][si] * (s_V[sj][si] - g * dt / dy * alpha*(s_h[sj + 1][si] - s_h[sj][si] /*+ D[globalIdx + n_d] - D[globalIdx]*/)
		- dt / (dy*(1 + s_kfV[sj - 1][si] * s_kfV[sj + 1][si])) * (s_kfV[sj + 1][si] * (s_V[sj + 1][si] - s_V[sj][si]) + s_kfV[sj - 1][si] * (s_V[sj][si] - s_V[sj - 1][si]))* s_V[sj][si] +
		+(s_U[sj][si] > 0) * -dt / ((1 + s_kfV[sj][si - 2])*dx) * s_kfV[sj][si - 1] * ((1 + 2 * s_kfV[sj][si - 2])*s_V[sj][si] - (1 + 3 * s_kfV[sj][si - 2])*s_V[sj][si - 1] + s_kfV[sj][si - 2] * s_V[sj][si - 2])     *(s_U[sj + 1][si - 1] + s_U[sj][si - 1]) / 2 +
		+(s_U[sj][si] < 0) * -dt / ((1 + s_kfV[sj][si + 2])*dx) * s_kfV[sj][si + 1] * (-(1 + 2 * s_kfV[sj][si + 2])*s_V[sj][si] + (1 + 3 * s_kfV[sj][si + 2])*s_V[sj][si + 1] - s_kfV[sj][si + 2] * s_V[sj][si + 2])     *(s_U[sj + 1][si] + s_U[sj][si]) / 2);

	kfUtemp[globalIdx] =  s_kfU[sj][si];
	kfVtemp[globalIdx] =  s_kfV[sj][si];










}

__global__ void
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
calcRHS(float *U, float *V, float* h, float* hx, float* hy, float* rowfact, float dt, float* RHS)
{



	__shared__       float   s_hx[BLOCK_SIZE_y + 1][BLOCK_SIZE_x + 1]; // 4-wide halo
	__shared__       float   s_hy[BLOCK_SIZE_y + 1][BLOCK_SIZE_x + 1]; // 4-wide halo
	__shared__        float   s_U[BLOCK_SIZE_y + 1][BLOCK_SIZE_x + 1]; // 4-wide halo
	__shared__        float   s_V[BLOCK_SIZE_y + 1][BLOCK_SIZE_x + 1]; // 4-wide halo		
	/*__shared__		__int8 s_kfU[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4];
	__shared__		__int8 s_kfV[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4];*/



	//int i = threadIdx.x;
	//int j = blockIdx.x*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int si = threadIdx.x + 1; // local i for shared memory access + halo offset
	unsigned int sj = threadIdx.y + 1; // local j for shared memory access
	unsigned int globalIdx = (j + 2) * n_d + i + 2;


	s_U[sj][si] = U[globalIdx];
	s_V[sj][si] = V[globalIdx];
	s_hx[sj][si] = hx[globalIdx];
	s_hy[sj][si] = hy[globalIdx];

	//Borders
	if (threadIdx.y == 0 ) {

		/*if (blockIdx.x == 0 && blockIdx.y ==0) {
			Htemp[globalIdx - 2 * n_d] = 1.5;
			Htemp[globalIdx + n_d * BLOCK_SIZE_y] = 1.5;
		}*/

		s_hx[sj - 1][si] = hx[globalIdx -  n_d];
		s_hy[sj - 1][si] = hy[globalIdx -  n_d];
		 s_U[sj - 1][si] =  U[globalIdx -  n_d];
		 s_V[sj - 1][si] =  V[globalIdx -  n_d];

		/*s_h[sj + BLOCK_SIZE_y][si] = h[globalIdx + n_d * (BLOCK_SIZE_y)];
		s_h[sj + BLOCK_SIZE_y][si] = h[globalIdx + n_d * (BLOCK_SIZE_y)];
		s_U[sj + BLOCK_SIZE_y][si] = U[globalIdx + n_d * (BLOCK_SIZE_y)];
		s_V[sj + BLOCK_SIZE_y][si] = V[globalIdx + n_d * (BLOCK_SIZE_y)];*/


	}

	if ((threadIdx.y == 0 ) /*& threadIdx.y < BLOCK_SIZE_y*/) {

		unsigned int jtemp = blockIdx.y*blockDim.y + threadIdx.x;
		unsigned int itemp = blockIdx.x*blockDim.x + threadIdx.y;
		unsigned int si2 = threadIdx.y + 1; // local i for shared memory access + halo offset
		unsigned int sj2 = threadIdx.x + 1; // local j for shared memory access
		unsigned int globalIdx2 = (jtemp + 2)* n_d + itemp+2;

		s_hx[sj2][si2 - 1] = hx[globalIdx2 - 1];
		s_hy[sj2][si2 - 1] = hy[globalIdx2 - 1];
		 s_U[sj2][si2 - 1] =  U[globalIdx2 - 1];
		 s_V[sj2][si2 - 1] =  V[globalIdx2 - 1];

		//s_h[sj2][si2 + BLOCK_SIZE_x] = h[globalIdx2 + BLOCK_SIZE_x];/*BLOCK_SIZE_x*/
		//s_h[sj2][si2 + BLOCK_SIZE_x] = h[globalIdx2 + BLOCK_SIZE_x];
		//s_U[sj2][si2 + BLOCK_SIZE_x] = U[globalIdx2 + BLOCK_SIZE_x];
		//s_V[sj2][si2 + BLOCK_SIZE_x] = V[globalIdx2 + BLOCK_SIZE_x];



	}

	__syncthreads();
	//int rowfactIdx = ;
	RHS[j * (n_d - 4) + i] = (h[globalIdx] - dt / dx * (s_hx[sj][si] * s_U[sj][si] - s_hx[sj][si - 1] * s_U[sj][si - 1])
		- dt / dy * (s_hy[sj][si] * s_V[sj][si] - s_hy[sj - 1][si] * s_V[sj - 1][si]))*rowfact[j * (n_d - 4) + i];// [(j) * (n_d - 4) + i];


	
	//int test = rowfact[rowfactIdx];
	//bool kek = 1;



	//int test2 = rowfact[rowfactIdx];
	/*RHS = D(j, i) + H(j, i) + 1.*(-dt / dx * (Hx(j, i).*Uexpl(j, i) - Hx(j, i - 1).*Uexpl(j, i - 1))...
		- dt / dy * (Hy(j, i).*Vexpl(j, i) - Hy(j - 1, i).*Vexpl(j - 1, i)));*/










}

__global__ void
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
genMatrix(int *row_ptr, int *col_ind, float *val, float* s_Hx, float* s_Hy, __int8* s_kfU, __int8* s_kfV, float* rowfact, float diffact)
{

	//assert(M == N);
	int n = n_d - 4;
	int N = n * n;

	float rowF;
	int numels = 0;
	//assert(n*n == N);
	//printf("laplace dimension = %d\n", n);
	//int idx = 0;
	//int j = blockIdx.y*blockDim.y + threadIdx.y;
	//int i = blockIdx.x*blockDim.x + threadIdx.x;

	//kernel is launched in rows
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int ix = idx % n;
	int iy = idx / n;
	int sj = threadIdx.x / n;
	int si = threadIdx.x % n;
	int globalIdx = ix + 2 + (iy + 2)*n_d;
	float temp, temp2;
	//int globalIdx2 = ix + iy * (n_d - 4);

	//printf("%d \n", globalIdx);

	// s_Hx[globalIdx2] = Hx[globalIdx];
	// s_Hy[globalIdx2] = Hy[globalIdx];
	//s_kfU[globalIdx2] = kfU[globalIdx];
	//s_kfV[globalIdx2] = kfV[globalIdx];

	//Borders
	//if (threadIdx.x == 0) {

	//	s_Hx[sj][si + 1] = Hx[globalIdx - n_d];
	//	s_Hy[sj][si + 1] = Hy[globalIdx - n_d];
	//	s_kfU[sj][si + 1] = kfU[globalIdx - n_d];
	//	s_kfV[sj][si + 1] = kfV[globalIdx - n_d];


	//	int ix2 = idx / n;
	//	int iy2 = idx % n;
	//	int sj2 = threadIdx.x % n;
	//	int si2 = threadIdx.x / n;
	//	int globalIdx2 = ix2 + 2 + (iy2 + 2)*n_d;

	//	s_Hx[sj2 + 1][si2] = Hx[globalIdx2 - 1];
	//	s_Hy[sj2 + 1][si2] = Hy[globalIdx2 - 1];
	//	s_kfU[sj2 + 1][si2] = kfU[globalIdx2 - 1];
	//	s_kfV[sj2 + 1][si2] = kfV[globalIdx2 - 1];


	//}


	__syncthreads();

	float center = 0;
	// calculate number of elements preceding current line
	if (iy == 0 && ix > 0) {
		numels = 3 + (ix - 1) * 4;
	}
	else if (iy > 0 && iy < n - 1) {
		numels = (n - 2) * 4 + 6 + (iy - 1) * ((n - 2) * 5 + 8) + (ix > 0)*(4 + 5 * (ix - 1));

	}
	else if (iy == n - 1) {
		numels = (n - 2) * 4 + 6 + (iy - 1) * ((n - 2) * 5 + 8) + (ix > 0)*(3 + 4 * (ix - 1));

	}
	row_ptr[idx] = numels;

	rowF = 1.0 / ((1 + (iy == 0 || iy == n - 1))*(1 + (ix == 0 || ix == n - 1)));
	rowfact[n*iy + ix] = rowF;

	//top
	if (iy > 0) {
		temp = s_kfV[globalIdx-n_d] * s_Hy[globalIdx-n_d];
		val[numels] = -diffact * (1 + (iy == n - 1)) *temp * rowF;
		//val[numels] = (1 + (iy == n - 1)) * rowF;
		col_ind[numels] = idx - n;
		numels++;
		center = center + (1 + (iy == n - 1)) * temp;//Hy[idx - n];
	}

	//left
	if (ix > 0) {
		temp = s_kfU[globalIdx-1] * s_Hx[globalIdx-1];
		val[numels] = -diffact * (1 + (ix == n - 1))*temp * rowF;
		//val[numels] = (1 + (ix == n - 1))* rowF;
		col_ind[numels] = idx - 1;
		numels++;
		center = center + (1 + (ix == n - 1)) * temp;//Hx[idx - 1];
	}

	//center

	temp =  s_kfV[globalIdx] * s_Hy[globalIdx];
	temp2 = s_kfU[globalIdx] * s_Hx[globalIdx];

	val[numels] = (1 - diffact * (-center + -(iy < n - 1)*(1 + (iy == 0))*temp - (ix < n - 1)*(1 + (ix == 0))*temp2))* rowF;
	
	col_ind[numels] = idx;
	numels++;

	//right
	if (ix < n - 1) {
		val[numels] = -diffact * (1 + (ix == 0))*temp2 * rowF;
		
		col_ind[numels] = idx + 1;
		numels++;
	}
	//bottom
	if (iy < n - 1) {
		val[numels] = -diffact * (1 + (iy == 0))*temp * rowF;
		
		col_ind[numels] = idx + n;
		numels++;
	}
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		row_ptr[N] = (n - 2)*((n - 2) * 5 + 16) + 12;

	}



}

void CG(int* d_col, int* d_row, float* d_val, float* d_x, float* d_r, float*d_p, float* d_Ax, int N, int nz,
	cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle, cublasStatus_t cublasStatus, cusparseMatDescr_t descr)
{

	//	int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
	float *val = NULL;
	const float tol = 1e-5f;
	const int max_iter = 10 * N;
	//float *x;
	//float *rhs;
	float a, b, na, r0, r1;
	//int *d_col, *d_row;
	//float *d_val, *d_x, 
	float   dot;
	int k;
	float alpha, beta, alpham1;

	//// This will pick the best possible CUDA capable device
	//cudaDeviceProp deviceProp;
	//int devID = 0;
	//
	//checkCuda(cudaGetDeviceProperties(&deviceProp, devID));

	//// Statistics about the GPU device
	//printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
	//	deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	/* Generate a random tridiagonal symmetric matrix in CSR format */
	//M = N = 1048576;
	//nz = (N - 2) * 3 + 4;
	//I = (int *)malloc(sizeof(int)*(N + 1));
	//J = (int *)malloc(sizeof(int)*nz);
	//val = (float *)malloc(sizeof(float)*nz);
	//genTridiag(I, J, val, N, nz);

	//x = (float *)malloc(sizeof(float)*N);
	//rhs = (float *)malloc(sizeof(float)*N);

	/*for (int i = 0; i < N; i++)
	{
		rhs[i] = 1.0;
		x[i] = 0.0;
	}*/



	//checkCuda(cudaMalloc((void **)&d_col, nz * sizeof(int)));
	//checkCuda(cudaMalloc((void **)&d_row, (N + 1) * sizeof(int)));
	//checkCuda(cudaMalloc((void **)&d_val, nz * sizeof(float)));
	//checkCuda(cudaMalloc((void **)&d_x, N * sizeof(float)));
	//checkCuda(cudaMalloc((void **)&d_r, N * sizeof(float)));


	/*cudaMemcpy(d_col, J, nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, N * sizeof(float), cudaMemcpyHostToDevice);*/

	alpha = 1.0;
	alpham1 = -1.0;
	beta = 0.0;
	r0 = 0.;

	cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);

	cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
	cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

	k = 1;

	while (r1 > tol*tol && k <= max_iter)
	{
		if (k > 1)
		{
			b = r1 / r0;
			cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
			cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
		}
		else
		{
			cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
		}

		cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
		cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
		a = r1 / dot;

		cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
		na = -a;
		cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

		r0 = r1;
		cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
		cudaDeviceSynchronize();
		if (cgoutput ) {
			printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
		}
		k++;
	}
	if (cgoutput ) {
	printf("Final residual: %e\n", sqrt(r1));

	fprintf(stdout, "&&&& conjugateGradientUM %s\n", (sqrt(r1) < tol) ? "PASSED" : "FAILED");
	printf("\n");
}
	/*cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

	float rsum, diff, err = 0.0;

	for (int i = 0; i < N; i++)
	{
		rsum = 0.0;

		for (int j = I[i]; j < I[i + 1]; j++)
		{
			rsum += val[j] * x[J[j]];
		}

		diff = fabs(rsum - rhs[i]);

		if (diff > err)
		{
			err = diff;
		}
	}*/

	

}

__global__ void
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
postCG(float* Htemp, float* H, float* Utemp, float* U, float* Vtemp, float* V, __int8* kfU, __int8* kfV, float alpha, float dt) {

	__shared__       float   s_h[BLOCK_SIZE_y + 1][BLOCK_SIZE_x + 1]; // 4-wide halo

	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int si = threadIdx.x; // local i for shared memory access + halo offset
	unsigned int sj = threadIdx.y; // local j for shared memory access
	unsigned int globalIdx = (j + 2) * n_d + i + 2;
	unsigned int CGIdx = j * (n_d - 4) + i;
	
	s_h[sj][si] = Htemp[CGIdx];

	if (threadIdx.y == BLOCK_SIZE_y-1 && blockIdx.y < (gridDim.y-1 ) )  {

		int sanity = CGIdx + n_d - 4;
		s_h[sj + 1][si] =  Htemp[sanity];

}

	//if (threadIdx.x == BLOCK_SIZE_x - 1 && blockIdx.y < (gridDim.y )) {

	//	s_h[sj ][si + 1] = Htemp[CGIdx + 1];

	//}

if (threadIdx.y == BLOCK_SIZE_y-1 && blockIdx.x < (gridDim.x-1 )) {
	unsigned int j2 = blockIdx.y*blockDim.y + threadIdx.x;
	unsigned int i2 = blockIdx.x*blockDim.x + threadIdx.y;
	unsigned int sj2 = threadIdx.x ;
	unsigned int si2 = threadIdx.y;

	s_h[sj2][si2 + 1] =  Htemp[(j2)*(n_d - 4) + i2 + 1];

	//if (threadIdx.x ==0){
	//	s_h[BLOCK_SIZE_y][BLOCK_SIZE_x] = 100000;// Htemp[(j2 + BLOCK_SIZE_y)*(n_d - 4) + i2 + BLOCK_SIZE_x];
	//}
}


__syncthreads();

H[globalIdx] = s_h[sj][si];

U[globalIdx] =  Utemp[globalIdx] - (1 - alpha)*g*dt / dx * kfU[globalIdx] * (s_h[sj][si + 1] - s_h[sj][si]);

V[globalIdx] =  Vtemp[globalIdx] - (1 - alpha)*g*dt / dy * kfV[globalIdx] * (s_h[sj + 1][si] - s_h[sj][si]);

}

void printfile(float* H_h, float*D_h);

	void updatecputhreadnew(float* h, float* U, float* V, float *D, __int8* kfU, __int8* kfV,  float dt, int tid, int numthreads, int iter,int ghost, bool tidal_case, int* pointer)
	{
		const int BLOCK_SIZE_x = n - ghost;
		const int BLOCK_SIZE_y = ceil((float)(n - ghost) / (float)numthreads);
		int globalIdx, si, sj;
		const float g = g_h;
		const float dx = dx_h;
		const float dy = dy_h;
		//float s_upos; float s_vpos;		
		int halo = ghost / 2;

		float tideheight = 0;
		float t = 0;

		float** s_h = NULL; 
		float** s_U = NULL;
		float** s_V = NULL; 
		float** s_hx = NULL;
		float** s_hy = NULL;		
		bool** s_kfU = NULL; 
		bool** s_kfV = NULL;

		s_h = new float*[(BLOCK_SIZE_y + ghost)]; 
		s_U = new float*[(BLOCK_SIZE_y + ghost)]; 
		s_V = new float*[(BLOCK_SIZE_y + ghost)];
		s_kfU = new bool*[(BLOCK_SIZE_y + ghost)]; 
		s_kfV = new bool*[(BLOCK_SIZE_y + ghost)];
		s_hx = new float*[BLOCK_SIZE_y + ghost];
		s_hy = new float*[BLOCK_SIZE_y + ghost];

		// Create a row for every pointer 
		for (int k = 0; k <= BLOCK_SIZE_y + ghost; k++)
		{
			  s_h[k] = new float[BLOCK_SIZE_x + ghost];
			  s_U[k] = new float[BLOCK_SIZE_x + ghost];
			  s_V[k] = new float[BLOCK_SIZE_x + ghost];
			s_kfU[k] = new bool[BLOCK_SIZE_x + ghost];
			s_kfV[k] = new bool[BLOCK_SIZE_x + ghost];
			 s_hx[k] = new float[BLOCK_SIZE_x + ghost];
			 s_hy[k] = new float[BLOCK_SIZE_x + ghost];

		}


		// copy global variables into shared memory
		for (sj = 0; sj < BLOCK_SIZE_y + 2 * halo; sj++) {
			for (si = 0; si < BLOCK_SIZE_x + 2*halo; si++) {
				globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
				
				s_h[sj][si] = h[globalIdx];
				s_U[sj][si] = U[globalIdx];
				s_V[sj][si] = V[globalIdx];
				s_kfU[sj][si] = kfU[globalIdx];
				s_kfV[sj][si] = kfV[globalIdx];
				//s_upos[sj][si] = s_U[sj][si] >= 0;
				//s_vpos[sj][si] = s_V[sj][si] >= 0;
			}
		}
		


		/*cb->await();
		if (tid==0) cb->reset();
		printf("tid: %d \n", tid);*/

		////Boundaries
		//si = halo;
		//for (int offset = 1; offset <= halo; offset++) {
		//	for (sj = halo; sj < BLOCK_SIZE_y + halo; sj++) {

		//		globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;

		//		s_h[sj][si - offset] = h[globalIdx - offset];
		//		s_U[sj][si - offset] = U[globalIdx - offset];
		//		s_V[sj][si - offset] = V[globalIdx - offset];
		//		s_kfU[sj][si - offset] = kfU[globalIdx - offset];
		//		s_kfV[sj][si - offset] = kfV[globalIdx - offset];

		//		s_h[sj][si + BLOCK_SIZE_x - 1 + offset] = h[globalIdx + BLOCK_SIZE_x - 1 + offset];
		//		s_U[sj][si + BLOCK_SIZE_x - 1 + offset] = U[globalIdx + BLOCK_SIZE_x - 1 + offset];
		//		s_V[sj][si + BLOCK_SIZE_x - 1 + offset] = V[globalIdx + BLOCK_SIZE_x - 1 + offset];
		//		s_kfU[sj][si + BLOCK_SIZE_x - 1 + offset] = kfU[globalIdx + BLOCK_SIZE_x - 1 + offset];
		//		s_kfV[sj][si + BLOCK_SIZE_x - 1 + offset] = kfV[globalIdx + BLOCK_SIZE_x - 1 + offset];

		//		if (offset == 1) {
		//			s_hx[sj][si - 1] = ((s_U[sj][si - 1] >= 0) *s_h[sj][si - 1]
		//				+ (s_U[sj][si - 1] < 0) *s_h[sj][si]);
		//		}
		//	}
		//}
		//sj = halo;
		//#pragma unroll
		//for (int offset = 1; offset <= halo; offset++) {
		//	for (si = halo; si < BLOCK_SIZE_x + halo; si++) {


		//		globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
		//	

		//		s_h[sj - offset][si] = h[globalIdx - offset * n];
		//		s_U[sj - offset][si] = U[globalIdx - offset * n];
		//		s_V[sj - offset][si] = V[globalIdx - offset * n];
		//		s_kfV[sj - offset][si] = kfV[globalIdx - offset * n];
		//		s_kfU[sj - offset][si] = kfU[globalIdx - offset * n];

		//		s_h[sj + BLOCK_SIZE_y - 1 + offset][si] = h[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
		//		s_U[sj + BLOCK_SIZE_y - 1 + offset][si] = U[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
		//		s_V[sj + BLOCK_SIZE_y - 1 + offset][si] = V[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
		//		s_kfV[sj + BLOCK_SIZE_y - 1 + offset][si] = kfV[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
		//		s_kfU[sj + BLOCK_SIZE_y - 1 + offset][si] = kfU[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
		//		
		//		if (offset == 1) {
		//			s_hy[sj - 1][si] = ((s_V[sj - 1][si] >= 0) *s_h[sj - 1][si]
		//				+ (s_V[sj - 1][si] < 0)  *s_h[sj][si]);
		//		}
		//	}
		//}
		//cb->get_current_waiting();

		syncthreads(1, pointer, numthreads, tid);
		
		
		
		

		//initial calculation of hx, hy, kfU and kfV
		for (sj = 0; sj < BLOCK_SIZE_y + 2*halo-1; sj++) {
			for (si = 0; si < BLOCK_SIZE_x + 2*halo-1; si++) {
				globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
			
				

				s_hy[sj][si] =
					  (V[globalIdx] > 0) * s_h[sj][si]
					+ (V[globalIdx] < 0) * s_h[sj + 1][si]
					+ (V[globalIdx] == 0)*fmax(s_h[sj][si], s_h[sj + 1][si]);

				s_hx[sj][si] =
					  (U[globalIdx] > 0 ) *s_h[sj][si]
					+((U[globalIdx] < 0 ) *s_h[sj][si + 1])
				     +(U[globalIdx] == 0) *fmax(s_h[sj][si + 1], s_h[sj][si ]);

				//wetting/drying
				s_kfU[sj][si ] = 1 - (s_hx[sj][si ] < droogval_h);
				s_kfV[sj ][si] = 1 - (s_hy[sj ][si] < droogval_h);
				/*if (si == BLOCK_SIZE_x) {
					s_kfU[sj][si] = 0;
				}
				if (tid == numthreads-1 && sj == BLOCK_SIZE_y) {
					s_kfV[sj][si] = 0;
				}*/
				if (si == BLOCK_SIZE_x + halo - 1) {
					s_kfU[sj][si] = 0;
				}

				if (sj == BLOCK_SIZE_y + halo - 1 && tid == numthreads - 1) {
					s_kfV[sj][si] = 0;
				}
				kfU[globalIdx] = s_kfU[sj][si];
				kfV[globalIdx] = s_kfV[sj][si];
			}
		}
		
		syncthreads(2, pointer, numthreads, tid);
		
			

		//main loop
		for (int z = 0; z < iter; z++) {
			
			
			//Boundaries
			si = halo;
			#pragma unroll
			for (int offset = 1; offset <= halo; offset++) {
				for ( sj = 0; sj < BLOCK_SIZE_y + 2*halo; sj++) {

					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;

					s_h[sj][si - offset] = h[globalIdx - offset];
					s_U[sj][si - offset] = U[globalIdx - offset];
					s_V[sj][si - offset] = V[globalIdx - offset];
					s_kfU[sj][si - offset] = kfU[globalIdx - offset];
					s_kfV[sj][si - offset] = kfV[globalIdx - offset];

					s_h[sj][si + BLOCK_SIZE_x-1+offset] = h[globalIdx + BLOCK_SIZE_x-1+offset];
					s_U[sj][si + BLOCK_SIZE_x-1+offset] = U[globalIdx + BLOCK_SIZE_x-1+offset];
					s_V[sj][si + BLOCK_SIZE_x-1+offset] = V[globalIdx + BLOCK_SIZE_x-1+offset];
					s_kfU[sj][si + BLOCK_SIZE_x - 1 + offset] = kfU[globalIdx + BLOCK_SIZE_x - 1 + offset];
					s_kfV[sj][si + BLOCK_SIZE_x - 1 + offset] = kfV[globalIdx + BLOCK_SIZE_x - 1 + offset];

					
						/*if (sj2 == 2 | sj2 == 3) {
							s_h[sj2 - 2][si2 - 1] = Hstart - D[globalIdx2 - 1] + tideheight;
							s_kfU[sj2 - 2][si2 - 1] = 1;

							s_h[sj2 + BLOCK_SIZE_y][si2 - 1] = Hstart - D[globalIdx2 - 1] + tideheight;
							s_kfU[sj2 + BLOCK_SIZE_y][si2 - 1] = 1;

						}*/
					
				/*	if (offset == 1) {
						s_hx[sj][si - 1] = ((s_U[sj][si - 1] >= 0) *s_h[sj][si - 1]
							+ (s_U[sj][si - 1] < 0) *s_h[sj][si]);
					}*/
					
				}
			}
			sj= halo;
			#pragma unroll
			for (int offset = 1; offset <= halo; offset++) {
				for ( si = 0; si < BLOCK_SIZE_x + 2*halo; si++) {
				

					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
					/*s_h[sj - 1][si] = h[globalIdx - n];
					s_U[sj - 1][si] = U[globalIdx - n];
					s_V[sj - 1][si] = V[globalIdx - n];
					s_kfV[sj2 - 1][si2] = kfV[globalIdx2 -  n];
					s_kfU[sj2 - 1][si2] = kfU[globalIdx2 -  n];*/

					   s_h[sj - offset][si] =     h[globalIdx - offset * n];
					   s_U[sj - offset][si] =     U[globalIdx - offset * n];
					   s_V[sj - offset][si] =     V[globalIdx - offset * n];
					 s_kfV[sj - offset][si] =   kfV[globalIdx - offset * n];
					 s_kfU[sj - offset][si] =   kfU[globalIdx - offset * n];

					  s_h[sj + BLOCK_SIZE_y-1+offset][si] =   h[globalIdx + n * (BLOCK_SIZE_y-1+offset)];
					  s_U[sj + BLOCK_SIZE_y-1+offset][si] =   U[globalIdx + n * (BLOCK_SIZE_y-1+offset)];
					  s_V[sj + BLOCK_SIZE_y-1+offset][si] =   V[globalIdx + n * (BLOCK_SIZE_y-1+offset)];
					s_kfV[sj + BLOCK_SIZE_y-1+offset][si] = kfV[globalIdx + n * (BLOCK_SIZE_y-1+offset)];
					s_kfU[sj + BLOCK_SIZE_y-1+offset][si] = kfU[globalIdx + n * (BLOCK_SIZE_y-1+offset)];
					
					/*if (offset == 1) {
						s_hy[sj - 1][si] = ((s_V[sj - 1][si] >= 0) *s_h[sj - 1][si]
							+ (s_V[sj - 1][si] < 0)  *s_h[sj][si]);
					}*/
				}
			}

			if (tidal_case) {
				tideheight = 2 * sin(t * 2 * M_PI / (12 * 3600));
				dt = safety * dx_h / (2 * sqrt(2 * g_h*(Hstart_h + tideheight)));
				t = t + dt;
				if (t == 3)//24 * 3600)
					break;
				for (si = 1; si <= 2; si++) {
					for (sj = 0; sj < BLOCK_SIZE_y + 2 * halo; sj++) {
						globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;

						h[globalIdx] = Hstart_h - D[globalIdx] + tideheight;
						kfU[globalIdx] = 1;
						s_h[sj][si] = Hstart_h - D[globalIdx] + tideheight;
						s_kfU[sj][si] = 1;
					}
				}
			}

			syncthreads(1, pointer, numthreads, tid);
			

			//update  & V
			#pragma unroll
			for ( sj= halo; sj < BLOCK_SIZE_y + halo; sj++) {
				for ( si = halo; si < BLOCK_SIZE_x + halo; si++) {

					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;

					// s_upos = s_U[sj][si] >= 0;
					
					U[globalIdx] = s_kfU[sj][si] * (s_U[sj][si] - g * dt / dx * (s_h[sj][si + 1] - s_h[sj][si] + D[globalIdx + 1] - D[globalIdx]) +
						-dt / (dx*(1 + s_kfU[sj][si - 1] * s_kfU[sj][si + 1])) * (s_kfU[sj][si + 1] * (s_U[sj][si + 1] - s_U[sj][si]) + s_kfU[sj][si - 1] * (s_U[sj][si] - s_U[sj][si - 1]))* s_U[sj][si] +
					  (s_V[sj][si] > 0) * -dt / ((1 + s_kfU[sj - 2][si])*dy) * s_kfU[sj - 1][si] * ( (1 + 2 * s_kfU[sj - 2][si])*s_U[sj][si] - (1 + 3 * s_kfU[sj - 2][si])*s_U[sj - 1][si] + s_kfU[sj - 2][si] * s_U[sj - 2][si])     *(s_V[sj - 1][si] + s_V[sj - 1][si + 1]) / 2 
					+ (s_V[sj][si] < 0) * -dt / ((1 + s_kfU[sj + 2][si])*dy) * s_kfU[sj + 1][si] * (-(1 + 2 * s_kfU[sj + 2][si])*s_U[sj][si] + (1 + 3 * s_kfU[sj + 2][si])*s_U[sj + 1][si] - s_kfU[sj + 2][si] * s_U[sj + 2][si])     *(s_V[sj][si]     + s_V[sj][si + 1])     / 2);
					
					V[globalIdx] = s_kfV[sj][si] * (s_V[sj][si] - g * dt / dy * (s_h[sj + 1][si] - s_h[sj][si] + D[globalIdx + n] - D[globalIdx])
						- dt / (dy*(1 + s_kfV[sj - 1][si] * s_kfV[sj + 1][si])) * (s_kfV[sj + 1][si] * (s_V[sj + 1][si] - s_V[sj][si]) + s_kfV[sj - 1][si] * (s_V[sj][si] - s_V[sj - 1][si]))* s_V[sj][si] +
						+(s_U[sj][si] > 0) * -dt / ((1 + s_kfV[sj][si - 2])*dx) * s_kfV[sj][si - 1] * ((1 + 2 * s_kfV[sj][si - 2])*s_V[sj][si] - (1 + 3 * s_kfV[sj][si - 2])*s_V[sj][si - 1] + s_kfV[sj][si - 2] * s_V[sj][si - 2])     *(s_U[sj + 1][si - 1] + s_U[sj][si - 1]) / 2 
					    +(s_U[sj][si] < 0) * -dt / ((1 + s_kfV[sj][si + 2])*dx) * s_kfV[sj][si + 1] * (-(1 + 2 * s_kfV[sj][si + 2])*s_V[sj][si] + (1 + 3 * s_kfV[sj][si + 2])*s_V[sj][si + 1] - s_kfV[sj][si + 2] * s_V[sj][si + 2])    *(s_U[sj + 1][si]     + s_U[sj][si])     / 2);

					//s_upos[sj][si] = (U[globalIdx] >= 0);

					////update U 
					//U[globalIdx] = s_kfU[sj][si] * (s_U[sj][si] - g * dt / dx * (s_h[sj][si + 1] - s_h[sj][si] + D[globalIdx + 1] - D[globalIdx]) +
					//	-dt / (dx*(1 + s_kfU[sj][si - 1] * s_kfU[sj][si + 1])) * (s_kfU[sj][si + 1] * (s_U[sj][si + 1] - s_U[sj][si]) + s_kfU[sj][si - 1] * (s_U[sj][si] - s_U[sj][si - 1]))* s_U[sj][si] +
					//	  (s_V[sj][si] > 0) * -dt / ((1 + s_kfU[sj - 2][si])*dy) * s_kfU[sj - 1][si] * ((1 + 2 * s_kfU[sj - 2][si])*s_U[sj][si] - (1 + 3 * s_kfU[sj - 2][si])*s_U[sj - 1][si] + s_kfU[sj - 2][si] * s_U[sj - 2][si])  *(s_V[sj - 1][si] + s_V[sj - 1][si + 1]) / 2
					//	+ (s_V[sj][si] < 0) * -dt / ((1 + s_kfU[sj + 2][si])*dy) * s_kfU[sj + 1][si] * (-(1 + 2 * s_kfU[sj + 2][si])*s_U[sj][si] + (1 + 3 * s_kfU[sj + 2][si])*s_U[sj + 1][si] - s_kfU[sj + 2][si] * s_U[sj + 2][si]) * (s_V[sj][si] + s_V[sj][si + 1]) / 2);


					//// update V

					//V[globalIdx] = s_kfV[sj][si] * (s_V[sj][si] - g * dt / dy * (s_h[sj + 1][si] - s_h[sj][si] + D[globalIdx + n_d] - D[globalIdx])
					//	- dt / (dy*(1 + s_kfV[sj - 1][si] * s_kfV[sj + 1][si])) * (s_kfV[sj + 1][si] * (s_V[sj + 1][si] - s_V[sj][si]) + s_kfV[sj - 1][si] * (s_V[sj][si] - s_V[sj - 1][si]))* s_V[sj][si] +
					//	+(s_U[sj][si] > 0) * -dt / ((1 + s_kfV[sj][si - 2])*dx) * s_kfV[sj][si - 1] * ((1 + 2 * s_kfV[sj][si - 2])*s_V[sj][si] - (1 + 3 * s_kfV[sj][si - 2])*s_V[sj][si - 1] + s_kfV[sj][si - 2] * s_V[sj][si - 2])     *(s_U[sj + 1][si - 1] + s_U[sj][si - 1]) / 2 +
					//	+(s_U[sj][si] < 0) * -dt / ((1 + s_kfV[sj][si + 2])*dx) * s_kfV[sj][si + 1] * (-(1 + 2 * s_kfV[sj][si + 2])*s_V[sj][si] + (1 + 3 * s_kfV[sj][si + 2])*s_V[sj][si + 1] - s_kfV[sj][si + 2] * s_V[sj][si + 2])     *(s_U[sj + 1][si] + s_U[sj][si]) / 2);

				}
			}

			syncthreads(2, pointer, numthreads, tid);
			

			////write temp values to shared memory after sync and update upos
			//for (sj= halo; sj < BLOCK_SIZE_y + halo; sj++) {
			//	for (si = halo; si < BLOCK_SIZE_x + halo; si++) {

			//		globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
			//		s_U[sj][si] = U[globalIdx];

			//	}
			//}

//			syncthreads(1, pointer, numthreads, tid);
//			
//
//			//Boundaries
//			si = halo;
//#pragma unroll
//			for (int offset = 1; offset <= halo; offset++) {
//				for (sj = 0; sj < BLOCK_SIZE_y + 2*halo; sj++) {
//
//					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
//
//					s_h[sj][si - offset] = h[globalIdx - offset];
//					s_U[sj][si - offset] = U[globalIdx - offset];
//					s_V[sj][si - offset] = V[globalIdx - offset];
//					s_kfU[sj][si - offset] = kfU[globalIdx - offset];
//					s_kfV[sj][si - offset] = kfV[globalIdx - offset];
//
//					s_h[sj][si + BLOCK_SIZE_x - 1 + offset] = h[globalIdx + BLOCK_SIZE_x - 1 + offset];
//					s_U[sj][si + BLOCK_SIZE_x - 1 + offset] = U[globalIdx + BLOCK_SIZE_x - 1 + offset];
//					s_V[sj][si + BLOCK_SIZE_x - 1 + offset] = V[globalIdx + BLOCK_SIZE_x - 1 + offset];
//					s_kfU[sj][si + BLOCK_SIZE_x - 1 + offset] = kfU[globalIdx + BLOCK_SIZE_x - 1 + offset];
//					s_kfV[sj][si + BLOCK_SIZE_x - 1 + offset] = kfV[globalIdx + BLOCK_SIZE_x - 1 + offset];
//
//					/*	if (offset == 1) {
//							s_hx[sj][si - 1] = ((s_U[sj][si - 1] >= 0) *s_h[sj][si - 1]
//								+ (s_U[sj][si - 1] < 0) *s_h[sj][si]);
//						}*/
//				}
//			}
//			sj = halo;
//#pragma unroll
//			for (int offset = 1; offset <= halo; offset++) {
//				for (si = 0; si < BLOCK_SIZE_x + 2*halo; si++) {
//
//
//					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
//					/*s_h[sj - 1][si] = h[globalIdx - n];
//					s_U[sj - 1][si] = U[globalIdx - n];
//					s_V[sj - 1][si] = V[globalIdx - n];
//					s_kfV[sj2 - 1][si2] = kfV[globalIdx2 -  n];
//					s_kfU[sj2 - 1][si2] = kfU[globalIdx2 -  n];*/
//
//					s_h[sj - offset][si] = h[globalIdx - offset * n];
//					s_U[sj - offset][si] = U[globalIdx - offset * n];
//					s_V[sj - offset][si] = V[globalIdx - offset * n];
//					s_kfV[sj - offset][si] = kfV[globalIdx - offset * n];
//					s_kfU[sj - offset][si] = kfU[globalIdx - offset * n];
//
//					s_h[sj + BLOCK_SIZE_y - 1 + offset][si] = h[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
//					s_U[sj + BLOCK_SIZE_y - 1 + offset][si] = U[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
//					s_V[sj + BLOCK_SIZE_y - 1 + offset][si] = V[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
//					s_kfV[sj + BLOCK_SIZE_y - 1 + offset][si] = kfV[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
//					s_kfU[sj + BLOCK_SIZE_y - 1 + offset][si] = kfU[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
//
//					/*if (offset == 1) {
//						s_hy[sj - 1][si] = ((s_V[sj - 1][si] >= 0) *s_h[sj - 1][si]
//							+ (s_V[sj - 1][si] < 0)  *s_h[sj][si]);
//					}*/
//				}
//			}
//
//			syncthreads(2, pointer, numthreads, tid);
			


			////update V
			//#pragma unroll
			//for (sj= halo; sj < BLOCK_SIZE_y + halo; sj++) {
			//	for (si = halo; si < BLOCK_SIZE_x + halo; si++) {
			//		globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;

			//		// s_vpos = (V[globalIdx] >= 0);

			//		V[globalIdx] = s_kfV[sj][si] * (s_V[sj][si] - g * dt / dy * (s_h[sj + 1][si] - s_h[sj][si] + D[globalIdx + n] - D[globalIdx])
			//			- dt / (dy*(1 + s_kfV[sj - 1][si] * s_kfV[sj + 1][si])) * (s_kfV[sj + 1][si] * (s_V[sj + 1][si] - s_V[sj][si]) + s_kfV[sj - 1][si] * (s_V[sj][si] - s_V[sj - 1][si]))* s_V[sj][si] +
			//			+(s_U[sj][si]>0 ? -dt / ((1 + s_kfV[sj][si - 2])*dx) * s_kfV[sj][si - 1] * ((1 + 2 * s_kfV[sj][si - 2])*s_V[sj][si] - (1 + 3 * s_kfV[sj][si - 2])*s_V[sj][si - 1] + s_kfV[sj][si - 2] * s_V[sj][si - 2])     *(s_U[sj + 1][si - 1] + s_V[sj][si - 1]) / 2 :
			//				-dt / ((1 + s_kfV[sj][si + 2])*dx) * s_kfV[sj][si + 1] * (-(1 + 2 * s_kfV[sj][si + 2])*s_V[sj][si] + (1 + 3 * s_kfV[sj][si + 2])*s_V[sj][si + 1] - s_kfV[sj][si + 2] * s_V[sj][si + 2])     *(s_U[sj + 1][si] + s_V[sj][si]) / 2));

			//		
			//	}
			//}

			//syncthreads(1, pointer, numthreads, tid);
			

			//write U and V to memory
			#pragma unroll
			for (sj= halo; sj < BLOCK_SIZE_y + halo; sj++) {
				for (si = halo; si < BLOCK_SIZE_x + halo; si++) {

					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
					s_V[sj][si] = V[globalIdx];
					//write temp values to shared memory after sync and update upos									
					s_U[sj][si] = U[globalIdx];

					

				}
			}

			syncthreads(1, pointer, numthreads, tid);
			

			//Boundaries
			si = halo;
#pragma unroll
			for (int offset = 1; offset <= halo; offset++) {
				for (sj = 0; sj < BLOCK_SIZE_y + 2*halo; sj++) {

					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;

					s_h[sj][si - offset] = h[globalIdx - offset];
					s_U[sj][si - offset] = U[globalIdx - offset];
					s_V[sj][si - offset] = V[globalIdx - offset];
					s_kfU[sj][si - offset] = kfU[globalIdx - offset];
					s_kfV[sj][si - offset] = kfV[globalIdx - offset];

					s_h[sj][si + BLOCK_SIZE_x - 1 + offset] = h[globalIdx + BLOCK_SIZE_x - 1 + offset];
					s_U[sj][si + BLOCK_SIZE_x - 1 + offset] = U[globalIdx + BLOCK_SIZE_x - 1 + offset];
					s_V[sj][si + BLOCK_SIZE_x - 1 + offset] = V[globalIdx + BLOCK_SIZE_x - 1 + offset];
					s_kfU[sj][si + BLOCK_SIZE_x - 1 + offset] = kfU[globalIdx + BLOCK_SIZE_x - 1 + offset];
					s_kfV[sj][si + BLOCK_SIZE_x - 1 + offset] = kfV[globalIdx + BLOCK_SIZE_x - 1 + offset];

					/*	if (offset == 1) {
							s_hx[sj][si - 1] = ((s_U[sj][si - 1] >= 0) *s_h[sj][si - 1]
								+ (s_U[sj][si - 1] < 0) *s_h[sj][si]);
						}*/
				}
			}
			sj = halo;
#pragma unroll
			for (int offset = 1; offset <= halo; offset++) {
				for (si = 0; si < BLOCK_SIZE_x + 2*halo; si++) {


					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
					/*s_h[sj - 1][si] = h[globalIdx - n];
					s_U[sj - 1][si] = U[globalIdx - n];
					s_V[sj - 1][si] = V[globalIdx - n];
					s_kfV[sj2 - 1][si2] = kfV[globalIdx2 -  n];
					s_kfU[sj2 - 1][si2] = kfU[globalIdx2 -  n];*/

					s_h[sj - offset][si] = h[globalIdx - offset * n];
					s_U[sj - offset][si] = U[globalIdx - offset * n];
					s_V[sj - offset][si] = V[globalIdx - offset * n];
					s_kfV[sj - offset][si] = kfV[globalIdx - offset * n];
					s_kfU[sj - offset][si] = kfU[globalIdx - offset * n];

					s_h[sj + BLOCK_SIZE_y - 1 + offset][si] = h[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
					s_U[sj + BLOCK_SIZE_y - 1 + offset][si] = U[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
					s_V[sj + BLOCK_SIZE_y - 1 + offset][si] = V[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
					s_kfV[sj + BLOCK_SIZE_y - 1 + offset][si] = kfV[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];
					s_kfU[sj + BLOCK_SIZE_y - 1 + offset][si] = kfU[globalIdx + n * (BLOCK_SIZE_y - 1 + offset)];

					/*if (offset == 1) {
						s_hy[sj - 1][si] = ((s_V[sj - 1][si] >= 0) *s_h[sj - 1][si]
							+ (s_V[sj - 1][si] < 0)  *s_h[sj][si]);
					}*/
				}
			}

			syncthreads(2, pointer, numthreads, tid);
			

			//update H
			
			#pragma unroll
			for (sj= halo; sj < BLOCK_SIZE_y + halo; sj++) {
				for (si = halo; si < BLOCK_SIZE_x + halo; si++) {

					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
					//calculate hy
					

					// update h
					h[globalIdx] = s_h[sj][si] +
						(1 - (1 - s_kfV[sj][si])*(1 - s_kfV[sj - 1][si])*(1 - s_kfU[sj][si])*(1 - s_kfU[sj][si - 1]))*(
							- dt / dx * (s_hx[sj][si] * s_U[sj][si] - s_hx[sj][si - 1] * s_U[sj][si - 1])
							- dt / dy * (s_hy[sj][si] * s_V[sj][si] - s_hy[sj - 1][si] * s_V[sj - 1][si]));

					
					//breakpoint = h[globalIdx];
					//if (breakpoint < -.5 || breakpoint > 3) {

					//	long x, y;

					//	for (y = 0; y < n; y++)
					//	{
					//		for (x = 0; x < n; x++) {
					//			//printf("%s[%02ld][%02ld]=%6.2f  ", name, y, x, a[y*n + x]);
					//			if (h[y*n + x] == 0 || h[y*n + x] == 1)
					//			{
					//				printf("%d  ", (int)h[y*n + x]);
					//			}
					//			else {
					//				printf("%6.2f  ", h[y*n + x]);
					//			}
					//		}
					//		printf("\n");
					//	}
					//	printf("\n");
					//}
						



				}

			}
			
			syncthreads(1, pointer, numthreads, tid);
			

			if ((z- 1) % plotstep == 0 && realtimeplot && tid == 0) {

			
				printfile(h, D);				
				printf("plot, tide = %f, dt = %f \n", tideheight, dt);
				Sleep(plotdelay);
			}

			syncthreads(2, pointer, numthreads, tid);

			#pragma unroll
			for (sj = halo; sj < BLOCK_SIZE_y + halo; sj++) {
				for (si = halo; si < BLOCK_SIZE_x + halo; si++) {
					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
					s_h[sj][si] = h[globalIdx];

				}
			}

			syncthreads(1, pointer, numthreads, tid);
			
			for (sj = halo; sj < BLOCK_SIZE_y + halo; sj++) {
				for (si = halo; si < BLOCK_SIZE_x + halo; si++) {
					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;


					s_hy[sj][si] =
						(V[globalIdx] > 0) * s_h[sj][si]
						+ (V[globalIdx] < 0) * s_h[sj + 1][si]
						+ (V[globalIdx] == 0)*fmax(s_h[sj][si], s_h[sj + 1][si]);

					s_hx[sj][si] = ((U[globalIdx] > 0) *s_h[sj][si]
						+ ((U[globalIdx] < 0)) *s_h[sj][si + 1])
						+ ((U[globalIdx] == 0)) *fmax(s_h[sj][si + 1], s_h[sj][si]);
				}
			}
			// Hx, Hy, kfU and kfV
			#pragma unroll
			for (sj= halo; sj < BLOCK_SIZE_y + halo; sj++) {
				for (si = halo; si < BLOCK_SIZE_x + halo; si++) {
					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
				

					/*s_hy[sj][si] =
						  (V[globalIdx] > 0) * s_h[sj][si]
						+ (V[globalIdx] < 0) * s_h[sj + 1][si]
						+ (V[globalIdx] == 0)*fmax(s_h[sj][si], s_h[sj + 1][si]);

					s_hx[sj][si] = ((U[globalIdx] > 0) *s_h[sj][si]
						+ ((U[globalIdx] < 0)) *s_h[sj][si + 1])
					+((U[globalIdx] == 0)) *fmax(s_h[sj][si + 1], s_h[sj][si]);*/

					//wetting/drying
				
					s_kfV[sj][si] = 1 - (s_hy[sj][si] < droogval_h);
					s_kfU[sj][si] = 1 - (s_hx[sj][si] < droogval_h);

					if (si == BLOCK_SIZE_x + halo-1) {
						s_kfU[sj][si] = 0;
					}

					if (sj == BLOCK_SIZE_y + halo-1 && tid == numthreads - 1) {
						s_kfV[sj][si] = 0;
					}
					//kfU[globalIdx-1] = s_kfU[sj][si-1];
					//kfV[globalIdx-n] = s_kfV[sj-1][si];
					//s_h[sj][si] = h[globalIdx];

					kfU[globalIdx] = s_kfU[sj][si];
					kfV[globalIdx] = s_kfV[sj][si];
				}
			}

			syncthreads(2, pointer, numthreads, tid);
			

			/*#pragma unroll
			for (sj = halo; sj < BLOCK_SIZE_y + halo; sj++) {
				for (si = halo; si < BLOCK_SIZE_x + halo; si++) {
					globalIdx = si + (sj + BLOCK_SIZE_y * tid) *n;
					s_h[sj][si] = h[globalIdx];					
					kfU[globalIdx] = s_kfU[sj][si];
					kfV[globalIdx] = s_kfV[sj][si];
				}
			}

			cb->await();*/

			
		}
	}
	
	__global__ void
		__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
		updateUV(float *h, float *U, float *V, float* Utemp, float* Vtemp, float *D, float dt, bool tidal_case = false, float tideheight = 0)
	{



		__shared__       float   s_h[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4]; // 4-wide halo
		__shared__       float   s_U[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4]; // 4-wide halo
		__shared__       float   s_V[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4]; // 4-wide halo
		__shared__		__int8 s_kfU[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4];
		__shared__		__int8 s_kfV[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4];



		//int i = threadIdx.x;
		//int j = blockIdx.x*blockDim.y + threadIdx.y;
		unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
		unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int si = threadIdx.x + 2; // local i for shared memory access + halo offset
		unsigned int sj = threadIdx.y + 2; // local j for shared memory access
		unsigned int globalIdx = (j + 2) * n_d + i + 2;

		float s_hymin; float s_hxmin; float s_hx; float s_hy;


		s_U[sj][si] = U[globalIdx];
		s_V[sj][si] = V[globalIdx];
		s_h[sj][si] = h[globalIdx];

		//Borders
		if (threadIdx.y == 0 | threadIdx.y == 1) {

			/*if (blockIdx.x == 0 && blockIdx.y ==0) {
				Htemp[globalIdx - 2 * n_d] = 1.5;
				Htemp[globalIdx + n_d * BLOCK_SIZE_y] = 1.5;
			}*/

			s_h[sj - 2][si] = h[globalIdx - 2 * n_d];
			s_U[sj - 2][si] = U[globalIdx - 2 * n_d];
			s_V[sj - 2][si] = V[globalIdx - 2 * n_d];

			s_h[sj + BLOCK_SIZE_y][si] = h[globalIdx + n_d * (BLOCK_SIZE_y)];
			s_U[sj + BLOCK_SIZE_y][si] = U[globalIdx + n_d * (BLOCK_SIZE_y)];
			s_V[sj + BLOCK_SIZE_y][si] = V[globalIdx + n_d * (BLOCK_SIZE_y)];

			if (threadIdx.x == 0) {
				for (int offset = 1; offset <= 2; offset++) {
					s_h[sj - 2][si - offset] = h[globalIdx - 2 * n_d - offset];
					s_U[sj - 2][si - offset] = U[globalIdx - 2 * n_d - offset];
					s_V[sj - 2][si - offset] = V[globalIdx - 2 * n_d - offset];
					/*if(blockIdx.y == 74 && threadIdx.y == 1){
						int printvar = globalIdx - 2 * n_d - offset;
						int printvar2 = globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1;
						printf("Vglob: %d \n", printvar);
						printf("Vglob: %d \n", printvar2);

						}*/
					/*int nlocal = n_d;
					int testvar = n_d * n_d;
					int sanity1 = testvar - (globalIdx + (BLOCK_SIZE_y + sj) * n_d - offset);
					int sanity2 = testvar - (globalIdx + (BLOCK_SIZE_y + sj) * n_d + BLOCK_SIZE_x + offset - 1);
					int sanity6 = globalIdx + (BLOCK_SIZE_y + sj) * n_d - offset;
					int sanity8 = globalIdx + (32 + sj) * nlocal - offset;
					float sanity7 = h[sanity6];
					float sanity3 = h[globalIdx + (BLOCK_SIZE_y + sj) * n_d - offset];
					float sanity4 = U[globalIdx + (BLOCK_SIZE_y + sj) * n_d - offset];
					float sanity5 = V[globalIdx + (BLOCK_SIZE_y + sj) * n_d - offset];*/
					
					s_h[sj - 2][BLOCK_SIZE_x + offset + 1] = h[globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1];
					s_U[sj - 2][BLOCK_SIZE_x + offset + 1] = U[globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1];
					s_V[sj - 2][BLOCK_SIZE_x + offset + 1] = V[globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1];

					s_h[BLOCK_SIZE_y + sj][si - offset] = h[globalIdx + (BLOCK_SIZE_y + sj - 1) * n_d - offset];
					s_U[BLOCK_SIZE_y + sj][si - offset] = U[globalIdx + (BLOCK_SIZE_y + sj - 1) * n_d - offset];
					s_V[BLOCK_SIZE_y + sj][si - offset] = V[globalIdx + (BLOCK_SIZE_y + sj - 1) * n_d - offset];

					s_h[BLOCK_SIZE_y + sj][BLOCK_SIZE_x + offset + 1] = h[globalIdx + (BLOCK_SIZE_y + sj - 1) * n_d + BLOCK_SIZE_x + offset - 1];
					s_U[BLOCK_SIZE_y + sj][BLOCK_SIZE_x + offset + 1] = U[globalIdx + (BLOCK_SIZE_y + sj - 1) * n_d + BLOCK_SIZE_x + offset - 1];
					s_V[BLOCK_SIZE_y + sj][BLOCK_SIZE_x + offset + 1] = V[globalIdx + (BLOCK_SIZE_y + sj - 1) * n_d + BLOCK_SIZE_x + offset - 1];






				}
			}
		}
	
		if ((threadIdx.y == 2 | threadIdx.y == 3) /*& threadIdx.y < BLOCK_SIZE_y*/) {
		
			unsigned int jtemp = blockIdx.y*blockDim.y + threadIdx.x;
			unsigned int itemp = blockIdx.x*blockDim.x + threadIdx.y;
			unsigned int si2 = threadIdx.y; // local i for shared memory access + halo offset
			unsigned int sj2 = threadIdx.x + 2; // local j for shared memory access
			unsigned int globalIdx2 = (jtemp + 2)* n_d + itemp;

			s_h[sj2][si2 - 2] = h[globalIdx2 - 2];
			s_U[sj2][si2 - 2] = U[globalIdx2 - 2];
			s_V[sj2][si2 - 2] = V[globalIdx2 - 2];

			s_h[sj2][si2 + BLOCK_SIZE_x] = h[globalIdx2 + BLOCK_SIZE_x];/*BLOCK_SIZE_x*/
			s_U[sj2][si2 + BLOCK_SIZE_x] = U[globalIdx2 + BLOCK_SIZE_x];
			s_V[sj2][si2 + BLOCK_SIZE_x] = V[globalIdx2 + BLOCK_SIZE_x];

			

		}

		__syncthreads();

		// hxy and kfuv on the borders
		if (threadIdx.y == 0 | threadIdx.y == 1) {


			s_kfU[sj - 2][si] = 1 - (
				((s_U[sj - 2][si] > 0) *s_h[sj - 2][si]
					+ (s_U[sj - 2][si] < 0) *s_h[sj - 2][si + 1]
					+ (s_U[sj - 2][si] == 0) *fmaxf(s_h[sj - 2][si], s_h[sj - 2][si + 1]))
				< droogval);

			s_kfV[sj - 2][si] = 1 -
				(((s_V[sj - 2][si] > 0)* s_h[sj - 2][si]
					+ (s_V[sj - 2][si] < 0) * s_h[sj - 1][si]
					+ (s_V[sj - 2][si] == 0) * fmaxf(s_h[sj - 2][si], s_h[sj - 1][si]))
					< droogval);


			s_kfU[sj + BLOCK_SIZE_y][si] = 1 - (
				((s_U[sj + BLOCK_SIZE_y][si] > 0) *s_h[sj + BLOCK_SIZE_y][si] +
				(s_U[sj + BLOCK_SIZE_y][si] < 0) *s_h[sj + BLOCK_SIZE_y][si + 1] +
					(s_U[sj + BLOCK_SIZE_y][si] == 0) *fmaxf(s_h[sj + BLOCK_SIZE_y][si], s_h[sj + BLOCK_SIZE_y][si + 1])) < droogval);

			s_kfV[sj + BLOCK_SIZE_y][si] = 1 - (((blockIdx.y != gridDim.y - 1) ?
				((s_V[sj + BLOCK_SIZE_y][si] > 0)* s_h[sj + BLOCK_SIZE_y][si]
					+ (s_V[sj + BLOCK_SIZE_y][si] < 0) * h[globalIdx + n_d * (BLOCK_SIZE_y + 1)]
					+ (s_V[sj + BLOCK_SIZE_y][si] == 0) * fmaxf(h[globalIdx + n_d * (BLOCK_SIZE_y + 1)], s_h[sj + BLOCK_SIZE_y][si])) : s_h[sj + BLOCK_SIZE_y][si]) < droogval);

		}

		if ((threadIdx.y == 2 | threadIdx.y == 3) /*& threadIdx.y < BLOCK_SIZE_y*/) {

			unsigned int jtemp = blockIdx.y*blockDim.y + threadIdx.x;
			unsigned int itemp = blockIdx.x*blockDim.x + threadIdx.y;
			unsigned int si2 = threadIdx.y; // local i for shared memory access + halo offset
			unsigned int sj2 = threadIdx.x + 2; // local j for shared memory access
			unsigned int globalIdx2 = (jtemp + 2)* n_d + itemp;




			s_kfU[sj2][si2 - 2] = 1 - (((s_U[sj2][si2 - 2] > 0) *s_h[sj2][si2 - 2]
				+ (s_U[sj2][si2 - 2] < 0) *s_h[sj2][si2 - 1]
				+ (s_U[sj2][si2 - 2] == 0) *fmaxf(s_h[sj2][si2 - 1], s_h[sj2][si2 - 2])) < droogval);


			s_kfV[sj2][si2 - 2] = 1 - (((s_V[sj2][si2 - 2] > 0)  * s_h[sj2][si2 - 2]
				+ (s_V[sj2][si2 - 2] < 0)  * s_h[sj2 + 1][si2 - 2]
				+ (s_V[sj2][si2 - 2] == 0)* fmaxf(s_h[sj2][si2 - 2], s_h[sj2 + 1][si2 - 2])) < droogval);

			s_kfU[sj2][si2 + BLOCK_SIZE_x] = 1 - (((blockIdx.x != gridDim.x - 1) ?
				(s_U[sj2][si2 + BLOCK_SIZE_x] > 0) *s_h[sj2][si2 + BLOCK_SIZE_x]
				+ (s_U[sj2][si2 + BLOCK_SIZE_x] < 0) *h[globalIdx2 + BLOCK_SIZE_x + 1]
				+ (s_U[sj2][si2 + BLOCK_SIZE_x] == 0) *fmaxf(s_h[sj2][si2 + BLOCK_SIZE_x], h[globalIdx2 + BLOCK_SIZE_x + 1]) : s_h[sj2][si2 + BLOCK_SIZE_x]) < droogval);

			s_kfV[sj2][si2 + BLOCK_SIZE_x] = 1 - (((s_V[sj2][si2 + BLOCK_SIZE_x] > 0) * s_h[sj2][si2 + BLOCK_SIZE_x]
				+ (s_V[sj2][si2 + BLOCK_SIZE_x] < 0)  * s_h[sj2 + 1][si2 + BLOCK_SIZE_x]
				+ (s_V[sj2][si2 + BLOCK_SIZE_x] == 0) * fmaxf(s_h[sj2 + 1][si2 + BLOCK_SIZE_x], s_h[sj2][si2 + BLOCK_SIZE_x])) < droogval);

			if (tidal_case && blockIdx.x == 0) {
				s_h[sj2][si2 - 1] = Hstart - D[globalIdx2 - 1] + tideheight;
				s_kfU[sj2][si2 - 1] = 1;

				if (sj2 == 2 | sj2 == 3) {
					s_h[sj2 - 2][si2 - 1] = Hstart - D[globalIdx2 - 1] + tideheight;
					s_kfU[sj2 - 2][si2 - 1] = 1;

					s_h[sj2 + BLOCK_SIZE_y][si2 - 1] = Hstart - D[globalIdx2 - 1] + tideheight;
					s_kfU[sj2 + BLOCK_SIZE_y][si2 - 1] = 1;

				}
			}
		}

		s_hx = (s_U[sj][si] > 0) *s_h[sj][si] +
			(s_U[sj][si] < 0) *s_h[sj][si + 1] +
			(s_U[sj][si] == 0) *fmaxf(s_h[sj][si], s_h[sj][si + 1]);

		s_hy = (s_V[sj][si] > 0)* s_h[sj][si]
			+ (s_V[sj][si] < 0) * s_h[sj + 1][si]
			+ (s_V[sj][si] == 0) * fmaxf(s_h[sj + 1][si], s_h[sj][si]);

		__syncthreads();

		//wetting/drying
		s_kfU[sj][si] = 1 - (s_hx < droogval);
		s_kfV[sj][si] = 1 - (s_hy < droogval);


		__syncthreads();

		if (blockIdx.x == gridDim.x - 1 && sj == BLOCK_SIZE_x + 1) {
			s_kfU[si][sj] = 0;
		}
		if (blockIdx.y == gridDim.y - 1 && sj == BLOCK_SIZE_y + 1) {
			s_kfV[sj][si] = 0;
		}

		//update U 
		Utemp[globalIdx] = s_kfU[sj][si] * (s_U[sj][si] - g * dt / dx * (s_h[sj][si + 1] - s_h[sj][si] + D[globalIdx + 1] - D[globalIdx]) +
			-dt / (dx*(1 + s_kfU[sj][si - 1] * s_kfU[sj][si + 1])) * (s_kfU[sj][si + 1] * (s_U[sj][si + 1] - s_U[sj][si]) + s_kfU[sj][si - 1] * (s_U[sj][si] - s_U[sj][si - 1]))* s_U[sj][si] +
			(s_V[sj][si] > 0) * -dt /   ((1 + s_kfU[sj - 2][si])*dy) * s_kfU[sj - 1][si] * ((1 + 2 * s_kfU[sj - 2][si])*s_U[sj][si] - (1 + 3 * s_kfU[sj - 2][si])*s_U[sj - 1][si] + s_kfU[sj - 2][si] * s_U[sj - 2][si])  *(s_V[sj - 1][si] + s_V[sj - 1][si + 1]) / 2
			+ (s_V[sj][si] < 0) * -dt / ((1 + s_kfU[sj + 2][si])*dy) * s_kfU[sj + 1][si] * (-(1 + 2 * s_kfU[sj + 2][si])*s_U[sj][si] + (1 + 3 * s_kfU[sj + 2][si])*s_U[sj + 1][si] - s_kfU[sj + 2][si] * s_U[sj + 2][si]) * (s_V[sj][si] + s_V[sj][si + 1]) / 2);

	
		// update V

		Vtemp[globalIdx] = s_kfV[sj][si] * (s_V[sj][si] - g * dt / dy * (s_h[sj + 1][si] - s_h[sj][si] + D[globalIdx + n_d] - D[globalIdx])
			- dt / (dy*(1 + s_kfV[sj - 1][si] * s_kfV[sj + 1][si])) * (s_kfV[sj + 1][si] * (s_V[sj + 1][si] - s_V[sj][si]) + s_kfV[sj - 1][si] * (s_V[sj][si] - s_V[sj - 1][si]))* s_V[sj][si] +
			+(s_U[sj][si] > 0) * -dt / ((1 + s_kfV[sj][si - 2])*dx) * s_kfV[sj][si - 1] * ((1 + 2 * s_kfV[sj][si - 2])*s_V[sj][si] - (1 + 3 * s_kfV[sj][si - 2])*s_V[sj][si - 1] + s_kfV[sj][si - 2] * s_V[sj][si - 2])     *(s_U[sj + 1][si - 1] + s_U[sj][si - 1]) / 2 +
			+(s_U[sj][si] < 0) * -dt / ((1 + s_kfV[sj][si + 2])*dx) * s_kfV[sj][si + 1] * (-(1 + 2 * s_kfV[sj][si + 2])*s_V[sj][si] + (1 + 3 * s_kfV[sj][si + 2])*s_V[sj][si + 1] - s_kfV[sj][si + 2] * s_V[sj][si + 2])     *(s_U[sj + 1][si] + s_U[sj][si]) / 2);

	}

	__global__ void
		__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
		updateUorV(float *h, float *U, float *V, float* Utemp, float* Vtemp, float *D, float dt, bool tidal_case = false, float tideheight = 0, bool calcU = 0, bool calcBoth = 1)
	{



		__shared__       float   s_h[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4]; // 4-wide halo
		__shared__       float   s_U[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4]; // 4-wide halo
		__shared__       float   s_V[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4]; // 4-wide halo
		__shared__		__int8 s_kfU[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4];
		__shared__		__int8 s_kfV[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4];



		//int i = threadIdx.x;
		//int j = blockIdx.x*blockDim.y + threadIdx.y;
		unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
		unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int si = threadIdx.x + 2; // local i for shared memory access + halo offset
		unsigned int sj = threadIdx.y + 2; // local j for shared memory access
		unsigned int globalIdx = (j + 2) * n_d + i + 2;

		float s_hymin; float s_hxmin; float s_hx; float s_hy;


		s_U[sj][si] = U[globalIdx];
		s_V[sj][si] = V[globalIdx];
		s_h[sj][si] = h[globalIdx];

		//Borders
		if (threadIdx.y == 0 | threadIdx.y == 1) {

			/*if (blockIdx.x == 0 && blockIdx.y ==0) {
				Htemp[globalIdx - 2 * n_d] = 1.5;
				Htemp[globalIdx + n_d * BLOCK_SIZE_y] = 1.5;
			}*/

			s_h[sj - 2][si] = h[globalIdx - 2 * n_d];
			s_U[sj - 2][si] = U[globalIdx - 2 * n_d];
			s_V[sj - 2][si] = V[globalIdx - 2 * n_d];

			s_h[sj + BLOCK_SIZE_y][si] = h[globalIdx + n_d * (BLOCK_SIZE_y)];
			s_U[sj + BLOCK_SIZE_y][si] = U[globalIdx + n_d * (BLOCK_SIZE_y)];
			s_V[sj + BLOCK_SIZE_y][si] = V[globalIdx + n_d * (BLOCK_SIZE_y)];

			if (threadIdx.x == 0) {
				for (int offset = 1; offset <= 2; offset++) {
					s_h[sj - 2][si - offset] = h[globalIdx - 2 * n_d - offset];
					s_U[sj - 2][si - offset] = U[globalIdx - 2 * n_d - offset];
					s_V[sj - 2][si - offset] = V[globalIdx - 2 * n_d - offset];

					s_h[sj - 2][BLOCK_SIZE_x + offset + 1] = h[globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1];
					s_U[sj - 2][BLOCK_SIZE_x + offset + 1] = U[globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1];
					s_V[sj - 2][BLOCK_SIZE_x + offset + 1] = V[globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1];

					s_h[BLOCK_SIZE_y + sj][si - offset] = h[globalIdx + (BLOCK_SIZE_y ) * n_d - offset];
					s_U[BLOCK_SIZE_y + sj][si - offset] = U[globalIdx + (BLOCK_SIZE_y ) * n_d - offset];
					s_V[BLOCK_SIZE_y + sj][si - offset] = V[globalIdx + (BLOCK_SIZE_y ) * n_d - offset];

					s_h[BLOCK_SIZE_y + sj][BLOCK_SIZE_x + offset + 1] = h[globalIdx + (BLOCK_SIZE_y ) * n_d + BLOCK_SIZE_x + offset - 1];
					s_U[BLOCK_SIZE_y + sj][BLOCK_SIZE_x + offset + 1] = U[globalIdx + (BLOCK_SIZE_y ) * n_d + BLOCK_SIZE_x + offset - 1];
					s_V[BLOCK_SIZE_y + sj][BLOCK_SIZE_x + offset + 1] = V[globalIdx + (BLOCK_SIZE_y ) * n_d + BLOCK_SIZE_x + offset - 1];






				}
			}
		}

		if ((threadIdx.y == 2 | threadIdx.y == 3) /*& threadIdx.y < BLOCK_SIZE_y*/) {

			unsigned int jtemp = blockIdx.y*blockDim.y + threadIdx.x;
			unsigned int itemp = blockIdx.x*blockDim.x + threadIdx.y;
			unsigned int si2 = threadIdx.y; // local i for shared memory access + halo offset
			unsigned int sj2 = threadIdx.x + 2; // local j for shared memory access
			unsigned int globalIdx2 = (jtemp + 2)* n_d + itemp;

			s_h[sj2][si2 - 2] = h[globalIdx2 - 2];
			s_U[sj2][si2 - 2] = U[globalIdx2 - 2];
			s_V[sj2][si2 - 2] = V[globalIdx2 - 2];

			s_h[sj2][si2 + BLOCK_SIZE_x] = h[globalIdx2 + BLOCK_SIZE_x];/*BLOCK_SIZE_x*/
			s_U[sj2][si2 + BLOCK_SIZE_x] = U[globalIdx2 + BLOCK_SIZE_x];
			s_V[sj2][si2 + BLOCK_SIZE_x] = V[globalIdx2 + BLOCK_SIZE_x];



		}

		__syncthreads();

		// hxy and kfuv on the borders
		if (threadIdx.y == 0 | threadIdx.y == 1) {


			s_kfU[sj - 2][si] = 1 - (
				((s_U[sj - 2][si] > 0) *s_h[sj - 2][si]
					+ (s_U[sj - 2][si] < 0) *s_h[sj - 2][si + 1]
					+ (s_U[sj - 2][si] == 0) *fmaxf(s_h[sj - 2][si], s_h[sj - 2][si + 1]))
				< droogval);

			s_kfV[sj - 2][si] = 1 -
				(((s_V[sj - 2][si] > 0)* s_h[sj - 2][si]
					+ (s_V[sj - 2][si] < 0) * s_h[sj - 1][si]
					+ (s_V[sj - 2][si] == 0) * fmaxf(s_h[sj - 2][si], s_h[sj - 1][si]))
					< droogval);


			s_kfU[sj + BLOCK_SIZE_y][si] = 1 - (
				((s_U[sj + BLOCK_SIZE_y][si] > 0) *s_h[sj + BLOCK_SIZE_y][si] +
				(s_U[sj + BLOCK_SIZE_y][si] < 0) *s_h[sj + BLOCK_SIZE_y][si + 1] +
					(s_U[sj + BLOCK_SIZE_y][si] == 0) *fmaxf(s_h[sj + BLOCK_SIZE_y][si], s_h[sj + BLOCK_SIZE_y][si + 1])) < droogval);

			s_kfV[sj + BLOCK_SIZE_y][si] = 1 - (((blockIdx.y != gridDim.y - 1) ?
				((s_V[sj + BLOCK_SIZE_y][si] > 0)* s_h[sj + BLOCK_SIZE_y][si]
					+ (s_V[sj + BLOCK_SIZE_y][si] < 0) * h[globalIdx + n_d * (BLOCK_SIZE_y + 1)]
					+ (s_V[sj + BLOCK_SIZE_y][si] == 0) * fmaxf(h[globalIdx + n_d * (BLOCK_SIZE_y + 1)], s_h[sj + BLOCK_SIZE_y][si])) : s_h[sj + BLOCK_SIZE_y][si]) < droogval);

		}

		if ((threadIdx.y == 2 | threadIdx.y == 3) /*& threadIdx.y < BLOCK_SIZE_y*/) {

			unsigned int jtemp = blockIdx.y*blockDim.y + threadIdx.x;
			unsigned int itemp = blockIdx.x*blockDim.x + threadIdx.y;
			unsigned int si2 = threadIdx.y; // local i for shared memory access + halo offset
			unsigned int sj2 = threadIdx.x + 2; // local j for shared memory access
			unsigned int globalIdx2 = (jtemp + 2)* n_d + itemp;




			s_kfU[sj2][si2 - 2] = 1 - (((s_U[sj2][si2 - 2] > 0) *s_h[sj2][si2 - 2]
				+ (s_U[sj2][si2 - 2] < 0) *s_h[sj2][si2 - 1]
				+ (s_U[sj2][si2 - 2] == 0) *fmaxf(s_h[sj2][si2 - 1], s_h[sj2][si2 - 2])) < droogval);


			s_kfV[sj2][si2 - 2] = 1 - (((s_V[sj2][si2 - 2] > 0)  * s_h[sj2][si2 - 2]
				+ (s_V[sj2][si2 - 2] < 0)  * s_h[sj2 + 1][si2 - 2]
				+ (s_V[sj2][si2 - 2] == 0)* fmaxf(s_h[sj2][si2 - 2], s_h[sj2 + 1][si2 - 2])) < droogval);

			s_kfU[sj2][si2 + BLOCK_SIZE_x] = 1 - (((blockIdx.x != gridDim.x - 1) ?
				(s_U[sj2][si2 + BLOCK_SIZE_x] > 0) *s_h[sj2][si2 + BLOCK_SIZE_x]
				+ (s_U[sj2][si2 + BLOCK_SIZE_x] < 0) *h[globalIdx2 + BLOCK_SIZE_x + 1]
				+ (s_U[sj2][si2 + BLOCK_SIZE_x] == 0) *fmaxf(s_h[sj2][si2 + BLOCK_SIZE_x], h[globalIdx2 + BLOCK_SIZE_x + 1]) : s_h[sj2][si2 + BLOCK_SIZE_x]) < droogval);

			s_kfV[sj2][si2 + BLOCK_SIZE_x] = 1 - (((s_V[sj2][si2 + BLOCK_SIZE_x] > 0) * s_h[sj2][si2 + BLOCK_SIZE_x]
				+ (s_V[sj2][si2 + BLOCK_SIZE_x] < 0)  * s_h[sj2 + 1][si2 + BLOCK_SIZE_x]
				+ (s_V[sj2][si2 + BLOCK_SIZE_x] == 0) * fmaxf(s_h[sj2 + 1][si2 + BLOCK_SIZE_x], s_h[sj2][si2 + BLOCK_SIZE_x])) < droogval);

			if (tidal_case && blockIdx.x == 0) {
				s_h[sj2][si2 - 1] = Hstart - D[globalIdx2 - 1] + tideheight;
				s_kfU[sj2][si2 - 1] = 1;

				if (sj2 == 2 | sj2 == 3) {
					s_h[sj2 - 2][si2 - 1] = Hstart - D[globalIdx2 - 1] + tideheight;
					s_kfU[sj2 - 2][si2 - 1] = 1;

					s_h[sj2 + BLOCK_SIZE_y][si2 - 1] = Hstart - D[globalIdx2 - 1] + tideheight;
					s_kfU[sj2 + BLOCK_SIZE_y][si2 - 1] = 1;

				}
			}
		}

		s_hx = (s_U[sj][si] > 0) *s_h[sj][si] +
			(s_U[sj][si] < 0) *s_h[sj][si + 1] +
			(s_U[sj][si] == 0) *fmaxf(s_h[sj][si], s_h[sj][si + 1]);

		s_hy = (s_V[sj][si] > 0)* s_h[sj][si]
			+ (s_V[sj][si] < 0) * s_h[sj + 1][si]
			+ (s_V[sj][si] == 0) * fmaxf(s_h[sj + 1][si], s_h[sj][si]);

		__syncthreads();

		//wetting/drying
		s_kfU[sj][si] = 1 - (s_hx < droogval);
		s_kfV[sj][si] = 1 - (s_hy < droogval);


		__syncthreads();

		if (blockIdx.x == gridDim.x - 1 && sj == BLOCK_SIZE_x + 1) {
			s_kfU[si][sj] = 0;
		}
		if (blockIdx.y == gridDim.y - 1 && sj == BLOCK_SIZE_y + 1) {
			s_kfV[sj][si] = 0;
		}

		__syncthreads();

		if (calcBoth || calcU)
		{

			//update U 
			Utemp[globalIdx] = s_kfU[sj][si] * (s_U[sj][si] - g * dt / dx * (s_h[sj][si + 1] - s_h[sj][si] + D[globalIdx + 1] - D[globalIdx]) +
				-dt / (dx*(1 + s_kfU[sj][si - 1] * s_kfU[sj][si + 1])) * (s_kfU[sj][si + 1] * (s_U[sj][si + 1] - s_U[sj][si]) + s_kfU[sj][si - 1] * (s_U[sj][si] - s_U[sj][si - 1]))* s_U[sj][si] +
				(s_V[sj][si] > 0) * -dt / ((1 + s_kfU[sj - 2][si])*dy) * s_kfU[sj - 1][si] * ((1 + 2 * s_kfU[sj - 2][si])*s_U[sj][si] - (1 + 3 * s_kfU[sj - 2][si])*s_U[sj - 1][si] + s_kfU[sj - 2][si] * s_U[sj - 2][si])  *(s_V[sj - 1][si] + s_V[sj - 1][si + 1]) / 2
				+ (s_V[sj][si] < 0) * -dt / ((1 + s_kfU[sj + 2][si])*dy) * s_kfU[sj + 1][si] * (-(1 + 2 * s_kfU[sj + 2][si])*s_U[sj][si] + (1 + 3 * s_kfU[sj + 2][si])*s_U[sj + 1][si] - s_kfU[sj + 2][si] * s_U[sj + 2][si]) * (s_V[sj][si] + s_V[sj][si + 1]) / 2);
		}
		if (calcBoth || calcU == false) {
			// update V

			Vtemp[globalIdx] = s_kfV[sj][si] * (s_V[sj][si] - g * dt / dy * (s_h[sj + 1][si] - s_h[sj][si] + D[globalIdx + n_d] - D[globalIdx])
				- dt / (dy*(1 + s_kfV[sj - 1][si] * s_kfV[sj + 1][si])) * (s_kfV[sj + 1][si] * (s_V[sj + 1][si] - s_V[sj][si]) + s_kfV[sj - 1][si] * (s_V[sj][si] - s_V[sj - 1][si]))* s_V[sj][si] +
				+(s_U[sj][si] > 0) * -dt / ((1 + s_kfV[sj][si - 2])*dx) * s_kfV[sj][si - 1] * ((1 + 2 * s_kfV[sj][si - 2])*s_V[sj][si] - (1 + 3 * s_kfV[sj][si - 2])*s_V[sj][si - 1] + s_kfV[sj][si - 2] * s_V[sj][si - 2])     *(s_U[sj + 1][si - 1] + s_U[sj][si - 1]) / 2 +
				+(s_U[sj][si] < 0) * -dt / ((1 + s_kfV[sj][si + 2])*dx) * s_kfV[sj][si + 1] * (-(1 + 2 * s_kfV[sj][si + 2])*s_V[sj][si] + (1 + 3 * s_kfV[sj][si + 2])*s_V[sj][si + 1] - s_kfV[sj][si + 2] * s_V[sj][si + 2])     *(s_U[sj + 1][si] + s_U[sj][si]) / 2);
		}
		
	}

	__global__ void
		__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
		updateH(float *h, float *U, float *V, float*Htemp,  float* D,  float dt, bool tidal_case = false, float tideheight = 0)
	{



		__shared__       float   s_h[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4]; // 4-wide halo
		__shared__       float   s_U[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4]; // 4-wide halo
		__shared__       float   s_V[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4]; // 4-wide halo
		__shared__		__int8 s_kfU[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4];
		__shared__		__int8 s_kfV[BLOCK_SIZE_y + 4][BLOCK_SIZE_x + 4];



		//int i = threadIdx.x;
		//int j = blockIdx.x*blockDim.y + threadIdx.y;
		unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
		unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int si = threadIdx.x + 2; // local i for shared memory access + halo offset
		unsigned int sj = threadIdx.y + 2; // local j for shared memory access
		unsigned int globalIdx = (j + 2) * n_d + i + 2;

		float htemp;
		float s_hymin; float s_hxmin; float s_hx; float s_hy;


		s_U[sj][si] = U[globalIdx];
		s_V[sj][si] = V[globalIdx];
		s_h[sj][si] = h[globalIdx];

		 //Exchange Borders
		if (threadIdx.y == 0 | threadIdx.y == 1) {

			/*if (blockIdx.x == 0 && blockIdx.y ==0) {
				Htemp[globalIdx - 2 * n_d] = 1.5;
				Htemp[globalIdx + n_d * BLOCK_SIZE_y] = 1.5;
			}*/

			s_h[sj - 2][si] = h[globalIdx - 2 * n_d];
			s_U[sj - 2][si] = U[globalIdx - 2 * n_d];
			s_V[sj - 2][si] = V[globalIdx - 2 * n_d];

			s_h[sj + BLOCK_SIZE_y][si] = h[globalIdx + n_d * (BLOCK_SIZE_y)];
			s_U[sj + BLOCK_SIZE_y][si] = U[globalIdx + n_d * (BLOCK_SIZE_y)];
			s_V[sj + BLOCK_SIZE_y][si] = V[globalIdx + n_d * (BLOCK_SIZE_y)];

			if (threadIdx.x == 0) {
				for (int offset = 1; offset <= 2; offset++) {
					s_h[sj - 2][si - offset] = h[globalIdx - 2 * n_d - offset];
					s_U[sj - 2][si - offset] = U[globalIdx - 2 * n_d - offset];
					s_V[sj - 2][si - offset] = V[globalIdx - 2 * n_d - offset];

					s_h[sj - 2][BLOCK_SIZE_x + offset + 1] = h[globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1];
					s_U[sj - 2][BLOCK_SIZE_x + offset + 1] = U[globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1];
					s_V[sj - 2][BLOCK_SIZE_x + offset + 1] = V[globalIdx - 2 * n_d + BLOCK_SIZE_x + offset - 1];

					s_h[BLOCK_SIZE_y + sj][si - offset] = h[globalIdx + (BLOCK_SIZE_y ) * n_d - offset];
					s_U[BLOCK_SIZE_y + sj][si - offset] = U[globalIdx + (BLOCK_SIZE_y ) * n_d - offset];
					s_V[BLOCK_SIZE_y + sj][si - offset] = V[globalIdx + (BLOCK_SIZE_y ) * n_d - offset];

					s_h[BLOCK_SIZE_y + sj][BLOCK_SIZE_x + offset + 1] = h[globalIdx + (BLOCK_SIZE_y ) * n_d + BLOCK_SIZE_x + offset - 1];
					s_U[BLOCK_SIZE_y + sj][BLOCK_SIZE_x + offset + 1] = U[globalIdx + (BLOCK_SIZE_y ) * n_d + BLOCK_SIZE_x + offset - 1];
					s_V[BLOCK_SIZE_y + sj][BLOCK_SIZE_x + offset + 1] = V[globalIdx + (BLOCK_SIZE_y ) * n_d + BLOCK_SIZE_x + offset - 1];

				}
			}
		}

	
		if ((threadIdx.y == 2 | threadIdx.y == 3) /*& threadIdx.y < BLOCK_SIZE_y*/) {

			unsigned int jtemp = blockIdx.y*blockDim.y + threadIdx.x;
			unsigned int itemp = blockIdx.x*blockDim.x + threadIdx.y;
			unsigned int si2 = threadIdx.y; // local i for shared memory access + halo offset
			unsigned int sj2 = threadIdx.x + 2; // local j for shared memory access
			unsigned int globalIdx2 = (jtemp + 2)* n_d + itemp;

			s_h[sj2][si2 - 2] = h[globalIdx2 - 2];
			s_U[sj2][si2 - 2] = U[globalIdx2 - 2];
			s_V[sj2][si2 - 2] = V[globalIdx2 - 2];

			s_h[sj2][si2 + BLOCK_SIZE_x] = h[globalIdx2 + BLOCK_SIZE_x];/*BLOCK_SIZE_x*/
			s_U[sj2][si2 + BLOCK_SIZE_x] = U[globalIdx2 + BLOCK_SIZE_x];
			s_V[sj2][si2 + BLOCK_SIZE_x] = V[globalIdx2 + BLOCK_SIZE_x];

			if (tidal_case && blockIdx.x == 0) {
				  s_h[sj2][si2-1] = Hstart - D[globalIdx2-1] + tideheight;
				s_kfU[sj2][si2-1] = 1;

				if (sj2 == 2 | sj2 == 3) {
					  s_h[sj2 - 2][si2-1] = Hstart - D[globalIdx2-1] + tideheight;
					s_kfU[sj2 - 2][si2-1] = 1;

					  s_h[sj2 + BLOCK_SIZE_y][si2-1] = Hstart - D[globalIdx2-1] + tideheight;
					s_kfU[sj2 + BLOCK_SIZE_y][si2-1] = 1;

				}
			}

		}

		__syncthreads();

		// calculate hxy and kfuv on the borders
		if (threadIdx.y == 0 | threadIdx.y == 1) {


			s_kfU[sj - 2][si] = 1 - (
				((s_U[sj - 2][si] > 0) *s_h[sj - 2][si]
					+ (s_U[sj - 2][si] < 0) *s_h[sj - 2][si + 1]
					+ (s_U[sj - 2][si] == 0) *fmaxf(s_h[sj - 2][si], s_h[sj - 2][si + 1]))
				< droogval);

			s_kfV[sj - 2][si] = 1 -
				(((s_V[sj - 2][si] > 0)* s_h[sj - 2][si]
					+ (s_V[sj - 2][si] < 0) * s_h[sj - 1][si]
					+ (s_V[sj - 2][si] == 0) * fmaxf(s_h[sj - 2][si], s_h[sj - 1][si]))
					< droogval);


			s_kfU[sj + BLOCK_SIZE_y][si] = 1 - (
				((s_U[sj + BLOCK_SIZE_y][si] > 0) *s_h[sj + BLOCK_SIZE_y][si] +
				(s_U[sj + BLOCK_SIZE_y][si] < 0) *s_h[sj + BLOCK_SIZE_y][si + 1] +
					(s_U[sj + BLOCK_SIZE_y][si] == 0) *fmaxf(s_h[sj + BLOCK_SIZE_y][si], s_h[sj + BLOCK_SIZE_y][si + 1])) < droogval);

			s_kfV[sj + BLOCK_SIZE_y][si] = 1 - (((blockIdx.y != gridDim.y - 1) ?
				((s_V[sj + BLOCK_SIZE_y][si] > 0)* s_h[sj + BLOCK_SIZE_y][si]
					+ (s_V[sj + BLOCK_SIZE_y][si] < 0) * h[globalIdx + n_d * (BLOCK_SIZE_y + 1)]
					+ (s_V[sj + BLOCK_SIZE_y][si] == 0) * fmaxf(h[globalIdx + n_d * (BLOCK_SIZE_y + 1)], s_h[sj + BLOCK_SIZE_y][si])) : s_h[sj + BLOCK_SIZE_y][si]) < droogval);

		}


		if ((threadIdx.y == 2 | threadIdx.y == 3) /*& threadIdx.y < BLOCK_SIZE_y*/) {

			unsigned int jtemp = blockIdx.y*blockDim.y + threadIdx.x;
			unsigned int itemp = blockIdx.x*blockDim.x + threadIdx.y;
			unsigned int si2 = threadIdx.y; // local i for shared memory access + halo offset
			unsigned int sj2 = threadIdx.x + 2; // local j for shared memory access
			unsigned int globalIdx2 = (jtemp + 2)* n_d + itemp;




			s_kfU[sj2][si2 - 2] = 1 - (((s_U[sj2][si2 - 2] > 0) *s_h[sj2][si2 - 2]
				+ (s_U[sj2][si2 - 2] < 0) *s_h[sj2][si2 - 1]
				+ (s_U[sj2][si2 - 2] == 0) *fmaxf(s_h[sj2][si2 - 1], s_h[sj2][si2 - 2])) < droogval);


			s_kfV[sj2][si2 - 2] = 1 - (((s_V[sj2][si2 - 2] > 0)  * s_h[sj2][si2 - 2]
				+ (s_V[sj2][si2 - 2] < 0)  * s_h[sj2 + 1][si2 - 2]
				+ (s_V[sj2][si2 - 2] == 0)* fmaxf(s_h[sj2][si2 - 2], s_h[sj2 + 1][si2 - 2])) < droogval);

			s_kfU[sj2][si2 + BLOCK_SIZE_x] = 1 - (((blockIdx.x != gridDim.x - 1) ?
				(s_U[sj2][si2 + BLOCK_SIZE_x] > 0) *s_h[sj2][si2 + BLOCK_SIZE_x]
				+ (s_U[sj2][si2 + BLOCK_SIZE_x] < 0) *h[globalIdx2 + BLOCK_SIZE_x + 1]
				+ (s_U[sj2][si2 + BLOCK_SIZE_x] == 0) *fmaxf(s_h[sj2][si2 + BLOCK_SIZE_x], h[globalIdx2 + BLOCK_SIZE_x + 1]) : s_h[sj2][si2 + BLOCK_SIZE_x]) < droogval);

			s_kfV[sj2][si2 + BLOCK_SIZE_x] = 1 - (((s_V[sj2][si2 + BLOCK_SIZE_x] > 0) * s_h[sj2][si2 + BLOCK_SIZE_x]
				+ (s_V[sj2][si2 + BLOCK_SIZE_x] < 0)  * s_h[sj2 + 1][si2 + BLOCK_SIZE_x]
				+ (s_V[sj2][si2 + BLOCK_SIZE_x] == 0) * fmaxf(s_h[sj2 + 1][si2 + BLOCK_SIZE_x], s_h[sj2][si2 + BLOCK_SIZE_x])) < droogval);


		}

		s_hx = (s_U[sj][si] > 0) *s_h[sj][si] +
			(s_U[sj][si] < 0) *s_h[sj][si + 1] +
			(s_U[sj][si] == 0) *fmaxf(s_h[sj][si], s_h[sj][si + 1]);

		s_hy = (s_V[sj][si] > 0)* s_h[sj][si]
			+ (s_V[sj][si] < 0) * s_h[sj + 1][si]
			+ (s_V[sj][si] == 0) * fmaxf(s_h[sj + 1][si], s_h[sj][si]);


		s_hxmin = (s_U[sj][si - 1] > 0) *s_h[sj][si - 1] +
			(s_U[sj][si - 1] < 0) *s_h[sj][si] +
			(s_U[sj][si - 1] == 0) *fmaxf(s_h[sj][si], s_h[sj][si - 1]);

		s_hymin = (s_V[sj - 1][si] > 0)* s_h[sj - 1][si]
			+ (s_V[sj - 1][si] < 0) * s_h[sj][si]
			+ (s_V[sj - 1][si] == 0) * fmaxf(s_h[sj - 1][si], s_h[sj][si]);

		//wetting/drying
		s_kfU[sj][si] = 1 - (s_hx < droogval);
		s_kfV[sj][si] = 1 - (s_hy < droogval);

		__syncthreads();

		// special border case because of staggered grid
		if (blockIdx.x == gridDim.x - 1 && sj == BLOCK_SIZE_x + 1) {
			s_kfU[si][sj] = 0;
		}
		if (blockIdx.y == gridDim.y - 1 && sj == BLOCK_SIZE_y + 1) {
			s_kfV[sj][si] = 0;
		}

		// update h
		htemp = s_h[sj][si] +
			(1 - (1 - s_kfV[sj][si])*(1 - s_kfV[sj - 1][si])*(1 - s_kfU[sj][si])*(1 - s_kfU[sj][si - 1]))*(
				-dt / dx * (s_hx* s_U[sj][si] - s_hxmin * s_U[sj][si - 1])
				-dt / dy * (s_hy* s_V[sj][si] - s_hymin * s_V[sj - 1][si]));

		Htemp[globalIdx] = htemp;
		// s_U[sj][si];// s_kfU[sj][si];// s_U[sj][si];




	}

	__int8 *initializeBoolArray(int n) {
		__int8 *ptr = 0;
		//printf("Initializing bool array \n");
		checkCuda(cudaMalloc(&ptr, n * n * sizeof(__int8)));
		//checkCudaError("Malloc for matrix on device failed !");

		return ptr;

	}

	float *initializeFloatArray(int n){
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

	__global__	void fillarrays(float *H, __int8 *kfU, __int8 *kfV, unsigned int halo) {

		//int i = threadIdx.x;
		//int j = blockIdx.x*blockDim.y + threadIdx.y;
		int j = blockIdx.y*blockDim.y + threadIdx.y;
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		int globalIdx = (j + halo) * n_d + i + halo;
		//printf("globalidx = %d \n", globalIdx);
		H[globalIdx] = Hstart;
		kfU[globalIdx] = 1;
		kfV[globalIdx] = 1;
 		// U[globalIdx] = 0;
		// V[globalIdx] = 0;

		__syncthreads();

		if (i == 0) {
			#pragma unroll
			for (unsigned int q = 1; q <= halo; q++) {
				H[globalIdx - q] =				0;
				H[globalIdx + n_d - q-halo] =	0;
				kfU[globalIdx - q] =			0;
				kfU[globalIdx + n_d - q-halo] =	0;
				kfV[globalIdx - q] =			0;
				kfV[globalIdx + n_d - q-halo] =	0;				
			}
			kfU[globalIdx + n_d - 2*halo-1] = 0;


		}
		if (j == 0) {
			#pragma unroll
			for (unsigned int q = 1; q <= halo; q++) {
				H[globalIdx - q * n_d] =				0;				
				kfU[globalIdx - q * n_d] =				0;
				kfV[globalIdx - q * n_d] = 0;
				H[globalIdx + n_d * (n_d - q-halo)] =	0;
				kfU[globalIdx + n_d * (n_d - q-halo)] =	0;
				kfV[globalIdx + n_d * (n_d - q - halo)] = 0;
			}
			kfV[globalIdx + n_d * (n_d  - 2*halo-1) ] =0;
		}

	/*	if (i == 0) {
			
			H[globalIdx - 2] = 0;
			H[globalIdx - 2] = 0;
			H[globalIdx + n_d - 2] = 0;
			H[globalIdx + n_d - 1] = 0;

			kfU[globalIdx - 1] = 0;
			kfU[globalIdx - 2] = 0;

			kfU[globalIdx +n_d- 1] = 0;
			kfU[globalIdx +n_d- 2] = 0;

			kfV[globalIdx - 1] = 0;
			kfV[globalIdx - 2] = 0;

			kfV[globalIdx + n_d - 1] =1;
			kfV[globalIdx + n_d - 2] =1;
			kfV[globalIdx + n_d - 3] =1;
			
		}
		if (j == 0) {
			H[globalIdx - n_d]				= 0;
			H[globalIdx - 2*n_d]			= 0;
			H[globalIdx + n_d * (n_d - 2)]  = 0;
			H[globalIdx + n_d * (n_d - 1)]  = 0;

			kfU[globalIdx - n_d] = 0;
			kfU[globalIdx - 2*n_d] = 0;

			kfU[globalIdx + n_d *(n_d- 1)] = 1;
			kfU[globalIdx + n_d *(n_d- 2)] = 1;
			kfU[globalIdx + n_d *(n_d- 3)] = 1;
			
		}
		*/

	}

	__global__	void addarrays(float *A, float*B) {

		
		int j = blockIdx.y*blockDim.y + threadIdx.y;
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		int globalIdx = (j+2 ) * n_d + i+2 ;
		
		A[globalIdx] += B[globalIdx];

		
		

	}

	__global__	void addarrays(float *A, float*B,int sign) {


		int j = blockIdx.y*blockDim.y + threadIdx.y;
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		int globalIdx = (j + 2) * n_d + i + 2;

		A[globalIdx] = A[globalIdx] + sign*B[globalIdx];

	}

	__global__ void Waterdrop(float *H, float height, int width, float step, unsigned int halo=1,bool droplet = false) {


		int j = blockIdx.y*blockDim.y + threadIdx.y ;
		int i = blockIdx.x*blockDim.x + threadIdx.x ;
		int globalIdx;
		if (droplet) {
			//globalIdx = (j + halo + droppar * (n_d - width)) * n_d + i + halo + droppar * (n_d - width);
			globalIdx = (j + halo + (int)n_d/3) * n_d + i + halo + (int)n_d/3;
		}
		else {
		globalIdx=	(j + halo)* n_d + i + halo;
		}
		float x = -1 + i * step;
		float y = -1 + j * step;


		float add =  height *expf(-5 * (x*x + y * y));
		

		H[globalIdx] = H[globalIdx] + add;

		__syncthreads();


		//printf("globalIdx: %d x: %.3f y: %.3f ", globalIdx, x, y);
		//printf("Idx: %d x: %d y: %d D: %d ", globalIdx, x , y ,  D);
		//printf("x: %.2f y: %.2f D: %.2f \n ",  x , y ,  D);
		//printf("i: %d j: %d \n" ,i ,j);
		//printf("D: %.3f \n" ,D);	
		//float height = 1.5*Hstart;
		//int width = (int)(n-2)/2;
		//for (float i = -1; i < 1; i = i + 2 / (width - 1));
		//[x, y] = ndgrid(-1:(2 / (width - 1)) : 1);
		//D = height * exp(-5 * (x. ^ 2 + y. ^ 2));
		//w = size(D, 1);
		//i = ceil(rand*(n - w)) + (1:w);
		//j = ceil(rand*(n - w)) + (1:w);
		//H(i, j) = H(i, j) + (1 + 4 * rand) / 5 * D;*/
	}

	__global__ void Tidal_bath(float *H, float *D, unsigned int ghost, bool implicit) {


		unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
		unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int globalIdx = (j + ghost / 2)* n_d + i + ghost / 2;
		int slopesteps = floorf(400 / dx);
		float dh = 4.5 / slopesteps;
		if (implicit = false) {
		if (i <= slopesteps) {
			D[globalIdx] = i * dh;
			H[globalIdx] = fmaxf(H[globalIdx] - i * dh, 0);
		}
		else {
			D[globalIdx] = 4.5;
			H[globalIdx] = fmaxf(H[globalIdx] - 4.5, 0);
		}
	}
	else {
	if (i <= slopesteps) {
		D[globalIdx] = i * dh;
		H[globalIdx] = fmaxf(H[globalIdx]  , i * dh);
	}
	else {
		D[globalIdx] = 4.5;
		H[globalIdx] = fmaxf(H[globalIdx]  , 4.5);
	}

}

	}

	void addarrayscpu(float *A, float *B, int size, int sign = 1) {
				
			#pragma unroll
			for (int i = 0; i < size; i++) {
				A[i] = A[i] + sign* B[i];
			}
		
	}

void copyConstants(){

    checkCuda(cudaMemcpyToSymbol(n_d, &n, sizeof(int), 0, cudaMemcpyHostToDevice)); //grid size
	checkCuda(cudaMemcpyToSymbol(g, &g_h, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(dx, &dx_h, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(dy, &dy_h, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(cf, &cf_h, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(Hstart, &Hstart_h, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(droppar, &droppar_h, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(droogval, &droogval_h, sizeof(float), 0, cudaMemcpyHostToDevice));
	//checkCuda(cudaMemcpyToSymbol(offset, &offset_h, sizeof(float), 0, cudaMemcpyHostToDevice));
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

void initializeWaterdrop(float *H, unsigned int ghost) {

	int width = (int)floor(((float)n - ghost) / 3);
	dim3 gridSizeDrop(ceil((float)width / 32), ceil((float)width / 32));
	dim3 blockSizeDrop(fmin(32,width),fmin(32,width));
	float height = .5*Hstart_h;
	float step = 2 / (float(width)-1);
	
	Waterdrop << <gridSizeDrop, blockSizeDrop >> > (H, height, width, step,ghost/2,true);
	
	CudaCheckError();
	
}

void initializeBathymetry( float *H, float* D,  unsigned int ghost, bool tide_case = false,bool implicit = false) {

	if (tide_case==false) {
		int width = (int)floor(((float)n - ghost));
		dim3 gridSizeDrop(ceil((float)width / 32), ceil((float)width / 32));
		dim3 blockSizeDrop(fmin(32, width), fmin(32, width));
		float height = (Hstart_h + offset);
		float step = 2 / (float(width) - 1);

		Waterdrop << <gridSizeDrop, blockSizeDrop >> > (D, height, width, step, ghost / 2);
		CudaCheckError();
		if (implicit==false) {
			Waterdrop << <gridSizeDrop, blockSizeDrop >> > (H, -height, width, step, ghost / 2);
			CudaCheckError();
		}
	}
	else
	{
		dim3 gridSize((n - ghost) / BLOCK_SIZE_x, (n - ghost) / BLOCK_SIZE_y);
		dim3 blockSize(BLOCK_SIZE_x, BLOCK_SIZE_y);
		Tidal_bath << <gridSize, blockSize >> > (H,D,ghost,implicit);

	}
	/*cudaArray* D;
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
		cudaChannelFormatKindFloat);
	checkCuda(cudaMallocArray(&D, &channelDesc, n*sizeof(float), n*sizeof(float)));
	
	int width = (int)floor(((float)n - ghost));
	dim3 gridSizeDrop(ceil((float)width / 32), ceil((float)width / 32));
	dim3 blockSizeDrop(fmin(32, width), fmin(32, width));
	float height = Hstart_h+offset ;
	float step = 2 / (float(width) - 1);

	Waterdrop << <gridSizeDrop, blockSizeDrop >> > (Dinput, height, width, step);

	CudaCheckError();
	checkCuda(cudaMemcpy(D, Dinput, n*n * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCuda(cudaBindTextureToArray(tex, D, channelDesc));*/
	//cudaBindTexture2D(NULL,tex, D,&channelDesc,n*sizeof(float),n*sizeof(float),0);
	//return D;
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

void showVector(const char *name,  int n,  float *a = 0, int* b = 0)
{




	long x;
		if (a > 0){

			for (x = 0; x < n; x++) {
				//printf("%s[%02ld][%02ld]=%6.2f  ", name, y, x, a[y*n + x]);
				if (a[  x] == 0 || a[ x] == 1)
				{
					printf("%d  ", (int)a[ x]);
				}
				else {
					printf("%6.2f  ", a[ x]);
				}
			}
	printf("\n");
}
		else if (b > 0) {
			for (x = 0; x < n; x++) {
				//printf("%s[%02ld][%02ld]=%6.2f  ", name, y, x, a[y*n + x]);
				
					printf("%d  ", b[ x]);
				
			}
			printf("\n");
		}

}

void showMatrix(const char* name, __int8* a, int n, int m)
{

	long x, y;

	for (y = 0; y < m; y++)
	{
		for (x = 0; x < n; x++) {
			//printf("%s[%02ld][%02ld]=%6.2f  ", name, y, x, a[y*n + x]);
			if (a[y * n + x] == 0 || a[y * n + x] == 1)
			{
				printf("%d  ", (int)a[y * n + x]);
			}
			else {
				printf("%d  ", (int)a[y * n + x]);
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

void copyfromCudaMatrix(__int8* h_a, __int8* d_a, int n, int m)
{
	//printf("Copying result back... ");
	checkCuda(cudaMemcpy(h_a, d_a, n * m * sizeof(__int8), cudaMemcpyDeviceToHost));
	//printf("success! \n");
	//checkCudaError("Matrix copy from device failed !");
}

void printfile(float* H_h, float*D_h=0) {

	int i = 0, x = 0, y = 0;
	ofstream myfile;
	myfile.open("example1.txt");
	int ntemp;
	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
		{
			//   myfile << i;
			if (D_h == 0)
				myfile << " " << H_h[i*n + j];// +D_h[i*n + j];
			else
			myfile << " " << H_h[i*n + j] + D_h[i*n + j];

			// myfile << " " << y << endl;
			// i++;
		  //   x = x + 2;
			// y = x + 1;
		}
		myfile << " " << endl;
	}

	myfile.close();
	
}

void printfile(float* H_h, int n) {

	int i = 0, x = 0, y = 0;
	ofstream myfile;
	myfile.open("example1.txt");
	int ntemp;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
		{
			//   myfile << i;
		
				myfile << " " << H_h[i*n + j] ;

			// myfile << " " << y << endl;
			// i++;
		  //   x = x + 2;
			// y = x + 1;
		}
		myfile << " " << endl;
	}

	myfile.close();

}

void printbench() {

	//\begin{ table }[]
	//\begin{ tabular }{ | l | l | l | l | l | l | l | }
	//\hline
	//N & 100x100 1 4 & 100x100 2 2 & 200x200 1 4 & 200x200 2 2 & 400x400 1 4 & 400x400 2 2 \\ \hline
	//wtime       & 0.062807    & 0.067490    & 0.383046    & 0.383043    & 2.755771    & 2.701833    \\ \hline
	//wtime\_ex   & 0.000746    & 0.004638    & 0.002001    & 0.007347    & 0.007880    & 0.019175    \\ \hline
	//wtime\_comp & 0.043053    & 0.049204    & 0.314177    & 0.320867    & 2.526069    & 2.468069    \\ \hline
	//wtime\_gc   & 0.003254    & 0.006273    & 0.004763    & 0.008953    & 0.033084    & 0.032092    \\ \hline
	//wtime\_idle & 0.015754    & 0.007375    & 0.062105    & 0.045876    & 0.188738    & 0.182497    \\ \hline
	//\end{ tabular }
	//\end{ table }

	int i = 0, x = 0, y = 0;
	ofstream myfile;
	myfile.open("Latex.txt");




//	myfile << "\\begin{ table }[]" << endl;// +D_h[i*n + j];
//	myfile << "\\begin{ tabular }{ | l | l | l | l | l | l | l | }" << endl;
//	myfile << "N & 100x100 1 4 & 100x100 2 2 & 200x200 1 4 & 200x200 2 2 & 400x400 1 4 & 400x400 2 2 \\\\ \\hline" << endl;
//	myfile << "wtime       & 0.062807    & 0.067490    & 0.383046    & 0.383043    & 2.755771    & 2.701833    \\\\ \\hline" << endl;
//	myfile << "wtime\_ex   & 0.000746    & 0.004638    & 0.002001    & 0.007347    & 0.007880    & 0.019175    \\\\ \\hline" << endl;
//	myfile << "wtime\_comp & 0.043053    & 0.049204    & 0.314177    & 0.320867    & 2.526069    & 2.468069    \\\\ \\hline" << endl;
//	myfile << "wtime\_gc   & 0.003254    & 0.006273    & 0.004763    & 0.008953    & 0.033084    & 0.032092    \\\\ \\hline" << endl;
//	myfile << "wtime\_idle & 0.015754    & 0.007375    & 0.062105    & 0.045876    & 0.188738    & 0.182497    \\\\ \\hline" << endl;
//	myfile << "\begin{\end{ tabular }" << endl;];
//	myfile << "\begin{\end{ table }" << endl;
////	
//// myfile << " " << y << endl;
//// i++;
////   x = x + 2;
//  // y = x + 1;
//
//	myfile << " " << endl;

	myfile << "\\begin{table}[]" << endl;// +D_h[i*n + j];
	myfile << "\\begin{tabular}{|l|";
	for (int k = 0; k < size(ns); k++) {
		myfile << "l|";
		
	}
	myfile << "}" << endl;
	myfile << "\\hline" << endl;
	myfile << "N ";
	for (int k = 0; k < size(ns); k++) {
		myfile << " & " << 96*ns[k]+4 ;

		
	}

	myfile << " \\\\ \\hline" << endl;
	
		for (int j = 0; j < 2 + size(threads); j++) {
			if (j < 2)
				myfile << "GPU" << j + 1 ;
			else
				myfile << "CPU" << threads[j-2] ;

			for (int k = 0; k < size(ns); k++) {
				myfile << " " << " & " <<  benchmarkresult[k][j] ;

			}
			myfile << " \\\\ \\hline" << endl;
		
		}
	myfile << "\\end{tabular}" << endl;
	myfile << "\\end{table}" << endl;

	myfile.close();

	myfile.open("Matlab.txt");
	for (int k = 0; k < size(ns); k++) {
		myfile << " " << 96 * ns[k] + 4 ;


	}
	myfile << endl;
	for (int j = 0; j < 2 + size(threads); j++) {
		
		for (int k = 0; k < size(ns); k++) {
			myfile << " " << benchmarkresult[k][j];

		}
		myfile << endl;

	}
	myfile.close();
}

float runprogramsplit(int iter, bool doublesplit = true)
{
	unsigned int ghost = 4;

	n = n + ghost;

	float tideheight = 0;
	float t = 0;
	if (tidal_case) {
		dx_h = 600.0 / (n);
		dy_h = dx_h;
		Hstart_h = 3;

	}
	else
	{
		dx_h = 1;
		dy_h = 1;
	}

	float *H_h = 0;
	float *D_h = 0;
	if (benchmark == false) {
	cudaMallocHost(&H_h, n * n * sizeof(float));
	cudaMallocHost(&D_h, n * n * sizeof(float));
}
	dim3 gridSize((n - ghost) / (BLOCK_SIZE_x), (n - ghost) / BLOCK_SIZE_y);
	dim3 blockSize(BLOCK_SIZE_x, BLOCK_SIZE_y);

	
		// Print device and precision
		cudaDeviceProp prop;
		checkCuda(cudaGetDeviceProperties(&prop, 0));
		printf("\nDevice Name: %s\n", prop.name);
		printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);
		printf("Max Shared Memory per Block: %d bytes \n", prop.sharedMemPerBlock);
		printf("Shared Memory per SM: %d bytes \n", prop.sharedMemPerMultiprocessor);
		printf("SM count: %d \n", prop.multiProcessorCount);
		printf("Max threads per SM: %d \n", prop.maxThreadsPerMultiProcessor);
		printf("Max threads per block: %d \n", prop.maxThreadsPerBlock);
		printf("block size: %d \n", BLOCK_SIZE_x*BLOCK_SIZE_y);
		printf("Max registers per thread: %d \n", prop.regsPerMultiprocessor / (prop.maxThreadsPerMultiProcessor / (BLOCK_SIZE_x*BLOCK_SIZE_y)) / (BLOCK_SIZE_x*BLOCK_SIZE_y));
		
		int blockmem = ((BLOCK_SIZE_y + ghost)*(BLOCK_SIZE_x + ghost)*(3 * sizeof(float) + 2 * sizeof(__int8)));
		printf("Block shared memory usage: %d bytes \n", blockmem);
		//check if blocks fit in shared memory:
		if (blockmem > prop.sharedMemPerBlock) {
			throw "Block size too large!! \n";
		}

		if ((prop.maxThreadsPerMultiProcessor / MIN_BLOCKS_PER_MP)*blockmem < 16 * pow(2, 10))
		{
			checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
			printf("Configured for 48kb of L1 cache \n");

		}
		printf("n = %d \n", n);
		if (n * n * (3 * sizeof(float) + 2 * sizeof(bool)) > prop.totalGlobalMem)
		{
			throw "Device out of memory!! max size = %d ", sqrt(prop.totalGlobalMem / (3 * sizeof(float) + 2 * sizeof(__int8)));
		}
	
	float *H = initializeFloatArray(n);
	float *U = initializeFloatArray(n);
	float *V = initializeFloatArray(n);
	float *Htemp = initializeFloatArray(n);
	float *Utemp = initializeFloatArray(n);
	float *Vtemp = initializeFloatArray(n);
	float *D = initializeFloatArray(n);
	float *plotvar = initializeFloatArray(n);
	__int8 *kfU = initializeBoolArray(n);
	__int8 *kfV = initializeBoolArray(n);

	copyConstants();

	printf("filling arrays... ");
	fillarrays << <gridSize, blockSize >> > (H, kfU, kfV, ghost / 2);
	CudaCheckError();
	printf("success! \n");





	/*__int8* kfU_h = 0;
	cudaMallocHost(&kfU_h, n * n * sizeof(__int8));
	copyfromCudaMatrix(kfU_h, kfV, n, n);
	showMatrix("H", kfU_h, n, n);*/


	printf("initializing water drop...  ");
	if (tidal_case == false) {
		initializeWaterdrop(H, ghost);
	}
	//if (showarray) {
	//	copyfromCudaMatrix(H_h, H, n, n);
	//	showMatrix("H", H_h, n, n);
	//}

	initializeBathymetry(H, D, ghost, tidal_case);


	/*if (showarray) {

		copyfromCudaMatrix(H_h, D, n, n);
		showMatrix("D", H_h, n, n);

	}*/
	printf("success! \n \n");

	if (showarray) {
		//addarrays << <gridSize, blockSize >> > (H, D);
		copyfromCudaMatrix(H_h, D, n, n);
		showMatrix("H", H_h, n, n);
		//addarrays << <gridSize, blockSize >> > (H, D,-1);
	}
	/*if (n - ghost == 32) {
		__int8* kfU_h = 0;
		cudaMallocHost(&kfU_h, n * n * sizeof(__int8));
		copyfromCudaMatrix(kfU_h, kfV, n, n);
		showMatrix("H", kfU_h, n, n);
	}*/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	if(benchmark == false)
	copyfromCudaMatrix(D_h, D, n, n);
	//printfile(H_h, D_h);
	//Sleep(1000)

	printf("Starting explicit method, %d time steps...  ", iter);
	for (int q = 1; q <= iter; q++) {

		if (tidal_case) {

			tideheight = 2 * sin(t * 2 * M_PI / (12 * 3600));
			dt = safety * dx_h / (2 * sqrt(2 * g_h*(Hstart_h + tideheight)));
			t = t + dt;
			if (t >= 3600 * 24) {
				printf("t = %f, iter = %d, dt = %f \n", t, q, dt);
				break;
			}
		}

		if (doublesplit) {
		updateUorV << <gridSize, blockSize >> > (H, U, V, Utemp, Vtemp, D, dt, tidal_case, tideheight);


		checkCuda(cudaMemcpyAsync(U, Utemp, n * n * sizeof(float), cudaMemcpyDeviceToDevice));
		checkCuda(cudaMemcpyAsync(V, Vtemp, n * n * sizeof(float), cudaMemcpyDeviceToDevice));
	}
	else {
		updateUorV << <gridSize, blockSize >> > (H, U, V, Utemp, Vtemp, D, dt, tidal_case, tideheight, true,false);
		checkCuda(cudaMemcpyAsync(U, Utemp, n * n * sizeof(float), cudaMemcpyDeviceToDevice));

		updateUorV << <gridSize, blockSize >> > (H, U, V, Utemp, Vtemp, D, dt, tidal_case, tideheight, false,false);
		checkCuda(cudaMemcpyAsync(V, Vtemp, n * n * sizeof(float), cudaMemcpyDeviceToDevice));
	}
		updateH << <gridSize, blockSize >> > (H, U, V, Htemp,  D, dt, tidal_case, tideheight);
		checkCuda(cudaMemcpyAsync(H, Htemp, n * n * sizeof(float), cudaMemcpyDeviceToDevice));

		/*H = Htemp;
		U = Utemp;
		V = Vtemp;

		float* Htemp = initializeFloatArray(n);
		float* Utemp = initializeFloatArray(n);
		float* Vtemp = initializeFloatArray(n);
		*/
			if ((q - 1) % plotstep  == 0 && realtimeplot && benchmark==false) {
				//addarrays << <gridSize, blockSize >> > (H, D);
		//if(realtimeplot){
		copyfromCudaMatrix(H_h, Htemp, n, n);
		printfile(H_h, D_h);
		//addarrays << <gridSize, blockSize >> > (H, D, -1);

		//printfile(H_h, D_h);
		printf("plot, tide = %f, dt = %f \n", tideheight, dt);
		Sleep(100);
			}

	}
	if (benchmark == false) {
		copyfromCudaMatrix(H_h, Htemp, n, n);
		//addarrays << <gridSize, blockSize >> > (H, D, -1);
		printfile(H_h, D_h);
	}


	cudaDeviceSynchronize();




	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	CudaCheckError();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("success! \n ");
	printf("update time: %.7f seconds \n", milliseconds / 1000);

	if (showarray&&benchmark==false) {
		/*copyfromCudaMatrix(kfU_h, kfV, n, n);
		showMatrix("H", kfU_h, n, n);*/

		copyfromCudaMatrix(H_h, H, n, n);
		showMatrix("H", H_h, n, n);
		addarrays << <gridSize, blockSize >> > (H, D);
		copyfromCudaMatrix(H_h, H, n, n);
		showMatrix("H+D", H_h, n, n);
		copyfromCudaMatrix(H_h, H, n, n);
		//printfile(H_h, D_h);

	}

	cudaDeviceReset();
	n = n - ghost;

	return milliseconds / 1000;
}

float runprogramimplicit(int iter)
{
	unsigned int ghost = 4;

	
	n = n + ghost;
	//n = 36;
	//n = 68;

	float tideheight = 0;
	float t = 0;
	if (tidal_case) {
		dx_h = 600.0 / (n);
		dy_h = dx_h;
		Hstart_h = 3;

	}
	else
	{
		dx_h = 1;
		dy_h = 1;
	}

	if (alpha <= .5)
		dt = implicitfactor * dt;

	float *H_h = 0;
	float *D_h = 0;
	if (benchmark == false) {
		cudaMallocHost(&H_h, n * n * sizeof(float));
		cudaMallocHost(&D_h, n * n * sizeof(float));
	}
	dim3 gridSize((n - ghost) / (BLOCK_SIZE_x), (n - ghost) / BLOCK_SIZE_y);
	dim3 blockSize(BLOCK_SIZE_x, BLOCK_SIZE_y);


	// Print device and precision
	cudaDeviceProp prop;
	checkCuda(cudaGetDeviceProperties(&prop, 0));
	printf("\nDevice Name: %s\n", prop.name);
	printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);
	printf("Max Shared Memory per Block: %d bytes \n", prop.sharedMemPerBlock);
	printf("Shared Memory per SM: %d bytes \n", prop.sharedMemPerMultiprocessor);
	printf("SM count: %d \n", prop.multiProcessorCount);
	printf("Max threads per SM: %d \n", prop.maxThreadsPerMultiProcessor);
	printf("Max threads per block: %d \n", prop.maxThreadsPerBlock);
	printf("block size: %d \n", BLOCK_SIZE_x*BLOCK_SIZE_y);
	printf("Max registers per thread: %d \n", prop.regsPerMultiprocessor / (prop.maxThreadsPerMultiProcessor / (BLOCK_SIZE_x*BLOCK_SIZE_y)) / (BLOCK_SIZE_x*BLOCK_SIZE_y));

	int blockmem = ((BLOCK_SIZE_y + ghost)*(BLOCK_SIZE_x + ghost)*(3 * sizeof(float) + 2 * sizeof(__int8)));
	printf("Block shared memory usage: %d bytes \n", blockmem);
	//check if blocks fit in shared memory:
	if (blockmem > prop.sharedMemPerBlock) {
		throw "Block size too large!! \n";
	}

	if ((prop.maxThreadsPerMultiProcessor / MIN_BLOCKS_PER_MP)*blockmem < 16 * pow(2, 10))
	{
		checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		printf("Configured for 48kb of L1 cache \n");

	}
	printf("n = %d \n", n);
	if (n * n * (3 * sizeof(float) + 2 * sizeof(bool)) > prop.totalGlobalMem)
	{
		throw "Device out of memory!! max size = %d ", sqrt(prop.totalGlobalMem / (3 * sizeof(float) + 2 * sizeof(__int8)));
	}

	float *H = initializeFloatArray(n);
	float *U = initializeFloatArray(n);
	float *V = initializeFloatArray(n);
	float *Htemp = initializeFloatArray(n-ghost);
	float *Utemp = initializeFloatArray(n);
	float *Vtemp = initializeFloatArray(n);
	float *D = initializeFloatArray(n);
	float *plotvar = initializeFloatArray(n);
	__int8 *kfU = initializeBoolArray(n);
	__int8 *kfV = initializeBoolArray(n);
	 float *Hx = initializeFloatArray(n);
	 float *Hy = initializeFloatArray(n);
	

	copyConstants();

	printf("filling arrays... ");
	fillarrays << <gridSize, blockSize >> > (H, kfU, kfV, ghost / 2);
	CudaCheckError();
	printf("success! \n");





	/*__int8* kfU_h = 0;
	cudaMallocHost(&kfU_h, n * n * sizeof(__int8));
	copyfromCudaMatrix(kfU_h, kfV, n, n);
	showMatrix("H", kfU_h, n, n);*/

	
	printf("initializing water drop...  ");
	if (tidal_case == false) {
		initializeWaterdrop(H, ghost);
	}
	//if (showarray) {
	//	copyfromCudaMatrix(H_h, H, n, n);
	//	showMatrix("H", H_h, n, n);
	//}

	initializeBathymetry(H, D, ghost, tidal_case,true);


	/*if (showarray) {

		copyfromCudaMatrix(H_h, D, n, n);
		showMatrix("D", H_h, n, n);

	}*/
	printf("success! \n \n");

	if (showarray) {
		//addarrays << <gridSize, blockSize >> > (H, D);
		copyfromCudaMatrix(H_h, D, n, n);
		//showMatrix("H", H_h, n, n);
		//addarrays << <gridSize, blockSize >> > (H, D,-1);
	}




	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);

	checkCudaErrors(cublasStatus);

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);

	checkCudaErrors(cusparseStatus);

	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);

	checkCudaErrors(cusparseStatus);

	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);


	// initialize matrix pointers
	int*   row_ptr = 0;
	int*   col_ind = 0;
	float* val = 0;
	float*  rowfact = 0;
	float* d_p = 0;
	float* d_Ax = 0;
	float* RHS = initializeFloatArray(n - ghost);

	float numels = (n - ghost - 2)*((n - ghost - 2) * 5 + 16) + 12;
	float diffact = (1 - alpha)*g_h*dt*dt / (dx_h*dx_h);

	//Allocate memory for sparse matrix vectors
	checkCuda(cudaMalloc(&row_ptr, ((n - ghost) * (n - ghost) + 1) * sizeof(int)));
	checkCuda(cudaMalloc(&col_ind, numels * sizeof(int)));
	checkCuda(cudaMalloc(&val, numels * sizeof(float)));
	checkCuda(cudaMalloc(&rowfact, (n - ghost) * (n - ghost) * sizeof(float)));
	checkCuda(cudaMalloc((void **)&d_p, (n - ghost) * (n - ghost) * sizeof(float)));
	checkCuda(cudaMalloc((void **)&d_Ax, (n - ghost) * (n - ghost) * sizeof(float)));

	//set up grid for matrix generation
	dim3 matgrid((n - ghost)*(n - ghost) / 512);
	dim3 matblock(512);
	
	if (benchmark == false)
		copyfromCudaMatrix(D_h, D, n, n);
	//printfile(H_h, D_h);
	//Sleep(1000)

	//start timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	float* showvar4 = 0;
	cudaMallocHost(&showvar4, ((n - ghost) * (n - ghost)) * sizeof(float));

	cudaFuncSetCacheConfig(genMatrix, cudaFuncCachePreferL1);

	printf("Starting Implicit method, %d time steps...  ", iter);
	//main loop
	for (int q = 1; q <= iter; q++) {

		if (tidal_case) {
			tideheight = 2 * sin(t * 2 * M_PI / (12 * 3600));
			dt = safety * dx_h / (2 * sqrt(2 * g_h*(Hstart_h + tideheight)));
			if (alpha <= .5)
				dt = implicitfactor * dt;
			t = t + dt;
			if (t >= 3600 * 24) {
				printf("t = %f, iter = %d, dt = %f \n", t, q, dt);
				break;
			}
		}


		//updateUorV << <gridSize, blockSize >> > (H, U, V, Utemp, Vtemp, D, dt, tidal_case, tideheight);

		calcUVexpl << <gridSize, blockSize >> > (H, U, V, Utemp, Vtemp, kfU, kfV, D, Hx, Hy, dt, alpha, tidal_case, tideheight);


		genMatrix << <matgrid, matblock >> > (row_ptr, col_ind, val, Hx, Hy, kfU, kfV, rowfact, diffact);

		calcRHS << <gridSize, blockSize >> > (Utemp, Vtemp, H, Hx, Hy, rowfact, dt, RHS);
		
		CG(col_ind, row_ptr, val, Htemp, RHS, d_p, d_Ax, (n - ghost)*(n - ghost), numels, cusparseHandle, cublasHandle, cublasStatus, descr);
	
		
		  postCG << <gridSize, blockSize >> > (Htemp, H, Utemp, U, Vtemp, V, kfU, kfV, alpha, dt);

		  if ((q - 1) % plotstep == 0 && realtimeplot && benchmark == false && showarray) {
			  printf("CG RESULT \n");
			  float* showvar4 = 0;
			  cudaMallocHost(&showvar4, ((n ) * (n )) * sizeof(float));
			  checkCuda(cudaMemcpy(showvar4, H, ((n ) * (n )) * sizeof(float), cudaMemcpyDeviceToHost));
			  // showVector("row_ptr", (n - ghost) * (n - ghost), showvar4);
			  printf("\n \n \n");

			  showMatrix("H", showvar4, n , n );
		  }

		 
		
		if ((q - 1) % plotstep == 0 && realtimeplot && benchmark == false) {

			
			//checkCuda(cudaMemcpy(showvar4, Hx, ((n - ghost) * (n - ghost)) * sizeof(float), cudaMemcpyDeviceToHost));

			copyfromCudaMatrix(H_h, H, n, n);
			printfile(H_h,n);
			printf("plot, tide = %f, dt = %f \n", tideheight, dt);			

			Sleep(200);
		}

	}



	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);


	/*
		cudaFree(d_col);
		cudaFree(d_row);
		cudaFree(d_val);
		cudaFree(d_x);
		cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ax);*/

	if (benchmark == false) {
		copyfromCudaMatrix(H_h, Htemp, n, n);
		//addarrays << <gridSize, blockSize >> > (H, D, -1);
		printfile(H_h, D_h);
	}
	CudaCheckError();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("Implicit, iterations: %d \n", iter);
	printf("success! \n");
	printf("computation time: %.7f seconds \n", milliseconds / 1000);

	if (showarray&&benchmark == false) {
		/*copyfromCudaMatrix(kfU_h, kfV, n, n);
		showMatrix("H", kfU_h, n, n);*/

		copyfromCudaMatrix(H_h, H, n, n);
		showMatrix("H", H_h, n, n);
		addarrays << <gridSize, blockSize >> > (H, D);
		copyfromCudaMatrix(H_h, H, n, n);
		showMatrix("H+D", H_h, n, n);
		copyfromCudaMatrix(H_h, H, n, n);
		//printfile(H_h, D_h);

	}

	cudaDeviceReset();
	n = n - ghost;

	return milliseconds / 1000;
}

double runprogramCPUnew(int iter, int maxthreads=0) {

	int ghost = 4;

	n = n + ghost;

	dim3 gridSize((n - ghost) / (BLOCK_SIZE_x), (n - ghost) / BLOCK_SIZE_y);
	dim3 blockSize(BLOCK_SIZE_x, BLOCK_SIZE_y);

	
	if (tidal_case) {
		dx_h = 600.0 / (n);
		dy_h = dx_h;
		Hstart_h = 3;

	}
	else
	{
		dx_h = 1;
		dy_h = 1;
	}

	float *H = initializeFloatArray(n);
	float *U = initializeFloatArray(n);
	float *V = initializeFloatArray(n);
	float *D = initializeFloatArray(n);
	__int8 *kfU = initializeBoolArray(n);
	__int8 *kfV = initializeBoolArray(n);
	float *plotvar = initializeFloatArray(n);
	copyConstants();

	//printf("filling arrays... ");
	fillarrays << <gridSize, blockSize >> > (H, kfU, kfV, ghost / 2);
	CudaCheckError();
	//printf("success! \n");

	
	//printf("initializing water drop... ");

	if (tidal_case == false) {
		initializeWaterdrop(H, ghost);
	}
	initializeBathymetry(H, D, ghost,tidal_case);
	
	//printf("success! \n");

	

	float* H_h = 0;
	float* V_h = 0;
	float* U_h = 0;
	float* D_h = 0;
	__int8* kfU_h = 0;
	__int8* kfV_h = 0;

	cudaMallocHost(&H_h, n * n * sizeof(float));
	cudaMallocHost(&U_h, n * n * sizeof(float));
	cudaMallocHost(&V_h, n * n * sizeof(float));
	cudaMallocHost(&D_h  , n * n * sizeof(float));
	cudaMallocHost(&kfU_h, n * n * sizeof(__int8));
	cudaMallocHost(&kfV_h, n * n * sizeof(__int8));

	copyfromCudaMatrix(H_h, H, n, n);
	copyfromCudaMatrix(U_h, U, n, n);
	copyfromCudaMatrix(V_h, V, n, n);
	copyfromCudaMatrix(D_h, D, n, n);
	copyfromCudaMatrix(kfU_h, kfU, n, n);
	copyfromCudaMatrix(kfV_h, kfV, n, n);


	unsigned numthreads;
	if (maxthreads > 0)
	{
		numthreads = maxthreads;
	}
	else
		numthreads = std::thread::hardware_concurrency();
	if ((n - ghost) % numthreads != 0)
		while ((n - ghost) % numthreads != 0)
			--numthreads;



	int* pointer = NULL;
	int count = 0;
	pointer = &count;

	printf("\n number of threads: %u : ", numthreads);
	/*if (numthreads > 1) {*/
		std::thread* t = NULL;
		t = new std::thread[numthreads];
		//auto cb = new cbar::cyclicbarrier(numthreads); //syncthreads(numthreads);	
		auto timestart = high_resolution_clock::now();
		for (unsigned int i = 0; i < numthreads; i++) {
			//(float* h, float* U, float* V, float *D, __int8* kfU, __int8* kfV, float dt, int tid, int numthreads, int iter, int ghost, cbar::cyclicbarrier* cb)
			t[i] = std::thread(updatecputhreadnew, H_h, U_h, V_h, D_h, kfU_h, kfV_h, dt, i, numthreads, iter, ghost, tidal_case, pointer);
		}

		for (unsigned int i = 0; i < numthreads; i++) {
			t[i].join();
		}

		auto timestop = high_resolution_clock::now();

		auto duration = duration_cast<microseconds>(timestop - timestart);
	/*}
	if (numthreads == 1)
		updatecputhreadsingle(H_h, U_h, V_h, D_h, kfU_h, kfV_h, dt, iter, ghost, tidal_case);
*/
		printf("computation time: %lf seconds \n", (double)duration.count()/1000000);

	//	printfile(H_h, D_h);
	





	if (showarray) {
		showMatrix("kfU", kfU_h, n, n);
		showMatrix("kfV", kfV_h, n, n);
		showMatrix("H", H_h, n, n);
		addarrayscpu(H_h, D_h,(n+ghost)*(n+ghost));
		showMatrix("H+D", H_h, n, n);
		
		
		
	}
	/*if (n - ghost == 32) {
		__int8* kfU_h = 0;
		cudaMallocHost(&kfU_h, n * n * sizeof(__int8));
		copyfromCudaMatrix(kfU_h, kfU, n, n);
		showMatrix("H", kfU_h, n, n);
	}
	if (n - ghost == 32) {
		__int8* kfU_h = 0;
		cudaMallocHost(&kfU_h, n * n * sizeof(__int8));
		copyfromCudaMatrix(kfU_h, kfV, n, n);
		showMatrix("H", kfU_h, n, n);
	}*/

	cudaDeviceReset();
	n = n - ghost;

	return (double)duration.count()/1000000;
}

int main()
{
	
	
	
	int averages = 1;
	int ncounter = 0;
	int threadcounter = 0;
	if (benchmark) {
		benchmarkresult = new double*[size(ns)];
		// Create a row for every pointer 
		for (int k = 0; k <= size(ns); k++)
		{
			benchmarkresult[k] = new double[2 + size(threads)];
		}
		averages = benchaverages;
		
			for (int var2 = 0; var2 < size(ns); var2++) {
				for (int var1 = 0; var1 < 2 + size(threads); var1++) {
				benchmarkresult[var2][var1] = 0;
			}
		}
	}


	

		for (int average = 1; average <= averages; average++) {
			//int maxthreads = 1;
			ncounter = 0;
			for (const int ni : ns) {

				n = 3 * 32 * ni;
				//n = 64;
				// n = 32;
			//	 main2();
			//	runprogram(iter);

				if (benchmark) {
					benchmarkresult[ncounter][0] += runprogramsplit(iter) / averages;
					//benchmarkresult[ncounter][1] += runprogramsplit(iter ,false) / averages;
					benchmarkresult[ncounter][1] += runprogramimplicit(iter/implicitfactor) / averages;
					//	printf("n = %d GPU: \n", n);
				}
				else
				{
					float junkvar = runprogramimplicit(iter);//runprogramsplit(iter,false);
				}
				//	runprogramCPUnew(iter,10);
					////runprogram(iter);
					//runprogrambenchmark(iter,1);
					//printf("\n");
					////printf("nobool: ");
					//runprogrambenchmark(iter,2);
					//printf("\n");
					////printf("borders: ");
					//runprogrambenchmark(iter,3);

					////RunProgramCPU(iter,0);
					////if (ni<20)

				
				if (ni<cpubenchthresh) {
					printf("\n CPU benchmarks: \n \n");
				threadcounter = 0;
				for (int maxthreads : threads) {

					benchmarkresult[ncounter][2 + threadcounter] += runprogramCPUnew(iter, maxthreads) / averages;
					threadcounter++;
				}
				printf(" \n");
			}

			
				ncounter++;
			}
		}
		if (benchmark) {
			printbench();
			printf("benchmark file written!");
		}
		cudaDeviceReset();
	return 0;
}  