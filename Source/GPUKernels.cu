/*
 * GPUKernels.cu
 *
 *  Created on: Oct 19, 2010
 *      Author: yiding
 */

#include <cuda_runtime.h>
#include <math.h>
#include <cuda.h>

#define NOID   0xFFFFFFFF

#define CUDA_MAJOR_VER 1
#define CUDA_MINOR_VER 3


typedef unsigned int CoordType;
typedef unsigned int IdType;
typedef unsigned short CapType;
typedef unsigned int SizeType;
typedef float CostType;

// template <unsigned N>
// struct ParallelHistAccumulator
// {
// public:
//   unsigned* bins[N];
//   unsigned numBins;
//   inline __device__ void init(unsigned numberOfBins)
//   {
//     numThreadBins = num;
//     for(auto i = 0u; i < N; ++i)
//     {
//       bins[i] = new unsigned[numBins];
//     }
//   }
//   inline __device__
//   inline __device__ void cleanup()
//   {
//     for(auto i = 0u; i < N; ++i)
//     {
//       delete[] bins[i];
//     }
//   }
//   inline __device__ unsigned incrementBin(unsigned bin)
//   {
//     return 0u;
//   }
//   inline __device__ unsigned* getHist()
//   {
//
//   }
// };

//Define CUDA Kernels in this file
__global__ void colorTiles_noshared(IdType** colorTiles, uint2* a, uint2* b, unsigned subNetCount, int yTiles, int xTiles, int minY, int maxY, int minX, int maxX)
{

}

__global__ void countSubnets_noshared(IdType** colorTiles, unsigned* tilesWithinRoutingRegion, int minY, int maxY, int minX, int maxX)
{

}
