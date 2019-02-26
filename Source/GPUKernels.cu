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

//Define CUDA Kernels in this file
__global__ void colorTiles_noshared(unsigned* colorTiles, size_t pitchColorTiles, uint2* a, uint2* b, unsigned subNetCount, int minY, int maxY, int minX, int maxX)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x + minX;
  int y = blockIdx.y * blockDim.y + threadIdx.y + minY;
  if (y >= minY && y <= maxY && x >= minX && x <= maxX)
  {
    IdType* elem = (IdType*)((char*)colorTiles + y * pitchColorTiles) + x;
    for(int i = 0; i < subNetCount; ++i)
    {
      if (a[i].x <= x && b[i].x >= x && a[i].y <= y && b[i].y >= y)
      {
        *elem = i;
        break;
      }
    }
  }
}


__global__ void histCalc_noshared(unsigned* tilesWithinRoutingRegion, size_t pitchTiles, IdType* colorTiles, size_t pitchColor, unsigned subNetCount, int minY, int maxY, int minX, int maxX, unsigned num_concurrency_bins)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x + minX;
  int y = blockIdx.y * blockDim.y + threadIdx.y + minY;
  if (y >= minY && y <= maxY && x >= minX && x <= maxX)
  {
    IdType* elem = (IdType*)((char*)colorTiles + y * pitchColor) + x;
    if(*elem != NOID)
    {
      unsigned* tileElem = (unsigned*)((char*)tilesWithinRoutingRegion + ((blockDim.x*gridDim.x*(blockIdx.y*blockDim.y+threadIdx.y)+threadIdx.x) % num_concurrency_bins) * pitchTiles) + (*elem);
      atomicAdd(tileElem, 1);
    }
  }
}

__global__ void sumHist_noshared(unsigned* tilesWithinRoutingRegion, size_t pitchTiles, unsigned* returnVal, unsigned subNetCount, unsigned num_concurrency_bins)
{
  int i = threadIdx.x;
  if(i < subNetCount)
  {
    for(int j = 0; j < num_concurrency_bins; ++j)
    {
      unsigned* elem = (unsigned*)((char*)tilesWithinRoutingRegion + j * pitchTiles) + i;
      returnVal[i] += *elem;
    }
  }
}
