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
  int y = blockIdx.y * blockDim.y + threadIdx.y + minY;
  int x = blockIdx.x * blockDim.x + threadIdx.x + minX;
  int yrange = maxY - minY + 1;
  int xrange = maxX - minX + 1;
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


__global__ void histCalc_noshared(unsigned* tilesWithinRoutingRegion, IdType* colorTiles, unsigned subNetCount, int minY, int maxY, int minX, int maxX, unsigned num_concurrency_bins)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int yrange = maxY - minY + 1;
  int xrange = maxX - minX + 1;
  if (y <= yrange && x <= xrange)
  {
    unsigned binIndex = (gridDim.x*blockDim.x*y+x) % num_concurrency_bins;
    if(colorTiles[y*yrange+x] != NOID)
    {
      atomicAdd(&tilesWithinRoutingRegion[binIndex*subNetCount+colorTiles[y*yrange+x]], 1);
    }
  }
}

__global__ void sumHist_noshared(unsigned* tilesWithinRoutingRegion, unsigned* returnVal, unsigned subNetCount, unsigned num_concurrency_bins)
{
  int i = threadIdx.x;
  if(i < subNetCount)
  {
    for(int j = 0; j < num_concurrency_bins; ++j)
    {
      returnVal[i] += tilesWithinRoutingRegion[j*subNetCount+i];
    }
  }
}
