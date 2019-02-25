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
__global__ void histCalc_noshared(unsigned** tilesWithinRoutingRegion, uint2* a, uint2* b, unsigned subNetCount, int minY, int maxY, int minX, int maxX, unsigned num_concurrency_bins)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if ((y <= maxY - minY + 1) && (x <= maxX - minX + 1))
  {
    unsigned binIndex = (gridDim.x*blockDim.x*y+x) % num_concurrency_bins;
    for(int i = 0; i < subNetCount; ++i)
    {
      if (a[i].x <= x && b[i].x >= x && a[i].y <= y && b[i].y >= y)
      {
        atomicAdd(&tilesWithinRoutingRegion[binIndex][i], 1);
        break;
      }
    }
  }
}

__global__ void sumHist_noshared(unsigned** tilesWithinRoutingRegion, unsigned* returnVal, unsigned subNetCount, unsigned num_concurrency_bins)
{
  int i = threadIdx.x;
  if(i < subNetCount)
  {
    for(int j = 0; j < num_concurrency_bins; ++j)
    {
      returnVal[i] += tilesWithinRoutingRegion[j][i];
    }
  }
}
