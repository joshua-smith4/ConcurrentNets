#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

__global__ void addVec(int *a, size_t pitch_a, int *b, size_t pitch_b, int *c, size_t pitch_c, int Nrow, int Ncol)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < Ncol && j < Nrow)
  {
    int* aElem = (int*)((char*)a + j * pitch_a) + i;
    int* bElem = (int*)((char*)b + j * pitch_b) + i;
    int* cElem = (int*)((char*)c + j * pitch_c) + i;
    *cElem = *aElem + * bElem;
  }
}

int main() {
  const unsigned Nrow = 10;
  const unsigned Ncol = 20;
  // std::random_device rd;
  // std::mt19937 gen(rd());
  // std::uniform_int_distribution<> dis(1, 100);
  int a[Nrow][Ncol];
  for (int i = 0; i < Nrow; ++i)
  {
    for(int j = 0; j < Ncol; ++j)
    {
      a[i][j] = rand() % 100;
    }
  }

  int b[Nrow][Ncol];
  for (int i = 0; i < Nrow; ++i)
  {
    for(int j = 0; j < Ncol; ++j)
    {
      b[i][j] = rand() % 100;
    }
  }

  // std::generate(vec_a.begin(), vec_a.end(), [&](){ return dis(gen); });
  // std::generate(vec_b.begin(), vec_b.end(), [&](){ return dis(gen); });
  cudaFree(0);
  std::cout << "finished generating numbers\n";
  int *d_a, *d_b, *d_c;
  std::size_t pitch_a, pitch_b, pitch_c;
  cudaMallocPitch(&d_a, &pitch_a, sizeof(int)*Nrow, Ncol);
  cudaMallocPitch(&d_b, &pitch_b, sizeof(int)*Nrow, Ncol);
  cudaMallocPitch(&d_c, &pitch_c, sizeof(int)*Nrow, Ncol);

  cudaMemcpy2D(d_a, pitch_a, a, sizeof(int)*Nrow, sizeof(int)*Nrow, Ncol, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_b, pitch_b, b, sizeof(int)*Nrow, sizeof(int)*Nrow, Ncol, cudaMemcpyHostToDevice);

  const unsigned THREADS_PER_BLOCK_X = 5;
  const unsigned THREADS_PER_BLOCK_Y = 5;
  const unsigned NUM_BLOCKS_X =
      (Ncol + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X;
  const unsigned NUM_BLOCKS_Y =
      (Nrow + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y;

  dim3 gridDim(NUM_BLOCKS_X, NUM_BLOCKS_Y);
  dim3 blockDim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
  addVec<<<gridDim, blockDim>>>(d_a, pitch_a, d_b, pitch_b, d_c, pitch_c, Nrow, Ncol);

  int c[Nrow][Ncol];

  cudaMemcpy2D(c, sizeof(int)*Nrow, d_c, pitch_c, sizeof(int) * Nrow, Ncol, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  for (int i = 0; i < Nrow; ++i)
  {
    for(int j = 0; j < Ncol; ++j)
    {
      std::cout << "(" << a[i][j] << "+" << b[i][j] << "=" << c[i][j] << ") ";
    }
    std::cout << std::endl;
  }
  // for (int i = 0; i < N; ++i)
  // {
  //   for(int j = 0; j < N; ++j)
  //   {
  //     std::cout << b[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // for (int i = 0; i < N; ++i)
  // {
  //   for(int j = 0; j < N; ++j)
  //   {
  //     std::cout << c[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  for (int i = 0; i < Nrow; ++i)
  {
    for(int j = 0; j < Ncol; ++j)
    {
      assert(c[i][j] == a[i][j] + b[i][j]);
    }
  }
  std::cout << "Passed!\n";
  return 0;
}
